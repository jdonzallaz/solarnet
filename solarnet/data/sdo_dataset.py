from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SDODataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        dataset_root_path: Path = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Dataset class for the sdo-dataset.
        The accepted format is the one of the sdo-dataset, using the special utility for dataset creation.
        It needs a csv file which contains the metadata and a folder which contains the actual data.

        If dataset_path is a file, it is used as the csv-metadata file. Its parent is used as the root folder for the
            data.
        If dataset_path is a folder, "sdo-dataset.csv" is searched in this folder and used as csv-metadata file.
            dataset_path is used as root folder for the data.
        If dataset_root_path is given, it is used as root folder for the data (instead of the rules above).

        :param dataset_path: Path to file (.csv) or folder to find the (meta)data like explained above.
        :param dataset_root_path: Path to root folder for the data (optional, if not given, the folder is inferred from
            dataset_path)
        :param transform: callable with transforms to apply on the (tensor) data.
        :param target_transform: callable with transforms to apply on the target.
        """

        # Check paths and find csv file and root path
        if not dataset_path.exists():
            raise FileNotFoundError(f"The given dataset_path {dataset_path} does not exist.")
        if dataset_path.is_file() and dataset_path.suffix != ".csv":
            raise AttributeError("dataset_path must be a .csv file or a folder")
        if dataset_root_path is not None and not dataset_root_path.is_dir():
            raise AttributeError("dataset_root_path must be a folder")

        if dataset_path.is_file():
            self.csv_file = dataset_path
            self.dataset_folder = dataset_path.parent
        else:
            self.csv_file = dataset_path / "sdo-dataset.csv"
            self.dataset_folder = dataset_path

        if dataset_root_path is not None:
            self.dataset_folder = dataset_root_path

        if not self.csv_file.exists():
            raise FileNotFoundError(f"The metadata file {self.csv_file} was not found.")
        elif not self.dataset_folder.exists():
            raise FileNotFoundError(f"The data folder {self.dataset_folder} was not found.")

        # Properties
        self.transform = transform
        self.target_transform = target_transform

        # Load dataset metadata
        self.dataset = pd.read_csv(self.csv_file, parse_dates=["datetime"])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        # Get a sample from the dataset metadata
        sample: pd.Series = self.dataset.iloc[index]

        # Get the target
        target = sample["peak_flux"]

        # Get the paths to images
        paths = sample.filter(like='path_', axis=0).values

        # Load a numpy array from .npz and convert to pytorch tensor, for each image
        tensors = [
            torch.from_numpy(
                np.load(self.dataset_folder / path)["x"]
            ) for path in paths
        ]

        # Stack tensors (images) together to make a multi-dim tensor (like an image with several channels)
        tensor = torch.stack(tensors)

        # Transform the data and target
        if self.transform:
            tensor = self.transform(tensor)
        if self.target_transform:
            target = self.target_transform(target)

        return tensor, target
