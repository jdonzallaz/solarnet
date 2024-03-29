from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from solarnet.data.dataset_utils import BaseDataset
from solarnet.data.transforms import sdo_dataset_normalize


class SDODataset(BaseDataset):
    TARGET_COLUMN = "peak_flux"
    DATETIME_COLUMN = "datetime"
    PATH_COLUMN_PREFIX = "path_"

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
        self.dataset = pd.read_csv(self.csv_file, parse_dates=[self.DATETIME_COLUMN])

        if len(self.dataset) == 0:
            raise RuntimeError(f"Dataset in {self.csv_file} is empty.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        # Get a sample from the dataset metadata
        sample: pd.Series = self.dataset.iloc[index]

        # Get the target
        target = sample[self.TARGET_COLUMN]

        # Get the paths to images
        paths = sample.filter(like=self.PATH_COLUMN_PREFIX, axis=0).values

        # Load a numpy array from .npz and convert to pytorch tensor, for each image
        tensors = [torch.from_numpy(np.load(self.dataset_folder / path)["x"]) for path in paths]

        # Stack tensors (images) together to make a multi-dim tensor (like an image with several channels)
        tensor = torch.stack(tensors)

        # Transform the data and target
        if self.transform:
            tensor = self.transform(tensor)
        if self.target_transform:
            target = self.target_transform(target)

        return tensor, target

    def y(self, indices: Optional[Sequence[int]] = None) -> list:
        if indices is None:
            targets = self.dataset[self.TARGET_COLUMN].tolist()
        else:
            targets = self.dataset.iloc[indices][self.TARGET_COLUMN].tolist()

        if self.target_transform is not None:
            targets = list(map(self.target_transform, targets))

        return targets


class SDODatasetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Path,
        csv_filename_prefix: str = "sdo-dataset",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        channel: Union[str, int] = 171,
        resize: int = 512,
    ):
        """
        Dataset class for the sdo-dataset.
        The accepted format is the one of the sdo-dataset, using the special utility for dataset creation.
        It needs a path to the dataset folder. The dataset must also contains 3 csv-files for metadata:
            sdo-dataset-train.csv, sdo-dataset-val.csv, sdo-dataset-test.csv
        By default, it applies the standard normalization process of sdo-dataset, depending on the given channel and resize
        The transform can be overridden by giving a custom transform.

        :param dataset_path: Path to the dataset folder (with metadata files in it)
        :param csv_filename_prefix: Filename prefix of the csv files. [-train,-val,-test] will be appended to it to find
                                    the csv files.
        :param transform: Optional transform for the data. If None, the standard normalization process of sdo-dataset is
                          applied.
        :param target_transform: Optional transform for the targets
        :param batch_size: batch size for the dataloader
        :param num_workers: num_workers for the dataloaders. Changing it is not advised on Windows.
                            It is encouraged on Unix systems (~ 4x number of GPUs).
        :param channel: The channel of the sdo data to process. Can be ignored if a custom transform is given.
        :param resize: Target size to which the image should be resized. Can be ignored if a custom transform is given.
        """

        super().__init__()
        self.dataset_path = dataset_path
        self.csv_filename_prefix = csv_filename_prefix
        self.transform = sdo_dataset_normalize(channel, resize) if transform is None else transform
        self.target_transform = target_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    @property
    def num_samples(self) -> int:
        return len(self.dataset_train)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.dataset_train = SDODataset(
                self.dataset_path / f"{self.csv_filename_prefix}-train.csv",
                transform=self.transform,
                target_transform=self.target_transform,
            )
            self.dataset_val = SDODataset(
                self.dataset_path / f"{self.csv_filename_prefix}-val.csv",
                transform=self.transform,
                target_transform=self.target_transform,
            )
            sample = self.dataset_train[0][0]

        if stage == "test" or stage is None:
            path = self.dataset_path / f"{self.csv_filename_prefix}-test.csv"
            if path.exists():
                self.dataset_test = SDODataset(
                    path,
                    transform=self.transform,
                    target_transform=self.target_transform,
                )
                sample = self.dataset_test[0][0]
            elif stage == "test":
                raise RuntimeError("Cannot prepare stage=test of datamodule with no test dataset.")

        self.dims = tuple((sample[0] if isinstance(sample, tuple) else sample).shape)

    def train_dataloader(self):
        if not self.has_setup_fit:
            raise RuntimeError("The SDODatasetDataModule setup has not been called.")
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            # Otherwise batch of 1 can cause error in batch_norm layers
            drop_last=True if len(self.dataset_train) % self.batch_size == 1 else False,
        )

    def val_dataloader(self):
        if not self.has_setup_fit:
            raise RuntimeError("The SDODatasetDataModule setup has not been called.")
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        if not self.has_setup_test:
            raise RuntimeError("The SDODatasetDataModule setup has not been called.")
        if not hasattr(self, "dataset_test"):
            raise RuntimeError("No test dataset in this datamodule.")
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
