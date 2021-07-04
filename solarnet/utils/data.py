from collections import Counter
from pathlib import Path
import random
from typing import Optional, TypeVar

import pytorch_lightning as pl
import torch
from torch import default_generator
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset

from solarnet.data.dataset_utils import BaseDataset
from solarnet.utils.plots import plot_image_grid

T = TypeVar("T")


def train_test_split(dataset: Dataset[T], test_size: float = 0.1, seed: Optional[int] = None):
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be in [0, 1].")

    test_size = int(test_size * len(dataset))
    train_size = len(dataset) - test_size

    generator = default_generator if seed is None else torch.Generator().manual_seed(seed)

    return torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)


def data_info(datamodule: pl.LightningDataModule, parameters: dict, save_path: Path = None) -> dict:
    """
    Summary of the data used for training/testing.
    Gives: class-balance in each split, shape of the data, range of the data, split sizes, plot of examples.
    Datasets in datamodule must have y() method to access targets (like BaseDataset).
    """

    info = {}

    # Prepare datasets
    datasets = {}
    try:
        datasets["train"] = datamodule.train_dataloader().dataset
    except Exception:
        pass
    try:
        datasets["val"] = datamodule.val_dataloader().dataset
    except Exception:
        pass
    try:
        datasets["test"] = datamodule.test_dataloader().dataset
    except Exception:
        pass

    # Analyze class-balance
    def class_balance(dataset: BaseDataset):
        if isinstance(dataset, BaseDataset):
            y = dataset.y()
        elif isinstance(dataset, Subset):
            ds = dataset.dataset
            if not isinstance(ds, BaseDataset):
                raise AttributeError("dataset must be a BaseDataset or a Subset of a BaseDataset.")
            y = ds.y(dataset.indices)
        else:
            raise AttributeError("dataset must be a BaseDataset or a Subset of a BaseDataset.")

        counter = Counter(y)
        class_names = [list(c.keys())[0] for c in parameters["data"]["targets"]["classes"]]
        return {class_names[i]: counter[i] for i in range(len(class_names))}

    if "data" in parameters and "targets" in parameters["data"] and "classes" in parameters["data"]["targets"]:
        info["class-balance"] = {}
        for ds_name, ds in datasets.items():
            info["class-balance"][ds_name] = class_balance(ds)

    # Shape of data
    info["shape"] = str(datamodule.size())

    # Range
    if "train" in datasets:
        if isinstance(datasets["train"][0][0], tuple):
            t = torch.cat([datasets["train"][0][0][0], datasets["train"][1][0][0], datasets["train"][2][0][0]])
        else:
            t = torch.cat([datasets["train"][0][0], datasets["train"][1][0], datasets["train"][2][0]])
        info["tensor-data"] = {
            "min": t.min().item(),
            "max": t.max().item(),
            "mean": t.mean().item(),
            "std": t.std().item(),
        }

    # Sizes of splits
    info["set-sizes"] = {}
    for ds_name, ds in datasets.items():
        info["set-sizes"][ds_name] = len(ds)

    # Examples of each batch
    def plot_batch(ds_name: str, ds: Dataset, n_images: int = 32):
        if isinstance(ds[0][0], tuple):
            images = [ds[i][0][0].squeeze() for i in random.sample(range(len(ds)), n_images)]
        else:
            images = [ds[i][0].squeeze() for i in random.sample(range(len(ds)), n_images)]
        plot_image_grid(images, max_images=n_images, columns=8, save_path=save_path / f"data-examples-{ds_name}.png")

    if save_path is not None:
        for ds_name, ds in datasets.items():
            plot_batch(ds_name, ds)

    return info
