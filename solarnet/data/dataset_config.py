import os
from pathlib import Path
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule

from solarnet.data import BaseDataset, SDOBenchmarkDataModule, SDOBenchmarkDataset, SDODataset, SDODatasetDataModule
from solarnet.utils.scaling import log_min_max_scale


def datamodule_from_config(parameters: dict) -> LightningDataModule:

    from solarnet.utils.target import flux_to_class_builder

    name = parameters["data"]["name"]
    path = Path(parameters["data"]["path"])

    regression = parameters["data"]['targets'] == "regression"
    reg_tt = log_min_max_scale
    target_transform = reg_tt if regression else flux_to_class_builder(parameters["data"]['targets']['classes'])

    if name == "sdo-dataset":
        datamodule = SDODatasetDataModule(
            path,
            target_transform=target_transform,
            batch_size=parameters['trainer']['batch_size'],
            resize=parameters['data']['size'],
            num_workers=0 if os.name == 'nt' else 8,  # Windows supports only 1, Linux supports more
        )
    elif name == "sdo-benchmark":
        datamodule = SDOBenchmarkDataModule(
            path,
            batch_size=parameters['trainer']['batch_size'],
            validation_size=parameters['data']['validation_size'],
            channel=parameters['data']['channel'],
            resize=parameters['data']['size'],
            seed=parameters['seed'],
            num_workers=0 if os.name == 'nt' else 8,  # Windows supports only 1, Linux supports more
            target_transform=target_transform,
            time_steps=parameters['data']['time_steps'],
        )
    else:
        raise ValueError("Dataset not defined")

    return datamodule


def dataset_from_config(params: dict, split: str, transform: Optional[Callable] = None) -> BaseDataset:

    from solarnet.utils.target import flux_to_class_builder

    data = params["data"]

    name = data["name"]
    path = Path(data["path"])

    regression = data['targets'] == "regression"
    reg_tt = log_min_max_scale
    target_transform = reg_tt if regression else flux_to_class_builder(data['targets']['classes'])

    if name == "sdo-dataset":
        dataset = SDODataset(
            path / f"sdo-dataset-{split}.csv",
            transform=transform,
            target_transform=target_transform,
        )
    elif name == "sdo-benchmark":
        if split == "val":
            raise ValueError("val split not supported for this dataset")
        elif split == "train":
            split = "training"
        dataset = SDOBenchmarkDataset(
            path / split / "meta_data.csv",
            path / split,
            channel=data["channel"],
            transform=transform,
            target_transform=target_transform,
            time_steps=data["time_steps"],
        )
    else:
        raise ValueError("Dataset not defined")

    return dataset