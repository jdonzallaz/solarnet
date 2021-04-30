import logging
from pathlib import Path

import numpy as np
from torchvision.transforms import transforms

from solarnet.data import BaseDataset, SDOBenchmarkDataset, SDODataset, dataset_mean_std, dataset_min_max
from solarnet.utils.scaling import log_min_max_scale
from solarnet.utils.target import compute_class_weight, flux_to_class_builder

logger = logging.getLogger(__name__)


def _stats_dataset(dataset: BaseDataset):
    """
    Compute and print stats about datasets.
    """

    print('Size:', len(dataset))
    print('Shape:', dataset[0][0].shape)

    logger.info("Computing mean, std")
    mean, std = dataset_mean_std(dataset)
    logger.info("Computing min, max")
    min, max = dataset_min_max(dataset)

    print("Mean:", mean)
    print("STD:", std)
    print("Min:", min)
    print("Max:", max)

    y = dataset.y
    print("First 10 targets:", y[:10])
    print("Targets distribution:", np.unique(y, return_counts=True))
    print("Resulting class-weight distribution:", compute_class_weight(dataset))


def stats_dataset(params: dict, split: str):
    data = params["data"]

    name = data["name"]
    path = Path(data["path"])

    regression = data['targets'] == "regression"
    reg_tt = log_min_max_scale
    target_transform = reg_tt if regression else flux_to_class_builder(data['targets']['classes'])

    dataset = None

    if name == "sdo-dataset":
        dataset = SDODataset(
            path / f"sdo-dataset-{split}.csv",
            transform=None,
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
            transform=transforms.ToTensor(),
            target_transform=target_transform,
            time_steps=data["time_steps"],
        )

    if dataset:
        _stats_dataset(dataset)
