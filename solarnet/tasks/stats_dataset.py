import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torchvision.transforms import transforms
from tqdm import tqdm

from solarnet.data import datamodule_from_config, dataset_from_config, dataset_mean_std, dataset_min_max
from solarnet.utils.plots import plot_histogram
from solarnet.utils.target import compute_class_weight

logger = logging.getLogger(__name__)


def stats_dataset(
    params: dict,
    split: str,
    n_bins: int = 100,
    hist_path: Optional[Path] = None,
    transform: bool = False,
):
    """
    Compute and print stats about datasets.
    """

    if transform:
        dm = datamodule_from_config(params)
        transform_fn = dm.transform
    else:
        transform_fn = None

    dataset = dataset_from_config(params, split, transform=transform_fn)

    print("Size:", len(dataset))
    print("Shape:", dataset[0][0].shape)

    y = dataset.y()
    print("First 10 targets:", y[:10])
    print("Targets distribution:", np.unique(y, return_counts=True))
    print("Resulting class-weight distribution:", compute_class_weight(dataset))

    logger.info("Computing mean, std")
    mean, std = dataset_mean_std(dataset)
    logger.info("Computing min, max")
    min, max = dataset_min_max(dataset)

    print("Mean:", mean)
    print("STD:", std)
    print("Min:", min)
    print("Max:", max)

    # Histogram
    if hist_path is not None:
        logger.info("Computing histogram")
        global_hist = torch.zeros(n_bins)
        for sample in tqdm(dataset):
            hist = torch.histc(sample[0].flatten(), bins=n_bins, min=min, max=max)
            global_hist = global_hist + hist
        plot_histogram(global_hist, min=min, max=max, save_path=hist_path)
        logger.info(f"Histogram saved to {hist_path}")
