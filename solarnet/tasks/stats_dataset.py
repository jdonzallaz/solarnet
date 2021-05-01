import logging

import numpy as np

from solarnet.data import dataset_from_config, dataset_mean_std, dataset_min_max
from solarnet.utils.target import compute_class_weight

logger = logging.getLogger(__name__)


def stats_dataset(params: dict, split: str):
    """
    Compute and print stats about datasets.
    """

    dataset = dataset_from_config(params, split)

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
