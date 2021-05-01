from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    @abstractmethod
    def y(self, indices: Optional[Sequence[int]] = None) -> list:
        pass


def dataset_mean_std(dataset: Dataset) -> Tuple[float, float]:
    """
    Compute the mean and standard deviation of a dataset. The dataset must return a tensor as first element in each
    sample tuple.
    This can be quite long because each sample is processed separately, to avoid saturating the memory with a huge tensor.

    :param dataset: The dataset for which to compute mean and std
    :return: The mean and std
    """

    len_dataset = len(dataset)

    # Mean
    sum_mean = torch.tensor(0, dtype=torch.float)
    for sample in dataset:
        sum_mean += sample[0].mean()

    global_mean = sum_mean / len_dataset

    # STD
    sum_variance = torch.tensor(0, dtype=torch.float)
    for sample in dataset:
        var, mean = torch.var_mean(sample[0])
        sum_variance += (mean - global_mean) ** 2 + var

    global_variance = sum_variance / len_dataset
    std = global_variance.sqrt()

    return global_mean.item(), std.item()


def dataset_min_max(dataset: Dataset) -> Tuple[float, float]:
    """
    Compute the min and max value of a dataset. The dataset must return a tensor as first element in each
    sample tuple.
    This can be quite long because each sample is processed separately, to avoid saturating the memory with a huge tensor.

    :param dataset: The dataset for which to compute min and max
    :return: The min and max values
    """

    min_value = torch.tensor(torch.finfo().max, dtype=torch.float)
    max_value = torch.tensor(torch.finfo().min, dtype=torch.float)

    for sample in dataset:
        # Min
        min_sample = sample[0].min()
        if min_sample < min_value:
            min_value = min_sample

        # Max
        max_sample = sample[0].max()
        if max_sample > max_value:
            max_value = max_sample

    return min_value.item(), max_value.item()
