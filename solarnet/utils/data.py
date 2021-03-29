from typing import Optional, TypeVar

import torch
from torch import default_generator
from torch.utils.data import Dataset

T = TypeVar('T')


def train_test_split(dataset: Dataset[T], test_size: float = 0.1, seed: Optional[int] = None):
    if test_size < 0 or test_size > 1:
        raise ValueError("Test size must be in [0, 1].")

    test_size = int(test_size * len(dataset))
    train_size = len(dataset) - test_size

    generator = default_generator if seed is None else torch.Generator().manual_seed(seed)

    return torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
