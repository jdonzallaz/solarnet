from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.utils import class_weight
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from solarnet.data.sdo_benchmark_dataset import SDOBenchmarkDataset
from solarnet.utils.data import train_test_split
from solarnet.utils.physics import flux_to_class, flux_to_id, flux_to_binary_class


class SDOBenchmarkDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir: Path, channel: str = '171', batch_size: int = 32, num_workers: int = 0,
                 validation_size: float = 0.1, resize: int = 64, seed: int = 42):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.channel = channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_size = validation_size
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.class_weight = None

        # self.target_transform = flux_to_id
        self.target_transform = flux_to_binary_class

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            dataset_train_val = SDOBenchmarkDataset(self.dataset_dir / 'training' / 'meta_data.csv',
                                                    self.dataset_dir / 'training', transform=self.transform,
                                                    target_transform=self.target_transform, channel=self.channel)
            self.dataset_train, self.dataset_val = train_test_split(dataset_train_val, self.validation_size, self.seed)
            self.dims = tuple(self.dataset_val[0][0].shape)

            y = dataset_train_val.get_y()
            self.compute_class_weight(y)

        if stage == 'test' or stage is None:
            self.dataset_test = SDOBenchmarkDataset(self.dataset_dir / 'test' / 'meta_data.csv',
                                                    self.dataset_dir / 'test', transform=self.transform,
                                                    target_transform=self.target_transform, channel=self.channel)
            self.dims = tuple(self.dataset_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)

    def compute_class_weight(self, y: list):
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        self.class_weight = torch.tensor(cw, dtype=torch.float)