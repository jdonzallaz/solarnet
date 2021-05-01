import datetime as dt
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from solarnet.data.dataset_utils import BaseDataset
from solarnet.utils.data import train_test_split


class SDOBenchmarkDataset(BaseDataset):
    def __init__(
        self,
        csv_file: Path,
        root_folder: Path,
        channel="171",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        time_steps: Union[int, List[int]] = 0
    ):
        metadata = pd.read_csv(csv_file, parse_dates=["start", "end"])

        self.root_folder = root_folder
        self.channel = channel
        self.transform = transform
        self.target_transform = target_transform

        self.time_steps_values = [0, 7 * 60, 10 * 60 + 30, 11 * 60 + 50]
        self.time_steps = time_steps if isinstance(time_steps, list) else [time_steps]

        self.setup(metadata)

    def setup(self, metadata):
        ls = []
        for i in range(len(metadata)):
            sample_metadata = metadata.iloc[i]
            target = sample_metadata["peak_flux"]
            if self.target_transform is not None and \
                isinstance(self.target_transform(target), int) and \
                self.target_transform(target) < 0:
                # Ignore sample if it is not part of a class
                continue

            sample_active_region, sample_date = sample_metadata["id"].split("_", maxsplit=1)

            paths: List[Path] = []
            for time_step in self.time_steps:
                image_date = sample_metadata["start"] + dt.timedelta(minutes=self.time_steps_values[time_step])
                image_date_str = dt.datetime.strftime(image_date, "%Y-%m-%dT%H%M%S")
                image_name = f"{image_date_str}__{self.channel}.jpg"
                paths.append(Path(sample_active_region) / sample_date / image_name)

            if not all((self.root_folder / path).exists() for path in paths):
                continue

            ls.append((paths, target))

        self.ls = ls

    def __len__(self) -> int:
        return len(self.ls)

    def __getitem__(self, index):
        metadata = self.ls[index]
        target = metadata[1]
        images = [Image.open(self.root_folder / path) for path in metadata[0]]
        to_tensor = transforms.ToTensor()
        images = [to_tensor(image) for image in images]

        if self.transform:
            images = [self.transform(image) for image in images]
        if self.target_transform:
            target = self.target_transform(target)

        if not torch.is_tensor(images[0]):
            return images[0], target

        # Put images of different time steps as one image of multiple channels (time steps ~ rgb)
        image = torch.cat(images, 0)

        return image, target

    def y(self, indices: Optional[Sequence[int]] = None) -> list:
        ls = self.ls
        if indices is not None:
            ls = (self.ls[i] for i in indices)

        if self.target_transform is not None:
            return [self.target_transform(y[1]) for y in ls]

        return [y[1] for y in ls]


class SDOBenchmarkDataModule(pl.LightningDataModule):
    def __init__(
        self, dataset_dir: Path,
        channel: str = '171',
        batch_size: int = 32,
        num_workers: int = 0,
        validation_size: float = 0.1,
        resize: int = 64,
        seed: int = 42,
        target_transform: Callable[[float], any] = None,
        time_steps: Union[int, List[int]] = 0,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.channel = channel
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_size = validation_size
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.target_transform = target_transform
        self.time_steps = time_steps

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.dataset_train_val = SDOBenchmarkDataset(self.dataset_dir / 'training' / 'meta_data.csv',
                                                         self.dataset_dir / 'training', transform=self.transform,
                                                         target_transform=self.target_transform, channel=self.channel,
                                                         time_steps=self.time_steps)
            self.dataset_train, self.dataset_val = train_test_split(self.dataset_train_val, self.validation_size,
                                                                    self.seed)
            self.dims = tuple(self.dataset_val[0][0].shape)

        if stage == 'test' or stage is None:
            self.dataset_test = SDOBenchmarkDataset(self.dataset_dir / 'test' / 'meta_data.csv',
                                                    self.dataset_dir / 'test', transform=self.transform,
                                                    target_transform=self.target_transform, channel=self.channel,
                                                    time_steps=self.time_steps)
            self.dims = tuple(self.dataset_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
