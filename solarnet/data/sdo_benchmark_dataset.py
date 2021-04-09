import datetime as dt
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SDOBenchmarkDataset(Dataset):
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
            if self.target_transform is not None and self.target_transform(target) < 0:
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

    def get_y(self):
        return [self.target_transform(y[1]) for y in self.ls]

    def __len__(self) -> int:
        return len(self.ls)

    def __getitem__(self, index):
        metadata = self.ls[index]
        target = metadata[1]
        images = [Image.open(self.root_folder / path) for path in metadata[0]]

        if self.transform:
            images = [self.transform(image) for image in images]
        if self.target_transform:
            target = self.target_transform(target)

        if not torch.is_tensor(images[0]):
            return images[0], target

        # Put images of different time steps as one image of multiple channels (time steps ~ rgb)
        image = torch.cat(images, 0)

        return image, target
