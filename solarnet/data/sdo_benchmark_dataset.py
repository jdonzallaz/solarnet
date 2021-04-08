import datetime as dt
import os
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
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
    ):
        metadata = pd.read_csv(csv_file, parse_dates=["start", "end"])

        self.root_folder = root_folder
        self.channel = channel
        self.transform = transform
        self.target_transform = target_transform

        self.time_steps = [0, 7 * 60, 10 * 60 + 30, 11 * 60 + 50]

        self.setup(metadata)

    def setup(self, metadata):
        ls = []
        for i in range(len(metadata)):
            sample_metadata = metadata.iloc[i]
            target = sample_metadata["peak_flux"]

            sample_active_region, sample_date = sample_metadata["id"].split("_", maxsplit=1)

            image_date = sample_metadata["start"] + dt.timedelta(minutes=self.time_steps[3])
            image_date_str = dt.datetime.strftime(
                image_date,
                "%Y-%m-%dT%H%M%S",
            )
            image_name = f"{image_date_str}__{self.channel}.jpg"
            image_path = self.root_folder / sample_active_region / sample_date / image_name
            if image_path.exists():
                ls.append([os.path.join(sample_active_region, sample_date, image_name), target])

        self.ls = ls

    def get_y(self):
        return [self.target_transform(y[1]) for y in self.ls]

    def __len__(self) -> int:
        return len(self.ls)

    def __getitem__(self, index):
        metadata = self.ls[index]
        target = metadata[1]
        image = Image.open(self.root_folder / metadata[0])

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

        return image, target

    def find_image(self, path: Path, expected_date: dt.datetime) -> Path:
        time_space = -1
        image_name = None

        for img in [name for name in os.listdir(path) if name.endswith('.jpg')]:
            img_datetime_raw, img_wavelength = os.path.splitext(img)[0].split("__")
            if img_wavelength != self.channel:
                continue
            img_datetime = dt.datetime.strptime(img_datetime_raw, "%Y-%m-%dT%H%M%S")

            current_time_space = abs((expected_date - img_datetime).total_seconds())
            if time_space == -1 or current_time_space < time_space:
                time_space = current_time_space
                image_name = img

        if image_name is None:
            raise ValueError("Error in dataset, cannot find image.")

        return path / image_name
