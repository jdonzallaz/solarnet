import time
from pathlib import Path
from typing import Union

import pytorch_lightning as pl
import torch
from torch import nn

from solarnet.utils.hardware import machine_summary


def pytorch_model_summary(model: nn.Module, path: Path, filename: str = 'model_summary.txt'):
    model_repr = repr(model)
    nb_parameters = sum([param.nelement() for param in model.parameters()])

    with open(path / filename, "w") as text_file:
        print(model_repr, file=text_file)
        print("=" * 80, file=text_file)
        print(f"Total parameters: {nb_parameters}", file=text_file)


def get_training_summary(
    model_file: Path,
    tracking_id: Union[str, int],
    start_time: float,
    datamodule: pl.LightningDataModule,
    early_stop_callback: pl.callbacks.EarlyStopping,
    checkpoint_callback: pl.callbacks.ModelCheckpoint,
    steps_per_epoch: int,
):
    model_size_kB = int(model_file.stat().st_size / 1000)

    model_checkpoint_step = torch.load(checkpoint_callback.best_model_path)["global_step"]

    return {
        "machine": machine_summary(),
        "training_time": f"{time.perf_counter() - start_time:.2f}s",
        "model_size": f"{model_size_kB}kB",
        "early_stopping_epoch": early_stop_callback.stopped_epoch,
        "model_checkpoint_step": model_checkpoint_step,
        "model_checkpoint_epoch": model_checkpoint_step // steps_per_epoch,
        "tracking_id": tracking_id,
        "dataset": {
            "training_set_size": len(datamodule.train_dataloader().dataset),
            "validation_set_size": len(datamodule.val_dataloader().dataset),
            "test_set_size": len(datamodule.test_dataloader().dataset),
        },
    }
