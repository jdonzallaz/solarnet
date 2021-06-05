from collections import Counter
import logging
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys

from solarnet.callbacks import TimerCallback
from solarnet.utils.hardware import machine_summary

logger = logging.getLogger(__name__)


def pytorch_model_summary(model: nn.Module, path: Path, filename: str = "model_summary.txt"):
    model_repr = repr(model)
    nb_parameters = sum([param.nelement() for param in model.parameters()])

    with open(path / filename, "w") as text_file:
        print(model_repr, file=text_file)
        print("=" * 80, file=text_file)
        print(f"Total parameters: {nb_parameters}", file=text_file)


def get_training_summary(
    model_file: Path,
    tracking_id: Union[str, int],
    timer_callback: TimerCallback,
    datamodule: pl.LightningDataModule,
    early_stop_callback: Optional[pl.callbacks.EarlyStopping],
    checkpoint_callback: pl.callbacks.ModelCheckpoint,
    steps_per_epoch: int,
):
    model_size_kB = int(model_file.stat().st_size / 1000)

    model_checkpoint_step = torch.load(checkpoint_callback.best_model_path)["global_step"]

    early_stopping_epoch = 0 if early_stop_callback is None else early_stop_callback.stopped_epoch

    return {
        "machine": machine_summary(),
        "training_time": timer_callback.get_time_formatted(),
        "model_size": f"{model_size_kB}kB",
        "early_stopping_epoch": early_stopping_epoch,
        "model_checkpoint_step": model_checkpoint_step,
        "model_checkpoint_epoch": model_checkpoint_step // steps_per_epoch,
        "tracking_id": tracking_id,
        "dataset": {
            "training_set_size": len(datamodule.train_dataloader().dataset),
            "validation_set_size": len(datamodule.val_dataloader().dataset),
            "test_set_size": len(datamodule.test_dataloader().dataset),
        },
    }


def print_incompatible_keys(incompatible_keys: _IncompatibleKeys):
    """
    Pretty print a summary of the incompatible keys returned by pytorch's load_state_dict.

    :param incompatible_keys: the _IncompatibleKeys returned by load_state_dict call.
    """

    missing_keys = [".".join(i.split(".")[:2]) for i in incompatible_keys.missing_keys]
    missing_keys_count = dict(Counter(missing_keys))
    unexpected_keys = [".".join(i.split(".")[:2]) for i in incompatible_keys.unexpected_keys]
    unexpected_keys_count = dict(Counter(unexpected_keys))

    logger.info("Missing keys:")
    for k, v in missing_keys_count.items():
        logger.info(f"    {k}: {v}")

    logger.info("Unexpected keys:")
    for k, v in unexpected_keys_count.items():
        logger.info(f"    {k}: {v}")
