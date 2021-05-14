import logging
import time
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning.loggers import LoggerCollection

from solarnet.logging import InMemoryLogger
from solarnet.utils.plots import plot_train_val_curve

logger = logging.getLogger(__name__)


class InfoLogCallback(pl_callbacks.Callback):
    """
    Log info about the stage. It uses the base python logger.info to log when the stage start/end.
    It also logs datamodule format and model architecture.
    """

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        self._log_stage(stage, "start...")

        if hasattr(trainer, "datamodule") and isinstance(trainer.datamodule, pl.LightningDataModule):
            logger.info(f"Data format: {trainer.datamodule.size()}")

        logger.info(f"Model: {pl_module}")

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        self._log_stage(stage, "end.")

    def _log_stage(self, stage: str, status: str):
        if stage == "fit":
            logger.info(f"Training {status}")
        elif stage == "validate":
            logger.info(f"Validation {status}")
        elif stage == "test":
            logger.info(f"Test {status}")
        elif stage == "predict":
            logger.info(f"Prediction {status}")
        elif stage == "tune":
            logger.info(f"Tuning {status}")


class TimerCallback(pl_callbacks.Callback):
    """
    Compute the time used for the stage. Time is computed on setup and teardown.
    """

    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        self.start_time = time.perf_counter()

    def teardown(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        self.end_time = time.perf_counter()
        logger.info(f"{stage} stage took {self.get_time_formatted()} to run.")

    def get_time(self) -> float:
        return self.end_time - self.start_time

    def get_time_formatted(self) -> str:
        return f"{self.get_time():.2f}s"


class PlotTrainValCurveCallback(pl_callbacks.Callback):
    """
    Plot the curve of a metric for training and validation.
    Depends on InMemoryLogger.
    """

    def __init__(self, path: Path = None, metric: str = "loss", filename_suffix: str = "_curve"):
        self.path = path
        self.metric = metric
        self.filename_suffix = filename_suffix

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_logger = trainer.logger
        if isinstance(pl_logger, LoggerCollection):
            pl_logger = next((l for l in pl_logger if isinstance(l, InMemoryLogger)), None)

        if not isinstance(pl_logger, InMemoryLogger):
            return

        if f"train_{self.metric}" not in pl_logger.metrics:
            return

        plot_train_val_curve(
            pl_logger.metrics,
            metric=self.metric,
            save_path=self.path / f"{self.metric}{self.filename_suffix}.png",
        )
