import logging
from pathlib import Path
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from solarnet.callbacks import InfoLogCallback, PlotTrainValCurveCallback, TimerCallback
from solarnet.tracking import InMemoryLogger, NeptuneNewTracking, Tracking
from solarnet.utils.pytorch import get_training_summary, pytorch_model_summary
from solarnet.utils.yaml import write_yaml

logger = logging.getLogger(__name__)


def validate_train_parameters(parameters: dict):
    """
    Validate the parameters needed for training. Only verify presence of parameters, not the type.
    """
    keys = {"path": None, "data": ["targets"], "trainer": ["patience", "epochs"], "system": ["gpus"], "tracking": None}

    _validate_train_parameters(parameters, keys)


def _validate_train_parameters(parameters: dict, keys: dict):
    """
    Sub-function to validate the parameters recursively.
    """

    for k, v in keys.items():
        if k not in parameters:
            raise RuntimeError(f'Missing key "{k}" in parameters for training')
        if isinstance(v, list):
            for i in v:
                if i not in parameters[k]:
                    raise RuntimeError(f'Missing key "{k}.{i}" in parameters for training')
        if isinstance(v, dict):
            _validate_train_parameters(parameters[k], v)


def train(
    parameters: dict,
    datamodule: LightningDataModule,
    model: LightningModule,
    callbacks: Optional[Union[List[Callback], Callback]] = None,
):
    validate_train_parameters(parameters)

    model_path = Path(parameters["path"])
    plot_path = Path(parameters["path"]) / "train_plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    regression = parameters["data"]["targets"] == "regression"
    n_class = 1 if regression else len(parameters["data"]["targets"]["classes"])

    # Callbacks
    if callbacks is None:
        callbacks = []
    elif not isinstance(callbacks, list):
        callbacks = [callbacks]
    timer_callback = TimerCallback()

    early_stop_callback = (
        EarlyStopping(
            monitor="val_loss", min_delta=0.00, patience=parameters["trainer"]["patience"], verbose=True, mode="min"
        )
        if parameters["trainer"]["patience"] > 0
        else None
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=str(model_path),
        filename="model",
    )
    callbacks += [
        early_stop_callback,
        checkpoint_callback,
        InfoLogCallback(),
        timer_callback,
        PlotTrainValCurveCallback(plot_path, "loss"),
        PlotTrainValCurveCallback(plot_path, "accuracy"),
    ]
    callbacks = [c for c in callbacks if c is not None]

    # Logging
    im_logger = InMemoryLogger()
    pl_logger = [im_logger]

    # Tracking
    tracking: Tracking = NeptuneNewTracking(
        parameters=parameters, tags=parameters.get("tags", []), disabled=not parameters["tracking"] or parameters["tune_lr"]
    )
    tracking_logger = tracking.get_callback("pytorch-lightning")
    if tracking_logger is not None:
        pl_logger.append(tracking_logger)
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    trainer = pl.Trainer(
        gpus=parameters["system"]["gpus"],
        logger=pl_logger,
        callbacks=callbacks,
        max_epochs=parameters["trainer"]["epochs"],
        num_sanity_val_steps=0,
        fast_dev_run=False,
        accelerator=None if parameters["system"]["gpus"] in [None, 0, 1] else "dp",
        sync_batchnorm=True,
        deterministic=True,
    )

    if parameters["tune_lr"]:
        lr_find(
            trainer,
            model,
            datamodule,
            model_path,
            min_lr=parameters.get("min_lr", 1e-6),
            max_lr=parameters.get("max_lr", 1e-2),
            num_training=parameters.get("num_training", 100),
        )
        return

    trainer.fit(model, datamodule=datamodule)

    # Save pytorch model / nn.Module / weights
    torch.save(model.state_dict(), Path(model_path / "model.pt"))

    # Save meaningful parameters for loading
    model_config = {
        "model": type(model).__name__,
        "backbone": model.backbone_name,
        "n_channel": datamodule.size(0),
        "output_size": model.output_size,
        "hparams": dict(model.hparams),
    }
    write_yaml(model_path / "model_config.yaml", model_config)

    # Log actual config used for training
    write_yaml(model_path / "config.yaml", parameters)

    # Log model summary
    pytorch_model_summary(model, model_path)

    # Output tracking run id to continue logging in test step
    run_id = tracking.get_id()

    # Metadata
    steps_per_epoch = len(datamodule.train_dataloader())
    metadata = get_training_summary(
        model_path / "model.ckpt",
        parameters,
        run_id,
        timer_callback,
        datamodule,
        early_stop_callback,
        checkpoint_callback,
        steps_per_epoch,
        save_path=plot_path,
    )
    write_yaml(model_path / "metadata.yaml", metadata)
    tracking.log_property("metadata", metadata)
    tracking.log_artifact(plot_path / "data-examples-train.png")
    tracking.log_artifact(plot_path / "data-examples-val.png")
    if (plot_path / "data-examples-test.png").exists():
        tracking.log_artifact(plot_path / "data-examples-test.png")

    tracking.end()


def lr_find(trainer, model, datamodule, path, min_lr=1e-6, max_lr=1e-2, num_training=100):
    lr_finder = trainer.tuner.lr_find(
        model, datamodule=datamodule, min_lr=min_lr, max_lr=max_lr, num_training=num_training
    )

    import matplotlib

    matplotlib.rcParams.update({"axes.grid": True, "grid.color": "k"})
    fig = lr_finder.plot(suggest=True)
    fig.savefig(str(path / "lr_finder.png"))

    print("Learning rate suggestion:", lr_finder.suggestion())
