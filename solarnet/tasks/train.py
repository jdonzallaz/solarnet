import logging
import time
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from solarnet.data.dataset_config import datamodule_from_config
from solarnet.logging import InMemoryLogger
from solarnet.logging.tracking import NeptuneNewTracking, Tracking
from solarnet.models import model_from_config
from solarnet.utils.plots import plot_train_val_curve
from solarnet.utils.pytorch import get_training_summary, pytorch_model_summary
from solarnet.utils.yaml import write_yaml

logger = logging.getLogger(__name__)


def train(parameters: dict):
    datamodule = datamodule_from_config(parameters)
    datamodule.setup()

    model = model_from_config(parameters, datamodule)

    _train(parameters, datamodule, model)


def _train(parameters: dict, datamodule: LightningDataModule, model: LightningModule):
    logger.info("Training...")
    start_time = time.perf_counter()

    seed_everything(parameters['seed'])
    model_path = Path(parameters["path"])
    plot_path = Path(parameters["path"]) / "train_plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    regression = parameters["data"]['targets'] == "regression"
    n_class = 1 if regression else len(parameters['data']['targets']['classes'])

    if not datamodule.has_setup_fit:
        datamodule.setup("fit")
    logger.info(f"Data format: {datamodule.size()}")

    logger.info(f"Model: {model}")

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=parameters['trainer']['patience'],
                                        verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        dirpath=str(model_path),
        filename="model",
    )
    callbacks = [early_stop_callback, checkpoint_callback]

    tracking: Tracking = NeptuneNewTracking(parameters=parameters, tags=[], disabled=not parameters['tracking'])
    tracking_logger = tracking.get_callback('pytorch-lightning')
    im_logger = InMemoryLogger()
    if tracking_logger is None:
        pl_logger = im_logger
    else:
        pl_logger = [im_logger, tracking_logger]

    if tracking_logger is not None:
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    trainer = pl.Trainer(gpus=parameters['gpus'],
                         logger=pl_logger,
                         callbacks=callbacks,
                         max_epochs=parameters['trainer']['epochs'],
                         val_check_interval=0.5,
                         num_sanity_val_steps=0,
                         fast_dev_run=False,
                         log_every_n_steps=10, flush_logs_every_n_steps=10,
                         accelerator=None if parameters['gpus'] in [None, 0, 1] else 'ddp',
                         )

    trainer.fit(model, datamodule=datamodule)

    # Log actual config used for training
    write_yaml(model_path / "config.yaml", parameters)

    # Log model summary
    pytorch_model_summary(model, model_path)

    # Save plot of history
    y_lim_loss = [0, 0.25] if regression else [0, 2] if n_class > 2 else [0, 1]
    plot_train_val_curve(im_logger.metrics, metric="loss", save_path=plot_path / "loss_curve.png", y_lim=y_lim_loss)
    if not regression:
        plot_train_val_curve(im_logger.metrics, metric="accuracy", key_suffix="_epoch",
                             save_path=plot_path / "accuracy_curve.png", y_lim=[0.25, 1.0])

    # Output tracking run id to continue logging in test step
    run_id = tracking.get_id()

    # Metadata
    steps_per_epoch = len(datamodule.train_dataloader())
    metadata = get_training_summary(model_path / "model.ckpt", run_id, start_time, datamodule, early_stop_callback,
                                    checkpoint_callback, steps_per_epoch)
    write_yaml(model_path / "metadata.yaml", metadata)

    tracking.end()
