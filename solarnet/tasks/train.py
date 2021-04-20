import logging
import os
import time
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from solarnet.data.sdo_benchmark_datamodule import SDOBenchmarkDataModule
from solarnet.logging import InMemoryLogger
from solarnet.logging.tracking import NeptuneNewTracking, Tracking
from solarnet.models import CNNClassification, CNNRegression
from solarnet.utils.plots import plot_loss_curve
from solarnet.utils.pytorch import get_training_summary, pytorch_model_summary
from solarnet.utils.scaling import log_min_max_scale
from solarnet.utils.target import flux_to_class_builder
from solarnet.utils.yaml import write_yaml

logger = logging.getLogger(__name__)


def train(parameters: dict):
    logger.info("Training...")
    start_time = time.perf_counter()

    seed_everything(parameters['seed'])

    ds_path = Path('data/sdo-benchmark')
    model_path = Path(parameters["path"])

    regression = parameters['data']['targets'] == "regression"
    reg_tt = log_min_max_scale
    target_transform = reg_tt if regression else flux_to_class_builder(parameters['data']['targets']['classes'])

    datamodule = SDOBenchmarkDataModule(
        ds_path,
        batch_size=parameters['trainer']['batch_size'],
        validation_size=parameters['data']['validation_size'],
        channel=parameters['data']['channel'],
        resize=parameters['data']['size'],
        seed=parameters['seed'],
        num_workers=0 if os.name == 'nt' else 4,  # Windows supports only 1, Linux supports more
        target_transform=target_transform,
        time_steps=parameters['data']['time_steps'],
    )
    datamodule.setup()
    logger.info(f"Data format: {datamodule.size()}")

    steps_per_epoch = len(datamodule.train_dataloader())
    total_steps = parameters['trainer']['epochs'] * steps_per_epoch

    if regression:
        model = CNNRegression(*datamodule.size(),
                              learning_rate=parameters['trainer']['learning_rate'],
                              class_weight=datamodule.class_weight,
                              total_steps=total_steps,
                              activation=parameters['model']['activation'])
    else:
        model = CNNClassification(*datamodule.size(),
                                  n_class=len(parameters['data']['targets']['classes']),
                                  learning_rate=parameters['trainer']['learning_rate'],
                                  class_weight=datamodule.class_weight,
                                  total_steps=total_steps,
                                  activation=parameters['model']['activation'])
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
                         # precision=16,
                         fast_dev_run=False,
                         # limit_train_batches=1,
                         # limit_val_batches=1,
                         # limit_test_batches=10,
                         log_every_n_steps=10, flush_logs_every_n_steps=10,
                         accelerator=None if parameters['gpus'] in [None, 0, 1] else 'ddp',
                         )

    trainer.fit(model, datamodule=datamodule)
    # trainer.tuner.scale_batch_size(model, init_val=32, mode='binsearch', datamodule=datamodule)
    # return

    # Log actual config used for training
    write_yaml(model_path / "config.yaml", parameters)

    # Log model summary
    pytorch_model_summary(model, model_path)

    # Save plot of history
    plot_loss_curve(im_logger.metrics, save_path=model_path / 'history.png')

    # Output tracking run id to continue logging in test step
    run_id = tracking.get_id()

    # Metadata
    metadata = get_training_summary(model_path / "model.ckpt", run_id, start_time, datamodule, early_stop_callback,
                                    checkpoint_callback)
    write_yaml(model_path / "metadata.yaml", metadata)

    tracking.end()
