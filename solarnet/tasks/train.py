import logging
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from solarnet.data.sdo_benchmark_datamodule import SDOBenchmarkDataModule
from solarnet.models.baseline import CNN
from solarnet.utils.target import flux_to_class_builder
from solarnet.utils.tracking import NeptuneTracking, Tracking

logger = logging.getLogger(__name__)


def train(parameters: dict):
    logger.info("Training...")

    seed_everything(parameters['seed'])

    ds_path = Path('data/sdo-benchmark')
    model_path = Path('models/baseline/')

    datamodule = SDOBenchmarkDataModule(
        ds_path,
        batch_size=parameters['trainer']['batch_size'],
        validation_size=parameters['data']['validation_size'],
        channel=parameters['data']['channel'],
        resize=parameters['data']['size'],
        seed=parameters['seed'],
        num_workers=0 if os.name == 'nt' else 4,  # Windows supports only 1, Linux supports more
        target_transform=flux_to_class_builder(parameters['data']['targets']['classes']),
        time_steps=parameters['data']['time_steps'],
    )
    datamodule.setup()
    logger.info(f"Data format: {datamodule.size()}")

    steps_per_epoch = len(datamodule.train_dataloader())
    total_steps = parameters['trainer']['epochs'] * steps_per_epoch

    model = CNN(*datamodule.size(),
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
        filename="model-{epoch:02d}",
        save_top_k=2,
    )
    callbacks = [early_stop_callback, checkpoint_callback]

    tracking: Tracking = NeptuneTracking(parameters=parameters, tags=[], disabled=not parameters['tracking'])
    pl_logger = tracking.get_pl_logger()

    if pl_logger is not None:
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

    # Copy and rename best checkpoint
    # TODO: move to utils
    path = Path(checkpoint_callback.best_model_path)
    if str(path) != ".":
        shutil.copy2(path, Path(path.parent, "model.ckpt"))

    tracking.end()
