import logging
from typing import Union

import torch
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning.core.decorators import auto_move_data
from solarnet.models.backbone import get_backbone
from solarnet.models.classifier import Classifier
from torch import nn, optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from solarnet.models.model_utils import BaseModel
from solarnet.utils.scaling import log_min_max_inverse_scale

logger = logging.getLogger(__name__)


class ImageRegression(BaseModel):
    """
    Model for image regression.
    This is a configurable class composed by a backbone (see solarnet.models.backbone.py) and
    a regressor head (actually a classifier with 1 output).
    It is also a LightningModule and nn.Module.
    """

    def __init__(
        self,
        n_channel: int = 1,
        learning_rate: float = 1e-4,
        backbone: Union[str, nn.Module] = "simple-cnn",
        backbone_output_size: int = 0,
        n_hidden: int = 512,
        dropout: float = 0.2,
        loss_fn: str = "mse",
        lr_scheduler: bool = False,
        lr_scheduler_warmup_steps: int = 100,
        lr_scheduler_total_steps: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        if isinstance(backbone, str):
            self.backbone, backbone_output_size = get_backbone(backbone, channels=n_channel, dropout=dropout, **kwargs)

        self.regressor = Classifier(backbone_output_size, 1, n_hidden, dropout)

        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "mae":
            self.loss_fn = nn.L1Loss()  # MAE
        else:
            raise RuntimeError("Undefined loss function")

        self.test_metrics = MetricCollection(
            [
                MeanAbsoluteError(),
                MeanSquaredError(),
            ]
        )

    @property
    def backbone_name(self) -> str:
        if isinstance(self.hparams.backbone, str):
            return self.hparams.backbone
        else:
            return type(self.hparams.backbone).__name__

    @property
    def output_size(self) -> int:
        return 1

    @auto_move_data
    def forward(self, image):
        return self.regressor(self.backbone(image))

    def training_step(self, batch, batch_id):
        return self.step(batch, step_type="train")

    def validation_step(self, batch, batch_id):
        return self.step(batch, step_type="val")

    def step(self, batch, step_type: str):
        image, y = batch
        y_pred = self(image)
        y_pred = torch.flatten(y_pred)
        y = y.float()
        loss = self.loss_fn(y_pred, y)

        self.log(f"{step_type}_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        y_pred = self(image)
        y_pred = torch.flatten(y_pred)
        y = log_min_max_inverse_scale(y)
        y_pred = log_min_max_inverse_scale(y_pred)

        self.test_metrics(y_pred, y)

    def test_epoch_end(self, outs):
        test_metrics = self.test_metrics.compute()
        self.log("test_mae", test_metrics["MeanAbsoluteError"])
        self.log("test_mse", test_metrics["MeanSquaredError"])

    def configure_optimizers(self):
        logger.info(f"configure_optimizers lr={self.hparams.learning_rate}")

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.learning_rate,
        )

        if not self.hparams.lr_scheduler:
            return optimizer

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            linear_warmup_decay(
                self.hparams.lr_scheduler_warmup_steps, self.hparams.lr_scheduler_total_steps, cosine=True
            ),
        )

        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            },
        )
