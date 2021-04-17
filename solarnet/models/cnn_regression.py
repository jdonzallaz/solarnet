import pytorch_lightning as pl
import torch
import torch_optimizer
from pytorch_lightning.core.decorators import auto_move_data
from torch import nn, optim
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from solarnet.models.cnn_module import CNNModule
from solarnet.utils.scaling import log_min_max_inverse_scale


class CNNRegression(pl.LightningModule):
    def __init__(self, channels: int, height: int, width: int, learning_rate: float = 1e-4,
                 class_weight: torch.FloatTensor = None, total_steps: int = 0, activation: str = 'relu'):
        super().__init__()

        self.save_hyperparameters()

        self.cnn = CNNModule(channels, 1, activation)
        # self.sigmoid = nn.Sigmoid()

        # self.loss_fn = nn.L1Loss()  # MAE
        self.loss_fn = nn.MSELoss()

        self.test_metrics = MetricCollection([
            MeanAbsoluteError(),
            MeanSquaredError(),
        ])

    @auto_move_data
    def forward(self, image):
        # return self.sigmoid(self.cnn(image))
        return self.cnn(image)

    def training_step(self, batch, batch_id):
        return self.step(batch, step_type="train")

    def validation_step(self, batch, batch_id):
        return self.step(batch, step_type="val")

    def step(self, batch, step_type: str):
        image, y = batch
        y_pred = self(image)
        y_pred = torch.flatten(y_pred)
        loss = self.loss_fn(y_pred, y.float())

        self.log(f"{step_type}_loss", loss, prog_bar=True, sync_dist=False)

        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        logits = self(image)
        y_pred = torch.flatten(logits)
        y = log_min_max_inverse_scale(y)
        y_pred = log_min_max_inverse_scale(y_pred)

        self.test_metrics(y_pred, y)

    def test_epoch_end(self, outs):
        test_metrics = self.test_metrics.compute()
        self.log('test_mae', test_metrics["MeanAbsoluteError"])
        self.log('test_mse', test_metrics["MeanSquaredError"])

    def configure_optimizers(self):
        optimizer = torch_optimizer.RAdam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            # betas=(0.9, 0.999),
            # eps=1e-8,
            # weight_decay=0,
        )

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate,
                                                  total_steps=self.hparams.total_steps,
                                                  pct_start=0.1,
                                                  # three_phase=True,
                                                  three_phase=False,
                                                  div_factor=1000,
                                                  verbose=False)
        scheduler = {'scheduler': scheduler, 'interval': 'step'}

        # return optimizer
        return [optimizer], [scheduler]
