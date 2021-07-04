import logging
from typing import List, Union

import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import linear_warmup_decay
from pytorch_lightning.core.decorators import auto_move_data
from solarnet.models.backbone import get_backbone
from solarnet.models.classifier import Classifier
from torch import nn, optim
from torchmetrics import Accuracy, F1, MetricCollection, Recall, StatScores

from solarnet.models.model_utils import BaseModel

logger = logging.getLogger(__name__)


class ImageClassification(BaseModel):
    """
    Model for image classification.
    This is a configurable class composed by a backbone (see solarnet.models.backbone.py) and
    a classifier.
    It is also a LightningModule and nn.Module.
    """

    def __init__(
        self,
        n_channel: int = 1,
        n_class: int = 2,
        learning_rate: float = 1e-4,
        class_weight: List[float] = None,
        backbone: Union[str, nn.Module] = "simple-cnn",
        backbone_output_size: int = 0,
        n_hidden: int = 512,
        dropout: float = 0.2,
        lr_scheduler: bool = False,
        lr_scheduler_warmup_steps: int = 100,
        lr_scheduler_total_steps: int = 0,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        if isinstance(backbone, str):
            self.backbone, backbone_output_size = get_backbone(
                backbone,
                channels=n_channel,
                dropout=dropout,
                **kwargs,
            )

        self.classifier = Classifier(backbone_output_size, n_class, n_hidden, dropout)

        if class_weight is not None:
            class_weight = torch.tensor(class_weight, dtype=torch.float)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_metrics = MetricCollection(
            [
                Accuracy(),
                F1(num_classes=self.hparams.n_class, average="macro"),
                Recall(num_classes=self.hparams.n_class, average="macro"),  # balanced acc.
                StatScores(
                    num_classes=self.hparams.n_class if self.hparams.n_class > 2 else 1,
                    reduce="micro",
                    multiclass=self.hparams.n_class > 2,
                ),
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
        return self.hparams.n_class

    @auto_move_data
    def forward(self, image):
        return self.classifier(self.backbone(image))

    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        image, _ = batch
        return self(image)

    def training_step(self, batch, batch_id):
        return self.step(batch, step_type="train")

    def validation_step(self, batch, batch_id):
        return self.step(batch, step_type="val")

    def step(self, batch, step_type: str):
        image, y = batch
        y_pred = self(image)
        loss = self.loss_fn(y_pred, y)

        self.log(f"{step_type}_loss", loss, prog_bar=True, sync_dist=True)

        # Compute accuracy
        y_pred = F.softmax(y_pred, dim=1)
        self.__getattr__(f"{step_type}_accuracy")(y_pred, y)
        self.log(f"{step_type}_accuracy", self.__getattr__(f"{step_type}_accuracy"), on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        y_pred = self(image)
        y_pred = F.softmax(y_pred, dim=1)

        self.test_metrics(y_pred, y)

    def test_epoch_end(self, outs):
        test_metrics = self.test_metrics.compute()

        tp, fp, tn, fn, _ = test_metrics.pop("StatScores")
        self.log("test_tp", tp)
        self.log("test_fp", fp)
        self.log("test_tn", tn)
        self.log("test_fn", fn)

        for key, value in test_metrics.items():
            self.log(f"test_{key.lower()}", value)

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
