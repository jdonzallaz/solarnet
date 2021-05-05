import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_optimizer
from pytorch_lightning.core.decorators import auto_move_data
from torch import nn, optim
from torchmetrics import Accuracy, F1, MetricCollection, Recall, StatScores

from solarnet.models.cnn_module import CNNModule


class CNNClassification(pl.LightningModule):
    def __init__(self, channels: int, height: int, width: int, n_class: int, learning_rate: float = 1e-4,
                 class_weight: torch.FloatTensor = None, total_steps: int = 0, activation: str = 'relu'):
        super().__init__()

        self.save_hyperparameters()

        self.cnn = CNNModule(channels, n_class, activation)

        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_metrics = MetricCollection([
            Accuracy(),
            F1(num_classes=self.hparams.n_class, average="macro"),
            Recall(num_classes=self.hparams.n_class, average='macro'),
            StatScores(
                num_classes=self.hparams.n_class if self.hparams.n_class > 2 else 1,
                reduce="micro",
                is_multiclass=self.hparams.n_class > 2
            ),
        ])

    @auto_move_data
    def forward(self, image):
        return self.cnn(image)

    def training_step(self, batch, batch_id):
        return self.step(batch, step_type="train")

    def validation_step(self, batch, batch_id):
        return self.step(batch, step_type="val")

    def step(self, batch, step_type: str):
        image, y = batch
        y_pred = self(image)
        loss = self.loss_fn(y_pred, y)

        self.log(f"{step_type}_loss", loss, prog_bar=True, sync_dist=False)

        # Compute accuracy
        y_pred = F.softmax(y_pred, dim=1)
        self.__getattr__(f"{step_type}_accuracy")(y_pred, y)
        self.log(f"{step_type}_accuracy", self.__getattr__(f"{step_type}_accuracy"), on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        logits = self(image)
        y_pred = torch.argmax(logits, dim=1)

        self.test_metrics(y_pred, y)

    def test_epoch_end(self, outs):
        test_metrics = self.test_metrics.compute()
        self.log('test_accuracy', test_metrics["Accuracy"])
        self.log('test_recall', test_metrics["Recall"])
        self.log('test_f1', test_metrics["F1"])

        tp, fp, tn, fn, _ = test_metrics["StatScores"]
        self.log('test_tp', tp)
        self.log('test_fp', fp)
        self.log('test_tn', tn)
        self.log('test_fn', fn)

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
