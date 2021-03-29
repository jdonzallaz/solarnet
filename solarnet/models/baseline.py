import pytorch_lightning as pl
import torch
import torch_optimizer
from pytorch_lightning.core.decorators import auto_move_data
from torch import nn, optim
from torchmetrics.functional import accuracy, f1


def conv_block(input_size, output_size, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(input_size, output_size, *args, **kwargs),
        nn.ReLU(),
        # nn.Dropout2d(0.1),
        # nn.Conv2d(input_size, output_size, (3, 3)),
        # nn.ReLU(),
        nn.MaxPool2d((2, 2)),
    )


class CNN(pl.LightningModule):
    def __init__(self, channels: int, height: int, width: int, n_class: int, learning_rate: float = 1e-4,
                 class_weight: torch.FloatTensor = None, total_steps: int = 0):
        super().__init__()

        self.save_hyperparameters()

        # Architecture
        self.conv_blocks = nn.Sequential(
            conv_block(channels, 16, kernel_size=3, padding=1),
            # conv_block(8, 16, kernel_size=3, padding=1),
            conv_block(16, 64, kernel_size=3, padding=1),
            nn.Flatten(),
        )
        self.linear_block = nn.Sequential(
            nn.Linear(int(64 * height * width / (4 ** 2)), 16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )
        self.out = nn.Sequential(
            # nn.Linear(int(32 * height * width / 4 / 4), 5),
            nn.Linear(16, n_class),
            # nn.Sigmoid(),
        )

        # Metrics
        # self.loss_fn = nn.L1Loss()
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        # self.loss_fn = nn.BCEWithLogitsLoss()
        # self.metrics = {
        #     "loss": [],
        #     "val_loss": [],
        # }
        # self.test_metrics = {}
        # self.mae = pl.metrics.MeanAbsoluteError()

    @auto_move_data
    def forward(self, image):
        image = self.conv_blocks(image)
        image = self.linear_block(image)

        return self.out(image)

    def training_step(self, batch, batch_id):
        loss = self.step(batch, step_type="train")
        return loss

        # tensorboard_logs = {"train_loss": loss}
        # return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_id):
        loss = self.step(batch, step_type="val")

        return loss

    # def test_step(self, batch, batch_id):
    #     loss, output, labels = self.step(batch, step_type="test")
    #
    #     return {"loss": loss, "preds": output, "target": labels}
    #
    # def test_step_end(self, outputs):
    #     self.mae(outputs["preds"], outputs["target"])

    def test_step(self, batch, batch_idx):
        image, y = batch
        logits = self(image)
        y_pred = torch.argmax(logits, dim=1)
        acc = accuracy(y_pred, y)
        self.log('test_accuracy', acc)
        f1_score = f1(y_pred, y, self.hparams.n_class, average='macro')
        self.log('test_f1', f1_score)

    def step(self, batch, step_type: str):
        image, y = batch

        y_pred = self(image)
        # print(y_pred.shape)
        # y_pred = torch.flatten(y_pred)
        # y_pred = y_pred.double()
        # y = y.double()
        # y_pred = y_pred.type_as(y)

        loss = self.loss_fn(y_pred, y)

        self.log(f"{step_type}_loss", loss, prog_bar=True, sync_dist=False)

        return loss

    # def validation_epoch_end(self, outputs):
    #     avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
    #
    #     self.metrics["val_loss"].append(avg_val_loss.item())

    # def training_epoch_end(self, outputs):
    #     avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
    #
    #     self.metrics["loss"].append(avg_train_loss.item())

    # def test_epoch_end(self, outputs):
    #     self.test_metrics["accuracy"] = round(self.mae.compute().item(), 4)

    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        optimizer = torch_optimizer.RAdam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            # betas=(0.9, 0.999),
            # eps=1e-8,
            # weight_decay=0,
        )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
        # scheduler = {'scheduler': scheduler, 'interval': 'epoch', 'reduce_on_plateau': True, 'monitor': 'val_loss'}
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