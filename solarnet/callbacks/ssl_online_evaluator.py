from typing import Optional, Sequence

import torch
from pl_bolts.callbacks import SSLOnlineEvaluator as SSLOnlineEvaluator_bolts
from pytorch_lightning import LightningModule, Trainer
from torch.nn import functional as F
from torchmetrics.functional import accuracy


class SSLOnlineEvaluator(SSLOnlineEvaluator_bolts):
    """
    This class is wrapper around SSLOnlineEvaluator class from lightning-bolts.
    It adds a the following parameters:
      learning_rate: to adjust the LR of the optimizer
      loss_weight: to add a custom weighting on the loss function
      start_epoch: to start the online learning with a delay
    hidden_dim is renamed to n_hidden
    """

    def __init__(
        self,
        dataset: str = "",
        drop_p: float = 0.2,
        n_hidden: Optional[int] = None,
        z_dim: int = None,
        num_classes: int = None,
        learning_rate: float = 1e-4,
        loss_weight: Optional[torch.FloatTensor] = None,
        start_epoch: int = 0,
    ):
        super().__init__(dataset, drop_p, n_hidden, z_dim, num_classes)

        self.learning_rate = learning_rate
        self.loss_weight = loss_weight
        self.start_epoch = start_epoch

    def on_pretrain_routine_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        super().on_pretrain_routine_start(trainer, pl_module)

        for g in self.optimizer.param_groups:
            g["lr"] = self.learning_rate

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if trainer.current_epoch >= self.start_epoch:
            x, y = self.to_device(batch, pl_module.device)

            with torch.no_grad():
                representations = self.get_representations(pl_module, x)

            representations = representations.detach()

            # forward pass
            mlp_preds = pl_module.non_linear_evaluator(representations)  # type: ignore[operator]
            weight = (
                None
                if self.loss_weight is None
                else self.loss_weight.to(pl_module.device).float()
            )
            mlp_loss = F.cross_entropy(mlp_preds, y, weight=weight)

            # update finetune weights
            mlp_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # log metrics
            train_acc = accuracy(mlp_preds, y)
            pl_module.log("online_train_acc", train_acc, on_step=True, on_epoch=False)
            pl_module.log("online_train_loss", mlp_loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
