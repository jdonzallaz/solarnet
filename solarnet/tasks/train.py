import logging
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from pytorch_lightning.callbacks import BackboneFinetuning
from torchvision.transforms import transforms

from solarnet.callbacks import SSLOnlineEvaluator
from solarnet.data.dataset_config import datamodule_from_config
from solarnet.data.sdo_dataset import SDODatasetDataModule
from solarnet.data.transforms import SDOSimCLRDataTransform, sdo_dataset_normalize
from solarnet.models import ImageClassification, ImageRegression, SimCLR
from solarnet.utils.dict import filter_dict_for_function_parameters
from solarnet.utils.target import compute_class_weight, flux_to_class_builder
from solarnet.utils.trainer import train

logger = logging.getLogger(__name__)


def model_from_config(parameters: dict, datamodule: LightningDataModule) -> LightningModule:
    steps_per_epoch = len(datamodule.train_dataloader())
    total_steps = parameters["trainer"]["epochs"] * steps_per_epoch

    if parameters["data"]["name"] == "sdo-dataset":
        class_weight = [0.5, 7]
    else:
        class_weight = compute_class_weight(datamodule.dataset_train)

    regression = parameters["data"]["targets"] == "regression"
    if regression:
        model = ImageRegression(
            n_channel=datamodule.size(0),
            lr_scheduler_total_steps=total_steps,
            **parameters["model"],
        )
    else:
        model = ImageClassification(
            n_channel=datamodule.size(0),
            n_class=len(parameters["data"]["targets"]["classes"]),
            class_weight=class_weight,
            lr_scheduler_total_steps=total_steps,
            **parameters["model"],
        )

    return model


def train_standard(parameters: dict):
    seed_everything(parameters["seed"], workers=True)

    datamodule = datamodule_from_config(parameters)
    datamodule.setup()

    model = model_from_config(parameters, datamodule)

    train(parameters, datamodule, model)


def train_ssl(parameters: dict):
    seed_everything(parameters["seed"], workers=True)

    base_resize = 512
    transform = SDOSimCLRDataTransform(
        parameters["data"]["size"],
        do_online_transform=True,
        transform_before=transforms.CenterCrop((base_resize // 2, base_resize - base_resize // 8)),
        transform_after=sdo_dataset_normalize(parameters["data"]["channel"]),
    )

    dm = SDODatasetDataModule(
        Path(parameters["data"]["path"]),
        transform=transform,
        target_transform=flux_to_class_builder(parameters["data"]["targets"]["classes"]),
        batch_size=parameters["trainer"]["batch_size"],
        num_workers=parameters["system"]["workers"],
    )
    dm.setup()

    model_params = filter_dict_for_function_parameters(parameters["model"], SimCLR.__init__)
    model = SimCLR(
        gpus=parameters["system"]["gpus"],
        num_samples=dm.num_samples,
        batch_size=dm.batch_size,
        n_channel=dm.size(0),
        max_epochs=parameters["trainer"]["epochs"],
        **model_params,
    )

    online_params = filter_dict_for_function_parameters(parameters["ssl"]["online"], SSLOnlineEvaluator.__init__)
    online_finetuner = SSLOnlineEvaluator(
        z_dim=model.hidden_mlp,
        num_classes=len(parameters["data"]["targets"]["classes"]),
        loss_weight=torch.tensor([0.85, 1.15], dtype=float),
        **online_params,
    )

    train(parameters, dm, model, callbacks=online_finetuner)


def finetune(parameters: dict):
    seed_everything(parameters["seed"], workers=True)

    datamodule = datamodule_from_config(parameters)
    datamodule.setup()

    total_steps = parameters["trainer"]["epochs"] * len(datamodule.train_dataloader())

    model = ImageClassification.from_pretrained(
        Path(parameters["finetune"]["base"]),
        n_class=len(parameters["data"]["targets"]["classes"]),
        class_weight=compute_class_weight(datamodule.dataset_train),
        lr_scheduler_total_steps=total_steps,
        **parameters["model"],
        print_incompatible_keys=True,
    )

    callbacks = []
    if parameters["finetune"]["backbone_unfreeze_epoch"] > 0:
        callbacks.append(BackboneFinetuning(parameters["finetune"]["backbone_unfreeze_epoch"], verbose=True))

    train(parameters, datamodule, model, callbacks=callbacks)
