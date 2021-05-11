import torch
from pytorch_lightning import LightningDataModule, LightningModule

from solarnet.models import CNNClassification, CNNRegression
from solarnet.utils.target import compute_class_weight


def model_from_config(parameters: dict, datamodule: LightningDataModule) -> LightningModule:
    steps_per_epoch = len(datamodule.train_dataloader())
    total_steps = parameters['trainer']['epochs'] * steps_per_epoch

    if parameters["data"]["name"] == "sdo-dataset":
        class_weight = torch.FloatTensor([1.0, 0.5])
    else:
        class_weight = compute_class_weight(datamodule.dataset_train)

    regression = parameters["data"]['targets'] == "regression"
    if regression:
        model = CNNRegression(*datamodule.size(),
                              learning_rate=parameters['trainer']['learning_rate'],
                              class_weight=class_weight,
                              total_steps=total_steps,
                              activation=parameters['model']['activation'])
    else:
        model = CNNClassification(*datamodule.size(),
                                  n_class=len(parameters['data']['targets']['classes']),
                                  learning_rate=parameters['trainer']['learning_rate'],
                                  class_weight=class_weight,
                                  total_steps=total_steps,
                                  activation=parameters['model']['activation'])

    return model
