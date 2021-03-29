import logging
import random
from pathlib import Path
from typing import Callable, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, Subset

from solarnet.data.sdo_benchmark_datamodule import SDOBenchmarkDataModule
from solarnet.data.sdo_benchmark_dataset import SDOBenchmarkDataset
from solarnet.models.baseline import CNN
from solarnet.utils.physics import flux_to_binary_class
from solarnet.utils.plots import image_grid, plot_confusion_matrix
from solarnet.utils.yaml import write_yaml

logger = logging.getLogger(__name__)


def test(parameters: dict):
    logger.info("Testing...")

    seed_everything(parameters["seed"])

    ds_path = Path("data/sdo-benchmark")
    model_path = Path("models/baseline/")
    labels = ['Quiet', '>=C']

    datamodule = SDOBenchmarkDataModule(
        ds_path,
        batch_size=parameters["trainer"]["batch_size"],
        validation_size=parameters["data"]["validation_size"],
        channel=parameters["data"]["channel"],
        resize=parameters["data"]["size"],
        seed=parameters["seed"],
    )
    datamodule.setup()
    logger.info(f"Data format: {datamodule.size()}")

    model = CNN.load_from_checkpoint(str(model_path / "model.ckpt"))
    logger.info(f"Model: {model}")

    trainer = pl.Trainer(
        gpus=parameters["gpus"],
        limit_test_batches=10,
    )

    # Evaluate model
    raw_metrics = trainer.test(model, datamodule=datamodule, verbose=True)

    # Write metrics
    metrics = {
        "accuracy": raw_metrics[0]["test_accuracy"],
        "f1": raw_metrics[0]["test_f1"],
    }
    write_yaml(model_path / "metrics.yaml", metrics)

    # Prepare a set of test samples
    model.freeze()
    dataset_image, dataloader = get_random_test_samples_dataloader(
        ds_path, transforms=datamodule.transform, target_transform=flux_to_binary_class,
        channel=parameters["data"]["channel"])
    y_pred, _ = predict(model, dataloader)
    images, y = map(list, zip(*dataset_image))
    image_grid(images, y, y_pred, labels=labels, path=Path(model_path / "test_samples.png"))

    # Confusion matrix
    y_pred, y = predict(model, datamodule.test_dataloader())
    plot_confusion_matrix(y, y_pred, labels, path=Path(model_path / "confusion_matrix.png"))


def get_random_test_samples_dataloader(ds_path: Path, nb_sample: int = 10, channel: str = '171',
                                       transforms: Optional[Callable] = None,
                                       target_transform: Optional[Callable] = None
                                       ) -> (Dataset, DataLoader):
    """ Return a random set of test samples """

    dataset_test_image = SDOBenchmarkDataset(ds_path / 'test' / 'meta_data.csv', ds_path / 'test', transform=None,
                                             target_transform=target_transform, channel=channel)
    dataset_test_tensors = SDOBenchmarkDataset(ds_path / 'test' / 'meta_data.csv', ds_path / 'test',
                                               transform=transforms, target_transform=target_transform, channel=channel)

    subset_indices = [random.randrange(len(dataset_test_image)) for _ in range(nb_sample)]
    subset_images = Subset(dataset_test_image, subset_indices)
    subset_tensors = Subset(dataset_test_tensors, subset_indices)
    dataloader_tensors = DataLoader(subset_tensors, batch_size=nb_sample, num_workers=0, shuffle=False)

    return subset_images, dataloader_tensors


def predict(model, dataloader):
    y_pred = []
    y = []

    with torch.no_grad():
        for i in dataloader:
            logits = model(i[0])
            y_pred += torch.argmax(logits, dim=1).tolist()
            y += i[1]

    return y_pred, y
