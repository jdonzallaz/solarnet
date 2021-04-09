import logging
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, Subset

from solarnet.data.sdo_benchmark_datamodule import SDOBenchmarkDataModule
from solarnet.data.sdo_benchmark_dataset import SDOBenchmarkDataset
from solarnet.models.baseline import CNN
from solarnet.utils.plots import image_grid, plot_confusion_matrix
from solarnet.utils.target import flux_to_class_builder
from solarnet.utils.tracking import NeptuneNewTracking, Tracking
from solarnet.utils.yaml import load_yaml, write_yaml

logger = logging.getLogger(__name__)


def test(parameters: dict):
    logger.info("Testing...")

    seed_everything(parameters["seed"])

    ds_path = Path("data/sdo-benchmark")
    model_path = Path("models/baseline/")
    labels = [list(x.keys())[0] for x in parameters['data']['targets']['classes']]
    parameters["gpus"] = min(1, parameters["gpus"])

    # Tracking
    tracking_path = model_path / "tracking.yaml"
    tracking: Optional[Tracking] = None
    if tracking_path.exists():
        tracking_data = load_yaml(tracking_path)
        run_id = tracking_data["run_id"]
        tracking = NeptuneNewTracking.resume(run_id)

    datamodule = SDOBenchmarkDataModule(
        ds_path,
        batch_size=parameters["trainer"]["batch_size"],
        validation_size=parameters["data"]["validation_size"],
        channel=parameters["data"]["channel"],
        resize=parameters["data"]["size"],
        seed=parameters["seed"],
        num_workers=0 if os.name == 'nt' else 4,  # Windows supports only 1, Linux supports more
        target_transform=flux_to_class_builder(parameters['data']['targets']['classes']),
        time_steps=parameters['data']['time_steps'],
    )
    datamodule.setup('test')
    logger.info(f"Data format: {datamodule.size()}")

    model = CNN.load_from_checkpoint(str(model_path / "model.ckpt"))
    logger.info(f"Model: {model}")

    trainer = pl.Trainer(
        gpus=parameters["gpus"],
    )

    # Evaluate model
    raw_metrics = trainer.test(model, datamodule=datamodule, verbose=True)

    tp = raw_metrics[0]["test_tp"]  # hits
    fp = raw_metrics[0]["test_fp"]  # false alarm
    tn = raw_metrics[0]["test_tn"]  # correct negative
    fn = raw_metrics[0]["test_fn"]  # miss
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Write metrics
    # Metric computation is inspired by hydrogo/rainymotion.
    metrics = {
        # Accuracy
        "accuracy": raw_metrics[0]["test_accuracy"],
        # Balanced accuracy is the recall using average=macro
        "balanced_accuracy": raw_metrics[0]["test_recall"],
        # F1-score macro
        "f1": raw_metrics[0]["test_f1"],
        # False Alarm Rate - computation inspired by hydrogo/rainymotion
        "far": fp / (tp + fp),
        # Heidke Skill Score - computation inspired by hydrogo/rainymotion
        "hss": (2 * (tp * tn - fn * fp)) / (fn ** 2 + fp ** 2 + 2 * tp * tn + (fn + fp) * (tp + tn)),
        # Probability Of Detection - computation inspired by hydrogo/rainymotion
        "pod": sensitivity,
        # Critical Success Index - computation inspired by hydrogo/rainymotion
        "csi": tp / (tp + fn + fp),
        # True Skill Statistic
        "tss": sensitivity + specificity - 1,
    }
    write_yaml(model_path / "metrics.yaml", metrics)
    if tracking:
        tracking.log_metrics(metrics, "metrics/test")

    # Prepare a set of test samples
    model.freeze()
    dataset_image, dataloader = get_random_test_samples_dataloader(
        ds_path,
        transforms=datamodule.transform,
        target_transform=flux_to_class_builder(parameters['data']['targets']['classes']),
        channel=parameters["data"]["channel"],
        time_steps=parameters['data']['time_steps'],
    )
    y_pred, _ = predict(model, dataloader)
    images, y = map(list, zip(*dataset_image))
    image_grid(images, y, y_pred, labels=labels, path=Path(model_path / "test_samples.png"))

    # Confusion matrix
    y_pred, y = predict(model, datamodule.test_dataloader())
    confusion_matrix_path = Path(model_path / "confusion_matrix.png")
    plot_confusion_matrix(y, y_pred, labels, path=confusion_matrix_path)
    if tracking:
        tracking.log_artifact(confusion_matrix_path, "metrics/test/confusion_matrix")
        tracking.end()


def get_random_test_samples_dataloader(
    ds_path: Path,
    nb_sample: int = 10,
    channel: str = '171',
    transforms: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    time_steps: Union[int, List[int]] = 0,
) -> (Dataset, DataLoader):
    """ Return a random set of test samples """

    dataset_test_image = SDOBenchmarkDataset(
        ds_path / 'test' / 'meta_data.csv',
        ds_path / 'test',
        transform=None,
        target_transform=target_transform,
        channel=channel
    )
    dataset_test_tensors = SDOBenchmarkDataset(
        ds_path / 'test' / 'meta_data.csv',
        ds_path / 'test',
        transform=transforms,
        target_transform=target_transform,
        channel=channel,
        time_steps=time_steps
    )

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
