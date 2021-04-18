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
from solarnet.logging.tracking import NeptuneNewTracking, Tracking
from solarnet.models import CNNClassification, CNNRegression
from solarnet.utils.metrics import stats_metrics
from solarnet.utils.plots import plot_confusion_matrix, plot_image_grid, plot_regression_line
from solarnet.utils.scaling import log_min_max_inverse_scale, log_min_max_scale
from solarnet.utils.target import flux_to_class_builder
from solarnet.utils.yaml import load_yaml, write_yaml

logger = logging.getLogger(__name__)


def test(parameters: dict, verbose: bool = False):
    logger.info("Testing...")

    seed_everything(parameters["seed"])

    ds_path = Path("data/sdo-benchmark")
    model_path = Path(parameters["path"])
    regression = parameters['data']['targets'] == "regression"
    labels = None if regression else [list(x.keys())[0] for x in parameters['data']['targets']['classes']]
    parameters["gpus"] = min(1, parameters["gpus"])
    reg_tt = log_min_max_scale
    target_transform = reg_tt if regression else flux_to_class_builder(parameters['data']['targets']['classes'])

    # Tracking
    tracking_path = model_path / "tracking.yaml"
    tracking: Optional[Tracking] = None
    if parameters["tracking"] and tracking_path.exists():
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
        target_transform=target_transform,
        time_steps=parameters['data']['time_steps'],
    )
    datamodule.setup('test')
    logger.info(f"Data format: {datamodule.size()}")

    model_class = CNNRegression if regression else CNNClassification
    model = model_class.load_from_checkpoint(str(model_path / "model.ckpt"))
    logger.info(f"Model: {model}")

    trainer = pl.Trainer(
        gpus=parameters["gpus"],
        logger=None,
    )

    # Evaluate model
    raw_metrics = trainer.test(model, datamodule=datamodule, verbose=verbose)

    if regression:
        metrics = {
            'mae': raw_metrics[0]["test_mae"],
            'mse': raw_metrics[0]["test_mse"],
        }
    else:
        tp = raw_metrics[0]["test_tp"]  # hits
        fp = raw_metrics[0]["test_fp"]  # false alarm
        tn = raw_metrics[0]["test_tn"]  # correct negative
        fn = raw_metrics[0]["test_fn"]  # miss

        metrics = {
            "accuracy": raw_metrics[0]["test_accuracy"],
            "balanced_accuracy": raw_metrics[0]["test_recall"],
            "f1": raw_metrics[0]["test_f1"],
            **stats_metrics(tp, fp, tn, fn)
        }

    write_yaml(model_path / "metrics.yaml", metrics)
    if tracking:
        tracking.log_metrics(metrics, "metrics/test")

    # Prepare a set of test samples
    model.freeze()
    dataset_image, dataloader = get_random_test_samples_dataloader(
        ds_path,
        transforms=datamodule.transform,
        target_transform=target_transform,
        channel=parameters["data"]["channel"],
        time_steps=parameters['data']['time_steps'],
    )
    y, y_pred = predict(model, dataloader, regression)
    images, _ = map(list, zip(*dataset_image))
    plot_image_grid(images, y, y_pred, labels=labels, save_path=Path(model_path / "test_samples.png"))

    # Confusion matrix or regression line
    y, y_pred = predict(model, datamodule.test_dataloader(), regression)

    if regression:
        plot_path = Path(model_path / "regression_line.png")
        plot_regression_line(y, y_pred, save_path=plot_path)
    else:
        plot_path = Path(model_path / "confusion_matrix.png")
        plot_confusion_matrix(y, y_pred, labels, save_path=plot_path)
    if tracking:
        plot_name = "regression_line" if regression else "confusion_matrix"
        tracking.log_artifact(plot_path, f"metrics/test/{plot_name}")
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


def predict(model, dataloader, is_regression: bool = False):
    if is_regression:
        y_pred = torch.tensor([])
        y = torch.tensor([])

        with torch.no_grad():
            for i in dataloader:
                y_pred = torch.cat((y_pred, model(i[0]).cpu().flatten()))
                y = torch.cat((y, i[1].cpu().flatten()))

        y = log_min_max_inverse_scale(y)
        y_pred = log_min_max_inverse_scale(y_pred)

        return y.tolist(), y_pred.tolist()

    y_pred = []
    y = []

    with torch.no_grad():
        for i in dataloader:
            logits = model(i[0])
            y_pred += torch.argmax(logits, dim=1).tolist()
            y += i[1].tolist()

    return y, y_pred
