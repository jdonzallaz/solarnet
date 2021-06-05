import logging
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import transforms

from solarnet.data import datamodule_from_config, dataset_from_config
from solarnet.tracking import NeptuneNewTracking, Tracking
from solarnet.models import ImageClassification, ImageRegression
from solarnet.utils.metrics import stats_metrics
from solarnet.utils.plots import plot_confusion_matrix, plot_image_grid, plot_regression_line, plot_roc_curve
from solarnet.utils.scaling import log_min_max_inverse_scale
from solarnet.utils.yaml import load_yaml, write_yaml

logger = logging.getLogger(__name__)


def test(parameters: dict, verbose: bool = False):
    logger.info("Testing...")

    seed_everything(parameters["seed"])

    model_path = Path(parameters["path"])
    plot_path = Path(parameters["path"]) / "test_plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    metadata_path = model_path / "metadata.yaml"
    metadata = load_yaml(metadata_path) if metadata_path.exists() else None

    regression = parameters["data"]["targets"] == "regression"
    labels = None if regression else [list(x.keys())[0] for x in parameters["data"]["targets"]["classes"]]
    n_class = 1 if regression else len(parameters["data"]["targets"]["classes"])
    parameters["system"]["gpus"] = min(1, parameters["system"]["gpus"])

    # Tracking
    tracking: Optional[Tracking] = None
    if parameters["tracking"] and metadata is not None and metadata["tracking_id"] is not None:
        run_id = metadata["tracking_id"]
        tracking = NeptuneNewTracking.resume(run_id)

    datamodule = datamodule_from_config(parameters)
    datamodule.setup("test")
    logger.info(f"Data format: {datamodule.size()}")

    model_class = ImageRegression if regression else ImageClassification
    model = model_class.load_from_checkpoint(str(model_path / "model.ckpt"))
    logger.info(f"Model: {model}")

    trainer = pl.Trainer(
        gpus=parameters["system"]["gpus"],
        logger=None,
    )

    # Evaluate model
    raw_metrics = trainer.test(model, datamodule=datamodule, verbose=verbose)
    raw_metrics = raw_metrics[0]

    if regression:
        metrics = {
            "mae": raw_metrics["test_mae"],
            "mse": raw_metrics["test_mse"],
        }
    else:
        tp = raw_metrics.pop("test_tp")  # hits
        fp = raw_metrics.pop("test_fp")  # false alarm
        tn = raw_metrics.pop("test_tn")  # correct negative
        fn = raw_metrics.pop("test_fn")  # miss

        metrics = {"balanced_accuracy": raw_metrics.pop("test_recall"), **stats_metrics(tp, fp, tn, fn)}

        for key, value in raw_metrics.items():
            metrics[key[len("test_") :]] = value
        metrics = dict(sorted(metrics.items()))

    write_yaml(model_path / "metrics.yaml", metrics)
    if tracking:
        tracking.log_metrics(metrics, "metrics/test")

    # Prepare a set of test samples
    model.freeze()
    nb_image_grid = 10
    dataset_image, dataloader = get_random_test_samples_dataloader(
        parameters,
        transform=datamodule.transform,
        nb_sample=nb_image_grid,
        classes=None if regression else list(range(n_class)),
    )
    y, y_pred, y_proba = predict(model, dataloader, regression, return_proba=True)
    images, _ = map(list, zip(*dataset_image))
    plot_image_grid(
        images,
        y,
        y_pred,
        y_proba,
        labels=labels,
        save_path=Path(plot_path / "test_samples.png"),
        max_images=nb_image_grid,
    )
    if tracking:
        tracking.log_artifact(plot_path / "test_samples.png", "metrics/test/test_samples")

    # Confusion matrix or regression line
    y, y_pred, y_proba = predict(model, datamodule.test_dataloader(), regression, return_proba=True)

    if regression:
        plot_path = Path(plot_path / "regression_line.png")
        plot_regression_line(y, y_pred, save_path=plot_path)

        if tracking:
            tracking.log_artifact(plot_path, "metrics/test/regression_line")
    else:
        # Confusion matrix
        confusion_matrix_path = Path(plot_path / "confusion_matrix.png")
        plot_confusion_matrix(y, y_pred, labels, save_path=confusion_matrix_path)
        # Roc curve
        if n_class <= 2:
            roc_curve_path = Path(plot_path / "roc_curve.png")
            plot_roc_curve(y, y_proba, n_class=n_class, save_path=roc_curve_path, figsize=(7, 5))

        if tracking:
            tracking.log_artifact(confusion_matrix_path, "metrics/test/confusion_matrix")
            if n_class <= 2:
                tracking.log_artifact(roc_curve_path, "metrics/test/roc_curve")

    if tracking:
        tracking.end()


def get_random_test_samples_dataloader(
    parameters: dict,
    nb_sample: int = 10,
    transform: Optional[Callable] = None,
    classes: Optional[List[int]] = None,
) -> Tuple[Dataset, DataLoader]:
    """ Return a random set of test samples """

    dataset_test_image = dataset_from_config(
        parameters, "test", transforms.Compose([transform, transforms.Normalize([-1], [2]), transforms.ToPILImage()])
    )
    dataset_test_tensors = dataset_from_config(parameters, "test", transform)

    if classes is not None:
        y = dataset_test_image.y()
        y = torch.tensor(y)

        # Find number of sample to choose for each class ((nb_sample / len(classes) +/- 1)
        split = (torch.arange(nb_sample + len(classes) - 1, nb_sample - 1, -1) // len(classes)).tolist()
        subset_indices = []
        for i, class_ in enumerate(classes):
            # Indices in the dataset corresponding to this class
            indices_for_class = torch.where(y == class_)[0]
            # Add random indices corresponding to this class
            subset_indices += indices_for_class[torch.randint(len(indices_for_class), (split[i],))].tolist()
    else:
        subset_indices = [random.randrange(len(dataset_test_image)) for _ in range(nb_sample)]

    subset_images = Subset(dataset_test_image, subset_indices)
    subset_tensors = Subset(dataset_test_tensors, subset_indices)
    dataloader_tensors = DataLoader(subset_tensors, batch_size=nb_sample, num_workers=0, shuffle=False)

    return subset_images, dataloader_tensors


def predict(model, dataloader, is_regression: bool = False, return_proba: bool = False):
    if is_regression:
        y_pred = torch.tensor([])
        y = torch.tensor([])

        with torch.no_grad():
            for i in dataloader:
                y_pred = torch.cat((y_pred, model(i[0]).cpu().flatten()))
                y = torch.cat((y, i[1].cpu().flatten()))

        y = log_min_max_inverse_scale(y)
        y_pred = log_min_max_inverse_scale(y_pred)

        if return_proba:
            return y.tolist(), y_pred.tolist(), None

        return y.tolist(), y_pred.tolist()

    y_pred = []
    y_proba = []
    y = []

    with torch.no_grad():
        for i in dataloader:
            logits = model(i[0])
            y_pred += torch.argmax(logits, dim=1).tolist()
            y_proba += F.softmax(logits, dim=1).tolist()
            y += i[1].tolist()

    if return_proba:
        return y, y_pred, y_proba

    return y, y_pred
