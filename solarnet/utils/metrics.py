from typing import Dict, Tuple

import torch
import torchmetrics


def far(tp: int, fp: int) -> float:
    # False Alarm Ratio
    # in [0,1], best at 0
    # Always majority class: error (division by zero)
    # Random: ?
    if tp + fp == 0:
        return 1.0
    return fp / (tp + fp)


def hss(tp: int, fp: int, tn: int, fn: int) -> float:
    # Heidke Skill Score - computation inspired by hydrogo/rainymotion
    # in [-inf,1], best at 1
    # Always majority class: 0
    # Random: 0
    return (2 * (tp * tn - fn * fp)) / (fn ** 2 + fp ** 2 + 2 * tp * tn + (fn + fp) * (tp + tn))


def pod(tp: int, fn: int) -> float:
    # Probability Of Detection - computation inspired by hydrogo/rainymotion
    # in [0,1], best at 1
    # Always majority class: 0
    # Random: 0.5
    return tp / (tp + fn)


def csi(tp: int, fp: int, fn: int) -> float:
    # Critical Success Index - computation inspired by hydrogo/rainymotion
    # in [0,1], best at 1
    # Always majority class: 0
    # Random: {percentage of majority class}
    return tp / (tp + fn + fp)


def tss(tp: int, fp: int, tn: int, fn: int) -> float:
    # True Skill Statistic
    # also computed as sensitivity + specificity - 1
    # in [-1,1], best at 1, no skill at 0
    # Always majority class: 0
    # Random: 0
    return tp / (tp + fn) + tn / (tn + fp) - 1


def accuracy(y: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torchmetrics.functional.accuracy(y_pred, y).item()


def balanced_accuracy(y: torch.Tensor, y_pred: torch.Tensor, n_class: int = 2) -> float:
    # Balanced accuracy is defined as the average of recall obtained on each class by sklearn
    return torchmetrics.functional.recall(y_pred, y, average="macro", num_classes=n_class).item()


def f1(y: torch.Tensor, y_pred: torch.Tensor, n_class: int = 2) -> float:
    return torchmetrics.functional.f1(y_pred, y, average="macro", num_classes=n_class).item()


def mae(y: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torchmetrics.functional.mean_absolute_error(y_pred, y)


def mse(y: torch.Tensor, y_pred: torch.Tensor) -> float:
    return torchmetrics.functional.mean_squared_error(y_pred, y)


def stat_scores(y: torch.Tensor, y_pred: torch.Tensor, n_class: int = 2) -> Tuple[int, int, int, int]:
    tp, fp, tn, fn, sup = torchmetrics.functional.stat_scores(
        y_pred, y, num_classes=n_class if n_class > 2 else 1, is_multiclass=n_class > 2, reduce="micro").tolist()
    return tp, fp, tn, fn


def stats_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    return {
        "far": far(tp, fp),
        "hss": hss(tp, fp, tn, fn),
        "pod": pod(tp, fn),
        "csi": csi(tp, fp, fn),
        "tss": tss(tp, fp, tn, fn),
    }


def classification_metrics(y: torch.Tensor, y_pred: torch.Tensor, n_class: int = 2) -> Dict[str, float]:
    tp, fp, tn, fn = stat_scores(y, y_pred, n_class)
    sm = stats_metrics(tp, fp, tn, fn)

    return {
        "accuracy": accuracy(y, y_pred),
        "balanced_accuracy": balanced_accuracy(y, y_pred, n_class),
        "f1": f1(y, y_pred, n_class),
        **sm,
    }


def regression_metrics(y: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    return {
        "mae": mae(y, y_pred),
        "mse": mse(y, y_pred),
    }
