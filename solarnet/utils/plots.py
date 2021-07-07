import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torchmetrics.functional import auroc, roc


def plot_confusion_matrix(
    y_true: Union[list, np.ndarray, torch.Tensor],
    y_pred: Union[list, np.ndarray, torch.Tensor],
    labels: list,
    figsize: tuple = (6, 4),
    save_path: Optional[Path] = None,
):
    """
    Print a confusion matrix with number and percentages, in the order given by labels.

    :param y_true: true values
    :param y_pred: predicted values
    :param labels: list of labels
    :param figsize: size of the figure
    :param save_path: optional path where the figure will be saved
    """

    if isinstance(labels[0], str) and not isinstance(y_true[0], str):
        # map string labels to integer (id) in results
        cm_labels = list(range(len(labels)))
    else:
        cm_labels = labels

    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    cm_sum = np.sum(cm, axis=1, keepdims=True).astype(float)
    cm_perc = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
    cm_perc *= 100
    cm_perc = cm_perc.astype(int)

    # Prepare annotations (number of sample and percentages)
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%d/%d\n%.1f%%" % (c, s, p)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%d\n%.1f%%" % (c, p)

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_perc, annot=annot, fmt="", cmap="Blues", annot_kws={"fontsize": 12})

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")


colors = {
    "black": "#000000",
    "green": "#34a853",
    "red": "#af001e",
}


def plot_image_grid(
    images: list,
    y: List[Union[int, float]] = None,
    y_pred: Optional[List[Union[int, float]]] = None,
    y_proba: Optional[List[List[float]]] = None,
    labels: List[str] = None,
    columns: int = 5,
    width: int = 22,
    height: Optional[int] = None,
    max_images: int = 10,
    label_font_size: int = 14,
    save_path: Optional[Path] = None,
):
    """
    Display a grid of images with labels. Compares true labels and predictions if predictions are given.
    The true values and predictions can be regression values (float).

    :param images: list of image (format supported by plt.imshow())
    :param y: list of labels (int) or values (float for regression)
    :param y_pred: list of predictions (int) or values (float for regression)
    :param y_proba: list of probabilities (float) for predictions
    :param labels: list of string labels
    :param columns: number of images to show in a row
    :param width: width of the figure
    :param height: height of the figure, optional. Computed if None.
    :param max_images: Number max of image to show from the given list
    :param label_font_size: Size of the labels
    :param save_path: optional path where the figure will be saved
    """

    def pretty_label(label: int) -> str:
        return label if labels is None else labels[label]

    if len(images) > max_images:
        images = images[0:max_images]

    is_regression = y is not None and not isinstance(y[0], int) and not isinstance(y[0], str)
    if is_regression:
        pretty_label = lambda x: f'{x:.1e}'

    height = width * math.ceil(len(images) / columns) / columns * 1.33

    plt.figure(figsize=(width, height))
    # plt.subplots_adjust(wspace=0.05)
    # plt.subplots_adjust(hspace=0.2)

    if y_proba is not None and not is_regression:
        # Take probability of y_pred
        y_proba = [p[y_pred[i]] for i, p in enumerate(y_proba)]

    for i, image in enumerate(images):
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)
        plt.axis("off")

        if y is not None:
            if y_pred is None:
                title = pretty_label(y[i])
                color = colors["black"]
            elif is_regression:
                title = f"y_true: {pretty_label(y[i])}\ny_pred: {pretty_label(y_pred[i])}"
                color = colors["black"]
            else:
                is_correct = y[i] == y_pred[i]
                if is_correct:
                    title = f"y_true & y_pred: {pretty_label(y[i])}"
                    color = colors["green"]
                else:
                    title = f"y_true: {pretty_label(y[i])} / y_pred: {pretty_label(y_pred[i])}"
                    color = colors["red"]
                if y_proba is not None:
                    title += f" ({y_proba[i]:.2f})"
            plt.title(title, fontsize=label_font_size, color=color, wrap=True)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")


def plot_train_val_curve(
    metrics: Dict[str, List[Dict[str, Union[float, int]]]],
    metric: str,
    key_suffix: str = "",
    y_lim=None,
    step_name: str = "Steps",
    smooth_factor=0.0,
    save_path: Path = None,
):
    """
    Plot the "metric" curve of training and validation.
    The metrics dict should have keys "train_{metric}{key_suffix}" and "val_{metric}{key_suffix}",
     each with a list as value. Lists should have "value" and step key. The step could be an arbitrary step,
     a batch number or an epoch and is used to align training and validation curves.

    :param metrics: A dict of train/val metrics, with list of values per step.
    :param metric: A string for the name of the metric. Also used as key in the metrics dict.
    :param key_suffix: A suffix for the key in the metrics dict. E.g. "_epoch" or "_batch".
    :param y_lim: An optional array (2 entries) to specify y-axis limits. Default to [0, 1].
    :param step_name: The name to give to the step axis on the plot. Default to "Steps".
    :param smooth_factor: A factor for smoothing the plot in [0, 1]. Default to 0 (no smoothing).
    :param save_path: optional path where the figure will be saved.
    """

    train_metric = metrics[f"train_{metric}{key_suffix}"]
    train_metric = {k: [dic[k] for dic in train_metric] for k in train_metric[0]}
    train_metric_steps = train_metric["step"]
    train_metric_values = smooth_curve(train_metric["value"], smooth_factor)

    val_metric = metrics[f"val_{metric}{key_suffix}"]
    val_metric = {k: [dic[k] for dic in val_metric] for k in val_metric[0]}
    val_metric_steps = val_metric["step"]
    val_metric_values = smooth_curve(val_metric["value"], smooth_factor)

    # Compute min and max value for y-axis limits
    if y_lim is None:
        min_value = min(min(train_metric["value"]), min(val_metric["value"]))
        max_value = max(max(train_metric["value"]), max(val_metric["value"]))
        # Add margin around plot
        margin = (max_value - min_value) / 5  # 20%
        min_value -= margin
        max_value += margin
        y_lim = [min_value, max_value]
        # Use up/down limit for known metrics (loss, accuracy)
        if metric == "loss":
            y_lim = [0, max_value]
        elif metric == "accuracy":
            y_lim = [min_value, 1]

    plt.ioff()

    fig = plt.figure(figsize=(8, 6))

    plt.plot(train_metric_steps, train_metric_values, "dodgerblue", label=f"Training {metric}")
    plt.plot(val_metric_steps, val_metric_values, "g", label=f"Validation {metric}")  # g is for "solid green line"

    plt.title(f"Training and validation {metric}")
    plt.xlabel(step_name)
    plt.ylabel(metric.capitalize())
    plt.gca().set_ylim(y_lim)
    plt.grid(alpha=0.75)
    plt.legend()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

    plt.close(fig)


def smooth_curve(points, factor=0.0):
    """
    Smooth an list of points by a given factor.
    A factor of 0 does not smooth the curve. A factor of 1 gives a straight line.

    :param points: An iterable of numbers
    :param factor: A factor in [0,1]
    :return: A smoothed list of numbers
    """

    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


def plot_regression_line(
    y_true: Union[list, np.ndarray, torch.Tensor],
    y_pred: Union[list, np.ndarray, torch.Tensor],
    lim: Tuple[float] = (8e-10, 5e-4),
    save_path: Path = None,
    figsize: tuple = (10, 8),
):
    """
    Plot a regression line which shows regression prediction against true values in a scatter plot.

    :param y_true: The true values
    :param y_pred: The prediction (real values / float)
    :param lim: The limit of the axis
    :param save_path: optional path where the figure will be saved.
    :param figsize: size of the figure
    """

    plt.figure(figsize=figsize)

    if not isinstance(y_true, list):
        y_true = y_true.tolist()
    if not isinstance(y_pred, list):
        y_pred = y_pred.tolist()

    # Plot join with scatter and histogram for density information
    g = seaborn.JointGrid(x=y_pred, y=y_true)

    # Add linear line for reference
    line = np.linspace(*lim, 10)
    g.ax_joint.plot(line, line, color="r", label="True values")

    # Add scatterplot
    g.plot_joint(plt.scatter, s=25, edgecolors="#ffffff", label="Predictions")

    # Add histogram with log-scaled bins
    bins = np.logspace(math.log10(lim[0]), math.log10(lim[1]), 100)
    g.plot_marginals(seaborn.histplot, bins=bins)

    # Make axis log-scaled
    g.ax_joint.set(xscale="log", yscale="log", xlim=lim, ylim=lim)
    g.ax_marg_x.set_xscale('log')
    g.ax_marg_y.set_yscale('log')

    # Add title, labels, legend, grid
    plt.suptitle("Regression line of predictions", y=1.01)
    g.set_axis_labels("Predicted peak emission flux [W/m\N{SUPERSCRIPT TWO}]",
                      "True peak emission flux [W/m\N{SUPERSCRIPT TWO}]")
    g.ax_joint.legend()
    g.ax_joint.grid(alpha=0.75)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')


def plot_roc_curve(
    y: Union[list, torch.Tensor],
    y_proba: Union[list, torch.Tensor],
    n_class: int = 2,
    save_path: Path = None,
    figsize: tuple = (10, 8),
):
    """
    Print a roc curve with auroc value.

    :param y_true: true values
    :param y_proba: probabilities of predicted values
    :param n_class: number of classes
    :param figsize: size of the figure
    :param save_path: optional path where the figure will be saved
    """

    if isinstance(y, list):
        y = torch.tensor(y)
    if isinstance(y_proba, list):
        y_proba = torch.tensor(y_proba)

    plt.figure(figsize=figsize)

    fpr, tpr, _ = roc(y_proba, y, num_classes=n_class)
    auroc_value = auroc(y_proba, y, num_classes=n_class)

    if isinstance(fpr, list) and len(fpr) == 2:
        # Take only positive class for multi-class with 2 class (equivalent to binary case)
        fpr = fpr[1]
        tpr = tpr[1]
    if isinstance(fpr, list) and len(fpr) > 2:
        raise ValueError("ROC curve not implemented for multiclass")

    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot([], [], ' ', label=f"AUROC = {auroc_value:.3f}")
    plt.xlim([-0.005, 1.005])
    plt.ylim([-0.005, 1.005])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.grid(alpha=0.75)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_histogram(
    hist: torch.Tensor,
    min: float = 0.0,
    max: float = 1.0,
    save_path: Path = None,
    figsize: tuple = (10, 8),
):
    """
    Print an histogram of values.

    :param hist: histogram of values
    :param min: minimum of values appearing in histogram
    :param max: maximum of values appearing in histogram
    :param figsize: size of the figure
    :param save_path: optional path where the figure will be saved
    """

    plt.figure(figsize=figsize)

    n_bins = len(hist)
    bins = torch.linspace(min, max, n_bins)
    plt.bar(bins, hist, width=max / n_bins * 0.5)
    plt.ylabel('Number of values in each bin')
    plt.grid(alpha=0.75)

    if save_path is None:
        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(save_path)
