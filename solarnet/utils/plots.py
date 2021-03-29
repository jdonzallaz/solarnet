from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true: Union[list, np.ndarray, torch.Tensor], y_pred: Union[list, np.ndarray, torch.Tensor],
                          labels: list, figsize: tuple = (6, 4), path: Optional[Path] = None):
    """
    Print a confusion matrix with number and percentages, in the order given by labels.

    :param y_true: true values
    :param y_pred: predicted values
    :param labels: list of labels
    :param figsize: size of the figure
    :param path: optional path where the figure will be saved
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
                annot[i, j] = '%d/%d\n%.1f%%' % (c, s, p)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%d\n%.1f%%' % (c, p)

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_perc, annot=annot, fmt='', cmap='Blues', annot_kws={"fontsize": 12})

    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')


colors = {
    'black': '#000000',
    'green': '#34a853',
    'red': '#af001e',
}


def image_grid(images: list, y: List[int], y_pred: Optional[List[int]] = None, labels: List[str] = None,
               columns: int = 5, width: int = 20, height: int = 6, max_images: int = 10, label_font_size: int = 14,
               path: Optional[Path] = None):
    """
    Display a grid of images with labels. Compares true labels and predictions if predictions are given

    :param images: list of image (format supported by plt.imshow())
    :param y: list of labels (int)
    :param y_pred: list of predictions (int)
    :param labels: list of string labels
    :param columns: number of images to show in a row
    :param width: width of the figure
    :param height: height of the figure
    :param max_images: Number max of image to show from the given list
    :param label_font_size: Size of the labels
    :param path: optional path where the figure will be saved
    """

    def pretty_label(label: int) -> str:
        return label if labels is None else labels[label]

    if len(images) > max_images:
        images = images[0:max_images]

    height = max(height, int(len(images) / columns) * height)

    plt.figure(figsize=(width, height))
    plt.subplots_adjust(wspace=0.05)
    plt.subplots_adjust(hspace=0.2)

    for i, image in enumerate(images):
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)
        plt.axis('off')

        if y_pred is None:
            title = pretty_label(y[i])
            color = colors['black']
        else:
            is_correct = y[i] == y_pred[i]
            if is_correct:
                title = f"y_true & y_pred: {pretty_label(y[i])}"
                color = colors['green']
            else:
                title = f"y_true: {pretty_label(y[i])} / y_pred: {pretty_label(y_pred[i])}"
                color = colors['red']
        plt.title(title, fontsize=label_font_size, color=color)

    if path is None:
        plt.show()
    else:
        plt.savefig(path, bbox_inches='tight')
