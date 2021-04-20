from sklearn.metrics import mean_squared_error
import torch

from solarnet.utils.metrics import regression_metrics, tss


# TODO: test other metrics


def test_regression_metrics():
    y_true = torch.tensor([0, 1, 0, 1, 0, 1])
    y_false = torch.tensor([1, 0, 1, 0, 1, 0])
    y_half = torch.tensor([1, 1, 1, 1, 1, 1])
    y_pred = torch.tensor([0, 1, 0, 1, 0, 1])

    assert regression_metrics(y_true, y_pred) == {
        "mae": 0,
        "mse": 0,
    }
    assert regression_metrics(y_false, y_pred) == {
        "mae": 1,
        "mse": 1,
    }
    assert regression_metrics(y_half, y_pred) == {
        "mae": 0.5,
        "mse": 0.5,
    }


def test_tss():
    assert tss(10, 0, 10, 0) == 1
    assert tss(0, 10, 0, 10) == -1
    assert tss(10, 10, 0, 0) == 0
    assert tss(10, 10, 10, 10) == 0
