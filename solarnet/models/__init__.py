from solarnet.models.cnn_classification import CNNClassification
from solarnet.models.cnn_regression import CNNRegression
from solarnet.models.model_config import model_from_config

__all__ = [
    CNNClassification,
    CNNRegression,
    model_from_config,
]
