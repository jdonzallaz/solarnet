from solarnet.models.backbone import get_backbone
from solarnet.models.classifier import Classifier
from solarnet.models.crnn import CRNN
from solarnet.models.image_classification import ImageClassification
from solarnet.models.image_regression import ImageRegression
from solarnet.models.model_utils import BaseModel, download_or_cached_or_local_model_path
from solarnet.models.simclr import SimCLR
from solarnet.models.simple_cnn import SimpleCNN

__all__ = [
    BaseModel,
    Classifier,
    CRNN,
    ImageClassification,
    ImageRegression,
    SimCLR,
    SimpleCNN,
    download_or_cached_or_local_model_path,
    get_backbone,
]
