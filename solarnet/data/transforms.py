import math
from typing import Callable, Optional, Union

import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as transforms_functional

# Same preprocess as github.com/i4Ds/SDOBenchmark
CHANNEL_PREPROCESS = {
    "94": {"min": 0.1, "max": 800, "scaling": "log10"},
    "131": {"min": 0.7, "max": 1900, "scaling": "log10"},
    "171": {"min": 5, "max": 3500, "scaling": "log10"},
    "193": {"min": 20, "max": 5500, "scaling": "log10"},
    "211": {"min": 7, "max": 3500, "scaling": "log10"},
    "304": {"min": 0.1, "max": 3500, "scaling": "log10"},
    "335": {"min": 0.4, "max": 1000, "scaling": "log10"},
    "1600": {"min": 10, "max": 800, "scaling": "log10"},
    "1700": {"min": 220, "max": 5000, "scaling": "log10"},
    "4500": {"min": 4000, "max": 20000, "scaling": "log10"},
    "continuum": {"min": 0, "max": 65535, "scaling": None},
    "magnetogram": {"min": -250, "max": 250, "scaling": None},
    "bx": {"min": -250, "max": 250, "scaling": None},
    "by": {"min": -250, "max": 250, "scaling": None},
    "bz": {"min": -250, "max": 250, "scaling": None},
}


def sdo_dataset_normalize(channel: Union[str, int], resize: Optional[int] = None):
    """
    Apply the normalization necessary for the sdo-dataset. Depending on the channel, it:
      - flip the image vertically
      - clip the "pixels" data in the predefined range (see above)
      - apply a log10() on the data
      - normalize the data to the [0, 1] range
      - normalize the data around 0 (standard scaling)

    :param channel: The kind of data to preprocess
    :param resize: Optional size of image (integer) to resize the image
    :return: a transforms object to preprocess tensors
    """

    preprocess_config = CHANNEL_PREPROCESS[str(channel).lower()]

    lambda_transform = lambda x: torch.clamp(
        transforms_functional.vflip(x),
        min=preprocess_config["min"],
        max=preprocess_config["max"],
    )

    mean = preprocess_config["min"]
    std = preprocess_config["max"] - preprocess_config["min"]

    if preprocess_config["scaling"] == "log10":
        base_lambda = lambda_transform
        lambda_transform = lambda x: torch.log10(base_lambda(x))
        mean = math.log10(preprocess_config["min"])
        std = math.log10(preprocess_config["max"]) - math.log10(preprocess_config["min"])

    transform = [
        transforms.Lambda(lambda_transform),
        transforms.Normalize(mean=[mean], std=[std]),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]

    if resize is not None:
        transform.insert(0, transforms.Resize(resize))

    return transforms.Compose(transform)


class SDOSimCLRDataTransform:
    """
    Prepare data for self-supervised training using SimCLR.
    This applies augmentations relevant for SDO image data (no color jittering or grayscale):
      - [WIP] random rotation
      - random resized crop
      - random horizontal flip
      - random vertical flip
    When this transform is called, it outputs 2 augmented tensors/image.
    Optionally, it can output a third transformed image with no augmentation (only resized) for online training.

    :param resize: The final image size after cropping
    :param do_online_transform: whether to output a third image for online training
    :param transform_before: optional transform(s) to apply before the augmentation transforms
    :param transform_after: optional transform(s) to apply after the augmentation transforms
    :param prob_random_horizontal_flip: probability for the random horizontal flip
    :param prob_random_vertical_flip: probability for the random vertical flip
    :param min_scale_resize: minimum scale of the cropping (0.5 means half the height of the original image)
    :param max_scale_resize: maximum scale of the cropping
    """

    def __init__(
        self,
        resize: int = 64,
        do_online_transform: bool = False,
        transform_before: Optional[Callable] = None,
        transform_after: Optional[Callable] = None,
        prob_random_horizontal_flip: float = 0.5,
        prob_random_vertical_flip: float = 0.5,
        min_scale_resize: float = 0.5,
        max_scale_resize: float = 1.0,
        # degree_random_rotation: float = 90,
        # prob_random_rotation: float = 0.5,
    ) -> None:

        self.resize = resize
        self.do_online_transform = do_online_transform
        self.transform_before = transform_before
        self.transform_after = transform_after
        self.prob_random_horizontal_flip = prob_random_horizontal_flip
        self.prob_random_vertical_flip = prob_random_vertical_flip
        self.min_scale_resize = min_scale_resize
        self.max_scale_resize = max_scale_resize
        # self.degree_random_rotation = degree_random_rotation
        # self.prob_random_rotation = prob_random_rotation

        _transforms = [
            # transforms.RandomApply([transforms.RandomRotation(self.degree_random_rotation)], self.prob_random_rotation),
            transforms.RandomResizedCrop(self.resize, scale=(min_scale_resize, max_scale_resize)),
            transforms.RandomHorizontalFlip(prob_random_horizontal_flip),
            transforms.RandomVerticalFlip(prob_random_vertical_flip),
        ]
        _online_transforms = [transforms.Resize(self.resize)]

        if transform_before is not None:
            _transforms.insert(0, transform_before)
            _online_transforms.insert(0, transform_before)
        if transform_after is not None:
            _transforms.append(transform_after)
            _online_transforms.append(transform_after)

        self.train_transform = transforms.Compose(_transforms)

        self.online_transform = transforms.Compose(_online_transforms)

    def __call__(self, sample):
        xi = self.train_transform(sample)
        xj = self.train_transform(sample)

        if self.do_online_transform:
            return xi, xj, self.online_transform(sample)
        return xi, xj
