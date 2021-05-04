import torchvision
from torch import nn

from solarnet.models.model_utils import BaseModel


class SqueezeNetModule(BaseModel):
    def __init__(self, channels: int, n_class: int):
        super().__init__()

        # Base model: SqueezeNet 1.1
        model = torchvision.models.squeezenet1_1()

        # Change number of channels in input (other params like original model)
        model.features[0] = nn.Conv2d(channels, 64, kernel_size=(3, 3), stride=(3, 3))

        # Change classifier layers
        # Remove Conv2d-Relu-AdaptiveAvgPool2d
        # Add AdaptiveAvgPool2d-Linear with correct number of classes
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Linear(512, n_class, bias=True),
        )

        self.model = model

    def forward(self, image):
        return self.model(image)
