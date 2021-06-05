from torch import nn


def conv_block(input_size, output_size, activation: str, dropout: float, pooling: int = 2, *args, **kwargs):
    if activation == "leakyrelu":
        activation_fn = nn.LeakyReLU()
    elif activation == "selu":
        activation_fn = nn.SELU()
    elif activation == "tanh":
        activation_fn = nn.Tanh()
    elif activation == "prelu":
        activation_fn = nn.PReLU()
    elif activation == "relu6":
        activation_fn = nn.ReLU6()
    else:
        activation_fn = nn.ReLU()

    module = nn.Sequential()
    module.add_module("conv", nn.Conv2d(input_size, output_size, *args, **kwargs))
    module.add_module("batchnorm", nn.BatchNorm2d(output_size))
    if pooling > 0:
        module.add_module("pooling", nn.MaxPool2d((pooling, pooling)))
    module.add_module("activation", activation_fn)
    if dropout > 0:
        module.add_module("dropout", nn.Dropout2d(dropout))

    return module


class SimpleCNN(nn.Module):
    """
    A simple CNN model with a configurable number convolution blocks and parameters. Number of filters is doubled in each block.
    A convolution block is composed of: Conv2d, BatchNorm2d,
      optional MaxPool2d, optional Dropout2d.
    You may want to add a classification head, this is just the backbone.
      See solarnet.models.classifier.Classifier.
      The size of output is available in output_size property.

    :param channels: number of input channels
    :param n_filter: number of filter in the first convolution.
    :param n_block: number of convolution block ()
    :param activation: type of activation: leakyrelu, selu, tanh,
                       prelu, relu6 or relu
    :param dropout: dropout rate
    :param pooling: size of the pooling (0 for no pooling)
    :param stride: size of the stride for Conv2d
    :param kernel_size: size of the kernel for Conv2d
    """

    def __init__(
        self,
        channels: int,
        n_filter: int = 16,
        n_block: int = 3,
        activation: str = "relu",
        dropout: float = 0.5,
        pooling: int = 2,
        stride: int = 3,
        kernel_size: int = 3,
    ):
        super().__init__()

        input_size = channels
        output_size = n_filter
        self.conv_blocks = nn.Sequential()
        for i in range(n_block):
            self.conv_blocks.add_module(
                f"conv_block_{i}",
                conv_block(
                    input_size,
                    output_size,
                    kernel_size=kernel_size,
                    padding=0,
                    stride=stride,
                    activation=activation,
                    dropout=dropout,
                    pooling=pooling,
                ),
            )
            self._output_size = output_size

            input_size = output_size
            output_size = input_size * 2

    def forward(self, image):
        return self.conv_blocks(image)

    @property
    def output_size(self):
        return self._output_size
