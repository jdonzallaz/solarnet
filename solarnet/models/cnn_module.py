from torch import nn


def conv_block(input_size, output_size, activation: str, *args, **kwargs):
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

    return nn.Sequential(
        nn.Conv2d(input_size, output_size, *args, **kwargs),
        nn.BatchNorm2d(output_size),
        # nn.Conv2d(input_size, output_size, (3, 3)),
        # nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        activation_fn,
        nn.Dropout2d(0.1),
    )


class CNNModule(nn.Module):
    def __init__(self, channels: int, n_class: int, activation: str = 'relu'):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            conv_block(channels, 16, kernel_size=3, padding=0, stride=3, activation=activation),
            conv_block(16, 32, kernel_size=3, padding=0, stride=3, activation=activation),
            conv_block(32, 64, kernel_size=3, padding=0, stride=3, activation=activation),
            nn.Flatten(),
        )
        self.linear_block = nn.Sequential(
            nn.Linear(64, 16),
            # nn.Linear(int(64 * height * width / (4 ** 3)), 16),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )
        self.out = nn.Sequential(
            # nn.Linear(int(32 * height * width / 4 / 4), 5),
            nn.Linear(16, n_class),
            # nn.Sigmoid(),
        )

    def forward(self, image):
        image = self.conv_blocks(image)
        image = self.linear_block(image)

        return self.out(image)
