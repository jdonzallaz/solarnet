import logging

import torch
from torch import nn

from solarnet.models.backbone import get_backbone

logger = logging.getLogger(__name__)


class CRNN(nn.Module):
    """
    Convolutional-Recurrent NN. Use a (single) CNN encoder for each input in
    the sequence, then use a GRU on each encoded portion of the input.
    The different channels of the input are used as separate sequence.
    A (32, 4, 128, 128) corresponds to an input of 32 (batch-size) sequences
    of 4 images of sizes (1, 128, 128).
    The encoder (backbone) can be chosen from: simple-cnn, resnets,
    squeezenets (see solarnet.models.backbone).
    kwargs are passed to the encoder/backbone.
    You may want to add a classification head, this is just the backbone.
      See solarnet.models.classifier.Classifier.
      The size of output is available in output_size property.

    :param channels: size of the sequences
    :param backbone: type of the backbone
    :param n_layer_rnn: number of GRU layers
    :param n_cell_rnn: number of cell per GRU layer
    """

    def __init__(
        self,
        channels: int,
        backbone: str = "simple-cnn",
        n_layer_rnn: int = 1,
        n_cell_rnn: int = 8,
        **kwargs,
    ):
        super().__init__()

        self.channels = channels

        self.cnn, output_size = get_backbone(backbone, channels=1, **kwargs)

        self.rnn = nn.GRU(input_size=output_size, hidden_size=n_cell_rnn, num_layers=n_layer_rnn, batch_first=True)

        self._output_size = n_cell_rnn

    def forward(self, image):
        imgs = torch.split(image, 1, 1)

        img = []
        for i in range(self.channels):
            img.append(self.cnn(imgs[i]))
        img = torch.stack(img, 1)

        img, _ = self.rnn(img)
        img = img[:, -1, :]

        return img

    @property
    def output_size(self):
        return self._output_size
