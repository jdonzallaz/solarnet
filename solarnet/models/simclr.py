from pl_bolts.models.self_supervised import SimCLR as SimCLR_bolts
from torch import nn

from solarnet.models.model_utils import BaseModel


class SimCLR(SimCLR_bolts, BaseModel):
    """
    This class is wrapper around SimCLR class from lightning-bolts.
    It adds a single parameter "n_channel" and overrides the first layer of the resnet encoder with the given number
      of channel (default: 3).
    The dataset parameter is now optional (default to ""). Other parameters (gpus, num_samples, batch_size) receive meaningful default.
    The arch parameter is renamed to "backbone".
    The "encoder" module is renamed to "backbone".
    """

    def __init__(
        self,
        gpus: int = 0,
        num_samples: int = 1,
        batch_size: int = 512,
        dataset: str = "",
        num_nodes: int = 1,
        backbone: str = "resnet50",
        hidden_mlp: int = 2048,
        feat_dim: int = 128,
        warmup_epochs: int = 10,
        max_epochs: int = 100,
        temperature: float = 0.1,
        first_conv: bool = True,
        maxpool1: bool = True,
        optimizer: str = "adam",
        exclude_bn_bias: bool = False,
        start_lr: float = 0.0,
        learning_rate: float = 1e-3,
        final_lr: float = 0.0,
        weight_decay: float = 1e-6,
        n_channel: int = 3,
        **kwargs,
    ):
        kwargs.pop('arch', None)
        super().__init__(
            gpus,
            num_samples,
            batch_size,
            dataset,
            num_nodes,
            backbone,
            hidden_mlp,
            feat_dim,
            warmup_epochs,
            max_epochs,
            temperature,
            first_conv,
            maxpool1,
            optimizer,
            exclude_bn_bias,
            start_lr,
            learning_rate,
            final_lr,
            weight_decay,
            n_channel=n_channel,
            backbone=backbone,  # Doubled here so it's added to hparams
            **kwargs,
        )

        self.n_channel = n_channel

        if self.n_channel != 3:
            # Other values are the default found in resnet models.
            self.encoder.conv1 = nn.Conv2d(
                self.n_channel,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )

        self.backbone = self.encoder
        del self.encoder

    def forward(self, x):
        # bolts resnet returns a list
        return self.backbone(x)[-1]

    @property
    def backbone_name(self) -> str:
        return self.arch

    @property
    def output_size(self) -> int:
        return self.hidden_mlp
