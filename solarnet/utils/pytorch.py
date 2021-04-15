from pathlib import Path

from torch import nn


def pytorch_model_summary(model: nn.Module, path: Path, filename: str = 'model_summary.txt'):
    model_repr = repr(model)
    nb_parameters = sum([param.nelement() for param in model.parameters()])

    with open(path / filename, "w") as text_file:
        print(model_repr, file=text_file)
        print("=" * 80, file=text_file)
        print(f"Total parameters: {nb_parameters}", file=text_file)
