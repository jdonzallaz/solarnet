import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from solarnet.utils.filesystem import clean_filename, rm_rf
from solarnet.utils.s3 import BUCKET_MODEL_REGISTRY, s3_download_folder, s3_exists
from solarnet.utils.yaml import load_yaml

MODEL_FILENAME = "model.pt"
MODEL_CONFIG_FILENAME = "model_config.yaml"


def download_or_cached_or_local_model_path(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    force_download: bool = False
) -> (Path, Optional[Path]):
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if not isinstance(pretrained_model_name_or_path, Path):
        path = Path(pretrained_model_name_or_path)
    else:
        path = pretrained_model_name_or_path

    if path.is_dir():
        path = path / MODEL_FILENAME
        if not path.exists():
            raise AttributeError(f"No model named {MODEL_FILENAME} found in dir {path.parent}.")

    if path.is_file():
        if not path.exists():
            raise AttributeError(f"Path to model {path} does not exist.")

    else:
        # bucket
        pretrained_model_name_or_path = clean_filename(pretrained_model_name_or_path)

        local_dir = Path("../.solarnet") / pretrained_model_name_or_path
        path = local_dir / MODEL_FILENAME
        if local_dir.exists() and path.exists() and not force_download:
            pass
        else:
            # check model exists in registry

            if not s3_exists(BUCKET_MODEL_REGISTRY, pretrained_model_name_or_path):
                raise AttributeError(f"Model {pretrained_model_name_or_path} does not exist in registry.")

            if force_download:
                rm_rf(local_dir)

            local_dir.mkdir(parents=True, exist_ok=True)
            s3_download_folder("solarnet-models", pretrained_model_name_or_path, local_dir)

    if (path.parent / MODEL_CONFIG_FILENAME).exists():
        config_path = path.parent / MODEL_CONFIG_FILENAME
    else:
        config_path = None

    return path, config_path


class BaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        force_download: bool = False,
        **kwargs,
    ) -> nn.Module:
        path, config_path = download_or_cached_or_local_model_path(
            pretrained_model_name_or_path,
            force_download=force_download
        )

        config = {}
        if config_path is not None:
            config = load_yaml(config_path)
        config = {**config, **kwargs}
        print(config)

        # load
        model = cls(**config)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)

        model.eval()

        return model
