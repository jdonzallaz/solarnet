import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import nn

from solarnet.utils.filesystem import clean_filename, rm_rf
from solarnet.utils.pytorch import print_incompatible_keys as print_incompatible_keys_fn
from solarnet.utils.s3 import BUCKET_MODEL_REGISTRY, s3_download_folder, s3_exists
from solarnet.utils.yaml import load_yaml

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.pt"
MODEL_CONFIG_FILENAME = "model_config.yaml"


def download_or_cached_or_local_model_path(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    force_download: bool = False,
) -> Tuple[Path, Optional[Path]]:
    """
    Retrieve a pretrained model. Depending on the given format (string or path), it either download or find it in the
    filesystem.
    If the `pretrained_model_name_or_path` is a string (the name/id of the model), it will check if it is present in the
    cache (in user folder ~/.solarnet/models). If not in cache, it will be downloaded from the MinIO models bucket.
    If the `pretrained_model_name_or_path` is a path, it will check for the model file and verify it exists.
    This return the model path and the model config path.

    :param pretrained_model_name_or_path: the name/id of the model for download, or a path where the model exists in the
                                          the local filesystem.
    :param force_download: if true, do not search in the cache and download the model.
    :return: a tuple with the model file (model.pt) and the config file (model_config.yaml)
    """

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

        local_dir = Path.home() / ".solarnet" / "models" / pretrained_model_name_or_path
        path = local_dir / MODEL_FILENAME
        if local_dir.exists() and path.exists() and not force_download:
            logger.info(f"Model {pretrained_model_name_or_path} found in cache")
        else:
            # check model exists in registry

            if not s3_exists(BUCKET_MODEL_REGISTRY, pretrained_model_name_or_path):
                raise AttributeError(f"Model {pretrained_model_name_or_path} does not exist in registry.")

            if force_download:
                rm_rf(local_dir)

            local_dir.mkdir(parents=True, exist_ok=True)
            s3_download_folder("solarnet-models", pretrained_model_name_or_path, local_dir)

            logger.info(f"Model {pretrained_model_name_or_path} downloaded from model registry")

    if (path.parent / MODEL_CONFIG_FILENAME).exists():
        config_path = path.parent / MODEL_CONFIG_FILENAME
    else:
        config_path = None

    return path, config_path


class BaseModel(pl.LightningModule):
    """
    Base class for the LightningModule models. It gives access to from_pretrained and load_pretrained methods.
    """

    backbone_name = "undefined"
    output_size = -1

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        force_download: bool = False,
        strict: bool = False,
        print_incompatible_keys: bool = False,
        **kwargs,
    ) -> pl.LightningModule:
        """
        Class method which will build a model, and download and load the weights of the given pre-trained model.
        If the `pretrained_model_name_or_path` is a string (the name/id of the model), it will check if it is present in
        the cache (in user folder ~/.solarnet/models). If not in cache, it will be downloaded from the MinIO models bucket.
        If the `pretrained_model_name_or_path` is a path, it will use the model.pt from this folder.
        kwargs are used to override model config.
        Use this method to create a model and load weights trained with the same model architecture.

        :param pretrained_model_name_or_path: the name/id of the model for download, or a path where the model exists in the
                                              the local filesystem.
        :param force_download: if true, do not search in the cache and download the model.
        :param strict: if true, an error will be raised if the architecture is not the same and weights fail to be loaded.
        :param kwargs: used to override model config.
        :return: a pl.LightningModule instance
        """

        path, config_path = download_or_cached_or_local_model_path(
            pretrained_model_name_or_path, force_download=force_download
        )

        # Config
        config = {}
        backbone = None
        if config_path is not None:
            config = load_yaml(config_path)
            backbone = config.get("backbone", None)
        logger.info(f"Model {pretrained_model_name_or_path} loaded with config:")
        logger.info(config)
        hparams = config.pop("hparams", {})
        hparams = {**hparams, **kwargs}

        # load
        model = cls(**hparams)
        if backbone is not None and model.backbone_name != "undefined" and backbone != model.backbone_name:
            raise AttributeError(f"Model {pretrained_model_name_or_path} is not compatible with class {cls}.")
        state_dict = torch.load(path)
        incompatible_keys = model.load_state_dict(state_dict, strict=strict)
        if print_incompatible_keys:
            print_incompatible_keys_fn(incompatible_keys)

        model.eval()

        return model

    def load_pretrained(
        self,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        force_download: bool = False,
        strict: bool = False,
        print_incompatible_keys: bool = False,
        fix_dict_keys: bool = True,
    ):
        """
        Instance method to (optionnaly) download and load the weights of the given pre-trained model.
        If the `pretrained_model_name_or_path` is a string (the name/id of the model), it will check if it is present in
        the cache (in user folder ~/.solarnet/models). If not in cache, it will be downloaded from the MinIO models bucket.
        If the `pretrained_model_name_or_path` is a path, it will use the model.pt from this folder.
        Use this method to load (some) weights from a pre-trained model while customizing the final architecture. For
          example to finetune on a downstream task. Use strict=False if the model architecture is not exactly the same.
          Otherwise, an error will be raised.

        :param pretrained_model_name_or_path: the name/id of the model for download, or a path where the model exists in the
                                              the local filesystem.
        :param force_download: if true, do not search in the cache and download the model.
        :param strict: if true, an error will be raised if the architecture is not the same and weights fail to be loaded.
        :param fix_dict_keys: if true, some known state keys will be renamed (ex: encoder to backbone).
        """

        path, config_path = download_or_cached_or_local_model_path(
            pretrained_model_name_or_path, force_download=force_download
        )

        # Config
        config = {}
        backbone = None
        if config_path is not None:
            config = load_yaml(config_path)
            backbone = config.get("backbone", None)
        logger.info(f"Model {pretrained_model_name_or_path} loaded with config:")
        logger.info(config)

        if backbone is not None and self.backbone_name != "undefined" and backbone != self.backbone_name:
            raise RuntimeError("The backbone of the pretrained model is different.")

        state_dict = torch.load(path)

        # Fix some know dict keys
        # encoder -> backbone
        if fix_dict_keys:
            state_dict = {k.replace("encoder.", "backbone."): v for k, v in state_dict.items()}

        incompatible_keys = self.load_state_dict(state_dict, strict=strict)
        if print_incompatible_keys:
            print_incompatible_keys_fn(incompatible_keys)

        self.eval()
