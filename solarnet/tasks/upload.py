import logging
from pathlib import Path

from solarnet.utils.s3 import BUCKET_MODEL_REGISTRY, s3_upload

logger = logging.getLogger(__name__)


def upload_model(path: Path, name: str):
    """
    Upload a model to the model registry.

    :param path: path to the model folder
    :param name: name of the model, used as folder name in the bucket
    """

    # param: name, path, readme, shape input, output, class/type
    # output: model saved in minio, in folder=name, params yaml, model pt, readme md

    # download_s3_folder(bucket_name, local_dir=Path('data') / dataset, s3_config=config)

    s3_upload(path / "model.pt", BUCKET_MODEL_REGISTRY, name + "/model.pt", replace_if_exists=True)
    s3_upload(path / "model_config.yaml", BUCKET_MODEL_REGISTRY, name + "/model_config.yaml", replace_if_exists=True)
    if (path / "README.md").exists():
        s3_upload(path / "README.md", BUCKET_MODEL_REGISTRY, name + "/README.md", replace_if_exists=True)
