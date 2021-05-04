import logging
import zipfile
from pathlib import Path

from solarnet.utils.s3 import s3_download_folder
from solarnet.utils.yaml import load_yaml

logger = logging.getLogger(__name__)

datasets = ['sdo-benchmark', 'sdo-benchmark-zip']


def download_dataset(dataset: str):
    if dataset not in datasets:
        raise ValueError("Dataset unknown")

    bucket_name = dataset
    is_zip = dataset.split("-")[-1] == "zip"
    if is_zip:
        dataset = "-".join(dataset.split("-")[:-1])

    if (Path('data') / f"{dataset}.placeholder").exists():
        return

    # Check MinIO config
    minio_config_file = Path('config') / 'minio.yaml'
    if not minio_config_file.exists():
        raise AttributeError(
            'Missing Minio config. Create config/minio.yaml with aws_access_key_id, aws_secret_access_key and endpoint_url keys.')

    config = load_yaml(minio_config_file)
    if 'aws_access_key_id' not in config or 'aws_secret_access_key' not in config or 'endpoint_url' not in config:
        raise AttributeError(
            'Missing Minio config. Create config/minio.yaml with aws_access_key_id, aws_secret_access_key and endpoint_url keys.')

    # Download dataset
    s3_download_folder(bucket_name, local_dir=Path('data') / dataset)

    # Unzip archive
    if is_zip:
        zip_file = Path('data') / dataset / f'{dataset}.zip'
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(Path('data') / dataset)
        zip_file.unlink()  # Delete archive

    # Add dataset placeholder file (mostly to indicate to DVC that the data is present)
    with open(Path('data') / f"{dataset}.placeholder", "w+") as f:
        f.write(f"{dataset} dataset version 1.0\n")
        f.write("This file is only a placeholder file. It is created when the dataset is downloaded and indicates that "
                "the dataset is present.\n")
