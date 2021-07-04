import logging
import zipfile
from pathlib import Path

from solarnet.utils.s3 import s3_download_file, BUCKET_DATA_REGISTRY
from solarnet.utils.yaml import load_yaml
from solarnet.utils.filesystem import unzip

logger = logging.getLogger(__name__)

datasets = ["sdo-benchmark"]


def download_dataset(dataset: str, destination: Path):
    if dataset not in datasets:
        raise ValueError("Dataset unknown")
    bucket_name = BUCKET_DATA_REGISTRY

    logger.info(f"Downloading {dataset} from {bucket_name} to {destination} ...")
    s3_download_file(bucket_name, s3_file=f"{dataset}.zip", local_dir=destination)

    logger.info(f"Unzipping archive...")
    unzip(destination / f"{dataset}.zip", destination / dataset, delete_file=True)
