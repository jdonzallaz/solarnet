import os
from pathlib import Path

import boto3
from botocore import UNSIGNED

from solarnet.utils.yaml import load_yaml

BUCKET_MODEL_REGISTRY = "solarnet-models"
BUCKET_DATA_REGISTRY = "solarnet-datasets"
# TODO: Move to config file
S3_PUBLIC_URL = "https://k8s-minio.isc.heia-fr.ch:9003"
WRITE_CONFIG_PATH = Path('config') / 'minio.yaml'


def s3_write_config():
    # Check MinIO config
    minio_config_file = WRITE_CONFIG_PATH
    if not minio_config_file.exists():
        raise AttributeError(
            'Missing Minio config. Create config/minio.yaml with aws_access_key_id, aws_secret_access_key and endpoint_url keys.')

    config = load_yaml(minio_config_file)
    if 'aws_access_key_id' not in config or 'aws_secret_access_key' not in config or 'endpoint_url' not in config:
        raise AttributeError(
            'Missing Minio config. Create config/minio.yaml with aws_access_key_id, aws_secret_access_key and endpoint_url keys.')

    return config


def s3_read_config():
    return {
        "endpoint_url": S3_PUBLIC_URL,
    }


def s3_exists(bucket: str, key: str):
    config = s3_read_config()
    s3 = boto3.resource('s3', **config, config=boto3.session.Config(signature_version=UNSIGNED))

    bucket = s3.Bucket(bucket)
    s3_objects = bucket.objects.filter(Prefix=key, MaxKeys=1)

    return any([w.key == key or w.key.startswith(key + "/") for w in list(s3_objects)])


def s3_upload(local_path: Path, bucket_name: str, s3_path: str, replace_if_exists: bool = False):
    config = s3_write_config()
    s3 = boto3.resource('s3', **config, config=boto3.session.Config(signature_version='s3v4'))

    bucket = s3.Bucket(bucket_name)

    if replace_if_exists or not s3_exists(bucket_name, s3_path):
        bucket.upload_file(str(local_path), s3_path)


def s3_download_folder(bucket_name: str, s3_folder: str = "", local_dir: Path = None):
    """
    Download the contents of a folder directory
    Use like: download_s3_folder('sdo-benchmark', '11386', Path('test-ds'))
    Check if necessary: https://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket

    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """

    config = s3_read_config()

    s3 = boto3.resource('s3', **config, config=boto3.session.Config(signature_version=UNSIGNED))

    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        if not os.path.exists(target):
            bucket.download_file(obj.key, target)
