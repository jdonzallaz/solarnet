import os
from pathlib import Path

import boto3
from tqdm import tqdm


def download_s3_folder(bucket_name: str, s3_folder: str = "", local_dir: Path = None, s3_config=None):
    """
    Download the contents of a folder directory
    Use like: download_s3_folder('sdo-benchmark', '11386', Path('test-ds'))
    Check if necessary: https://stackoverflow.com/questions/31918960/boto3-to-download-all-files-from-a-s3-bucket

    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
        s3_config: a dict with config passed to boto3 resource (aws_access_key_id, aws_secret_access_key, endpoint_url)
    """

    if s3_config is None:
        s3_config = {}

    s3 = boto3.resource('s3', **s3_config, config=boto3.session.Config(signature_version='s3v4'), )

    bucket = s3.Bucket(bucket_name)
    for obj in tqdm(bucket.objects.filter(Prefix=s3_folder)):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        if not os.path.exists(target):
            bucket.download_file(obj.key, target)
