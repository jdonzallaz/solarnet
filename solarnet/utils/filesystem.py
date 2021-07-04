import re
import shutil
from pathlib import Path
import zipfile


def rename_file(path: Path, new_name: str, keep_old: bool = False):
    """
    Rename a local file.

    :param path: path to the file
    :param new_name: new name to give to the file
    :param keep_old: whether to keep the old file or remove it
    """

    if path.exists() and path.is_file():
        shutil.copy2(path, Path(path.parent, new_name))

    if not keep_old:
        path.unlink()


def clean_filename(filename: str):
    """
    Remove unsupported symbols from a filename.

    :param filename: The filename to clean
    :return: The cleaned filename
    """

    return re.sub("[/\\\ ^()%:'\"<>|?*]", "", filename)


def rm_rf(path: Path):
    """
    Remove a local file/dir recursively.

    :param path: The path to remove.
    """

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def unzip(zip_file: Path, destination: Path, delete_file: bool = False):
    """
    Unzip a zip file to the destination folder. Optionnaly delete the zip file after the decompression.

    :param zip_file: file to decompress
    :param destination: destination folder
    :param delete_file: whether to delete the zip file after the decompression
    """

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(destination)
    zip_file.unlink()  # Delete archive
