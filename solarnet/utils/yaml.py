from pathlib import Path

from ruamel.yaml import YAML


def load_yaml(file: Path) -> dict:
    """
    Load dict from yaml file (yaml version 1.2)

    :param file: path to yaml file
    :return: dict of data
    """

    yaml = YAML(typ="safe")
    data = yaml.load(file)

    return data


def write_yaml(file: Path, data: dict):
    """
    Write dict to yaml file (yaml version 1.2)

    :param file: path to yaml file to create
    :param data: dict of data to write in the file
    """

    yaml = YAML()
    yaml.width = 4096
    yaml.dump(data, file)
