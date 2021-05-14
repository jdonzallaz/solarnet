import logging
import os
from enum import Enum
from pathlib import Path
from typing import List, Optional

import hydra
import typer
from omegaconf import OmegaConf

from solarnet.tasks.dataset import make_dataset
from solarnet.tasks.download_dataset import download_dataset
from solarnet.tasks.stats_dataset import stats_dataset
from solarnet.tasks.upload import upload_model
from solarnet.utils.log import init_log, set_log_level
from solarnet.utils.yaml import load_yaml

init_log()

from solarnet.tasks.test import test
from solarnet.tasks.train import train

set_log_level(logging.WARNING)
logger = logging.getLogger()
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


def read_config(parameters_overrides: Optional[List[str]]) -> dict:
    # Initialize configuration (using Hydra)
    hydra.initialize(config_path="./../config")
    config = hydra.compose(config_name="config", overrides=parameters_overrides)
    return OmegaConf.to_container(config, resolve=True)


@app.command('train')
def train_command(
    parameters_overrides: Optional[List[str]] = typer.Argument(None),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose: set_log_level(logging.INFO)

    config = read_config(parameters_overrides)
    logger.info(f"Params: {config}")

    train(config)


@app.command('test')
def test_command(
    parameters_overrides: Optional[List[str]] = typer.Argument(None),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose: set_log_level(logging.INFO)

    config = read_config(parameters_overrides)
    logger.info(f"Params: {config}")

    test(config, verbose)


@app.command('download')
def download_command(
    dataset: str = typer.Argument("sdo-benchmark"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose: set_log_level(logging.INFO)

    download_dataset(dataset)


@app.command('dataset')
def dataset_command(
    config_file: Path = typer.Argument(Path('config') / 'dataset.yaml'),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose: set_log_level(logging.INFO)

    params = load_yaml(config_file)

    make_dataset(params)


class Split(str, Enum):
    train = "train"
    val = "val"
    test = "test"


@app.command('data-stats')
def data_stats_command(
    split: Split = Split.train,
    n_bins: int = 100,
    hist_path: Optional[Path] = None,
    transform: bool = typer.Option(False, "--transform", "-t"),
    parameters_overrides: Optional[List[str]] = typer.Argument(None),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose: set_log_level(logging.INFO)

    config = read_config(parameters_overrides)
    logger.info(f"Params: {config}")

    stats_dataset(config, split.value, n_bins, hist_path, transform)


@app.command('upload')
def upload_command(
    path: Path,
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose: set_log_level(logging.INFO)

    upload_model(path)


# Command to add options before the command (-v train ...)
@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    if verbose: set_log_level(logging.INFO)


if __name__ == "__main__":
    app()
