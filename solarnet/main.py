import logging
import os
from pathlib import Path

import typer

from solarnet.tasks.download_dataset import download_dataset
from solarnet.utils.log import init_log, set_log_level
init_log()

from solarnet.tasks.test import test
from solarnet.tasks.train import train
from solarnet.utils.yaml import load_yaml

set_log_level(logging.WARNING)
logger = logging.getLogger()
app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"


@app.command('train')
def train_command(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose: set_log_level(logging.INFO)

    params = load_yaml(Path('config') / 'config.yaml')
    logger.info(f"Params: {params}")

    train(params)


@app.command('test')
def test_command(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose: set_log_level(logging.INFO)

    params = load_yaml(Path('config') / 'config.yaml')
    logger.info(f"Params: {params}")

    test(params, verbose)


@app.command('download')
def train_command(
    dataset: str = typer.Argument("sdo-benchmark"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose: set_log_level(logging.INFO)

    download_dataset(dataset)


# Command to add options before the command (-v train ...)
@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")):
    if verbose: set_log_level(logging.INFO)


if __name__ == "__main__":
    app()
