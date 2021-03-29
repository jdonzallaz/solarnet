import abc
import logging
from pathlib import Path
from typing import Union

import neptune
from pytorch_lightning.loggers import NeptuneLogger

from solarnet.utils.yaml import load_yaml

logger = logging.getLogger(__name__)


class Tracking:
    @abc.abstractmethod
    def __init__(self, parameters: dict = {}, tags: list = [], disabled: bool = False, extras: dict = None):
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement __init__()")

    @abc.abstractmethod
    def get_keras_callback(self) -> any:
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement get_keras_callback()")

    @abc.abstractmethod
    def get_pl_logger(self) -> any:
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement get_pl_logger()")

    @abc.abstractmethod
    def log_metrics(self, metrics: dict, prefix: str = ''):
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement log_metrics()")

    @abc.abstractmethod
    def log_metric(self, name: str, value: float):
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement log_metric()")

    @abc.abstractmethod
    def log_artifact(self, path: Path):
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement log_artifact()")

    @abc.abstractmethod
    def log_property(self, name: str, value: Union[str, int, float]):
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement log_property()")

    @abc.abstractmethod
    def log_text(self, name: str, value: str):
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement log_text()")

    @abc.abstractmethod
    def end(self):
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement end()")


class NeptuneTracking(Tracking):
    def __init__(self, parameters: dict = None, tags: list = None, disabled: bool = False, extras: dict = None):
        if parameters is None:
            parameters = {}
        if tags is None:
            tags = []
        self.disabled = disabled
        if extras is None:
            extras = {}

        if not disabled:
            self.name = parameters['name'] if 'name' in parameters else 'Unnamed'

            neptune_config_file = Path('config') / 'neptune.yaml'
            if not neptune_config_file.exists():
                raise AttributeError('Missing Neptune config. Create config/neptune.yaml with token and project keys.')

            config = load_yaml(neptune_config_file)
            if 'project' not in config or 'token' not in config:
                raise AttributeError('Missing Neptune config. Create config/neptune.yaml with token and project keys.')

            self.project_name = config['project']
            self.api_token = config['token']

            logger.info("Experiment tracked with Neptune:")
            neptune.init(project_qualified_name=self.project_name, api_token=self.api_token)

            self.exp = neptune.create_experiment(name=self.name, tags=tags, params=parameters, **extras)
        else:
            logger.info("Tracking is disabled.")

    def get_pl_logger(self) -> any:
        if not self.disabled:
            return NeptuneLogger(
                api_key=self.api_token,
                project_name=self.project_name,
                close_after_fit=False,
                experiment_id=self.exp.id)
        return None

    def log_metrics(self, metrics: dict, prefix: str = ''):
        if not self.disabled:
            for metric in metrics:
                neptune.log_metric(f'{prefix}{metric}', metrics[metric])

    def log_metric(self, name: str, value: float):
        if not self.disabled:
            neptune.log_metric(name, value)

    def log_artifact(self, path: Path):
        if not self.disabled:
            path = str(path)
            neptune.log_artifact(path)

    def log_property(self, name: str, value: Union[str, int, float]):
        if not self.disabled:
            neptune.set_property(name, str(value))

    def log_text(self, name: str, value: str):
        if not self.disabled:
            neptune.log_text(name, value)

    def end(self):
        if not self.disabled:
            neptune.stop()
