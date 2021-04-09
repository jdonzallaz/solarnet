import abc
import logging
from pathlib import Path
from typing import Optional, Union

import neptune
import neptune.new as neptune_new
from pytorch_lightning.loggers import NeptuneLogger

from solarnet.utils.yaml import load_yaml

logger = logging.getLogger(__name__)


class Tracking:
    @abc.abstractmethod
    def __init__(self, parameters: dict = {}, tags: list = [], disabled: bool = False, extras: dict = None):
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement __init__()")

    @abc.abstractmethod
    def get_callback(self, integration_name: str) -> any:
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement get_callback()")

    @abc.abstractmethod
    def get_id(self) -> Optional[Union[str, int]]:
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement get_id()")

    @abc.abstractmethod
    def log_metrics(self, metrics: dict, prefix: str = ''):
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement log_metrics()")

    @abc.abstractmethod
    def log_metric(self, name: str, value: float):
        raise NotImplementedError(f"Class {self.__class__.__name__} doesn't implement log_metric()")

    @abc.abstractmethod
    def log_artifact(self, path: Path, name: str = None):
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

    def get_callback(self, integration_name: str) -> any:
        if not self.disabled:
            if integration_name == 'pytorch-lightning':
                return NeptuneLogger(
                    api_key=self.api_token,
                    project_name=self.project_name,
                    close_after_fit=False,
                    experiment_id=self.exp.id)
        return None

    def get_id(self) -> Optional[Union[str, int]]:
        if not self.disabled:
            return self.exp.id

    def log_metrics(self, metrics: dict, prefix: str = ''):
        if not self.disabled:
            for metric in metrics:
                neptune.log_metric(f'{prefix}{metric}', metrics[metric])

    def log_metric(self, name: str, value: float):
        if not self.disabled:
            neptune.log_metric(name, value)

    def log_artifact(self, path: Path, name: str = None):
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


class NeptuneNewTracking(Tracking):
    def __init__(self, parameters: dict = None, tags: list = None, disabled: bool = False, extras: dict = None):
        if parameters is None:
            parameters = {}
        if tags is None:
            tags = []
        self.disabled = disabled
        if extras is None:
            extras = {}

        if not disabled:
            neptune_config_file = Path('config') / 'neptune.yaml'
            if not neptune_config_file.exists():
                raise AttributeError('Missing Neptune config. Create config/neptune.yaml with token and project keys.')

            config = load_yaml(neptune_config_file)
            if 'project' not in config or 'token' not in config:
                raise AttributeError('Missing Neptune config. Create config/neptune.yaml with token and project keys.')

            self.project_name = config['project']
            self.api_token = config['token']

            logger.info("Experiment tracked with Neptune:")

            if "run_id" in extras:

                # Resume logging (do not add name and tags)
                self.run = neptune_new.init(
                    project=self.project_name,
                    api_token=self.api_token, run=extras.pop("run_id"),
                    **extras,
                )
            else:
                self.run = neptune_new.init(
                    project=self.project_name,
                    api_token=self.api_token,
                    name=parameters.get('name'),
                    tags=tags,
                    **extras,
                )

            self.run['parameters'] = parameters
        else:
            logger.info("Tracking is disabled.")

    @classmethod
    def resume(cls, run_id: str):
        return cls(extras={"run_id": run_id})

    def get_callback(self, integration_name: str) -> any:
        if not self.disabled:
            if integration_name == 'pytorch-lightning':
                return NeptuneLogger(
                    api_key=self.api_token,
                    project_name=self.project_name,
                    close_after_fit=False,
                    experiment_id=self.run['sys/id'].fetch(),
                )
        return None

    def get_id(self) -> Optional[Union[str, int]]:
        if not self.disabled:
            return self.run['sys/id'].fetch()

    def log_metrics(self, metrics: dict, prefix: str = ''):
        if not self.disabled:
            self.run[prefix] = metrics

    def log_metric(self, name: str, value: float):
        if not self.disabled:
            self.run[name] = value

    def log_artifact(self, path: Path, name: str = None):
        if not self.disabled:
            if name is None:
                name = f'artifacts/{path.stem}'
            self.run[name].upload(str(path))

    def log_property(self, name: str, value: Union[str, int, float]):
        if not self.disabled:
            self.run[name] = value

    def log_text(self, name: str, value: str):
        if not self.disabled:
            self.run[name] = value

    def end(self):
        if not self.disabled:
            self.run.stop()
