import argparse
from typing import Any, Dict, Optional, Union

from pytorch_lightning.loggers import LightningLoggerBase


class InMemoryLogger(LightningLoggerBase):
    """
    A simple logger to save metrics in memory. Metrics are the accessible using `logger.metrics`.
    """

    def __init__(self):
        super().__init__()

        self._metrics = {}
        self._hyperparams = {}

    @property
    def experiment(self) -> Any:
        return None

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append({"value": value, "step": step})

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        self._hyperparams = {**self._hyperparams, **dict(params)}

    @property
    def name(self) -> str:
        return ""

    @property
    def version(self) -> Union[int, str]:
        return 0

    @property
    def metrics(self):
        return self._metrics

    @property
    def hyperparams(self):
        return self._hyperparams
