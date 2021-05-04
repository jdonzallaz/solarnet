import os
import optuna
import subprocess

import typer

from solarnet.utils.yaml import load_yaml, write_yaml
from pathlib import Path

os.environ["MKL_THREADING_LAYER"] = "GNU"


def objective_builder(model: str, metric: str = "f1"):
    def objective(trial):
        # Load parameters
        config_path = Path("config") / "config.yaml"
        parameters = load_yaml(config_path)

        # Suggest parameters
        parameters["trainer"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 6e-3, log=True
        )
        parameters["trainer"]["epochs"] = trial.suggest_int("epochs", 5, 30, 5)
        parameters["trainer"]["batch_size"] = trial.suggest_int("batch_size", 32, 256, 32)
        parameters["data"]["channel"] = trial.suggest_categorical(
            "channel",
            [
                "94",
                "131",
                "171",
                "193",
                "211",
                "304",
                "335",
                "1700",
                "continuum",
                "magnetogram",
            ],
        )
        # parameters["data"]["size"] = trial.suggest_int("size", 128, 256)
        parameters["model"]["activation"] = trial.suggest_categorical(
            "activation", ["relu", "selu", "relu6", "tanh", "prelu", "leakyrelu"]
        )

        # Write parameters
        write_yaml(config_path, parameters)

        # Run pipeline
        process = subprocess.run(["dvc", "repro", "--glob", f"*@{model}", "-q"])

        # Check metric
        metrics = load_yaml(Path("models") / model / "metrics.yaml")
        return metrics[metric]

    return objective


def run(model: str, metric: str = "f1", direction: str = "max", n_trials: int = 100):
    if direction == "max":
        direction = optuna.study.StudyDirection.MAXIMIZE
    else:
        direction = optuna.study.StudyDirection.MINIMIZE

    study = optuna.create_study(direction=direction)
    study.optimize(objective_builder(model, metric), n_trials=n_trials)

    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    if optuna.visualization.is_available():
        path = Path("models") / model
        optuna.visualization.plot_slice(study) \
            .write_image(str(path / "optuna_slice.png"))
        optuna.visualization.plot_optimization_history(study) \
            .write_image(str(path / "optuna_history.png"))
        optuna.visualization.plot_parallel_coordinate(study) \
            .write_image(str(path / "optuna_coordinate.png"))
        optuna.visualization.plot_param_importances(study) \
            .write_image(str(path / "optuna_importance.png"))


if __name__ == "__main__":
    typer.run(run)
