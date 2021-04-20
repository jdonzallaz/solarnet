# SolarNet

> Deep Learning for Solar Physics Prediction

## Quickstart

### Install environment

Using pip:

```sh
python -m venv venv

source venv/bin/activate  # On Unix
.\venv\Scripts\activate  # On Windows

pip install -r requirements.txt

deactivate  # When the job is done
```

Using conda:

```sh
conda env create --file environment.yaml

conda activate solarnet-conda-env

conda deactivate  # When the job is done
```

### Reproduce experiments

```sh
dvc repro
```

## Results

Up-to-date results are available in `models/` folder.

- [Binary classification model](models/binary/report.md)
- [Multiclass classification model](models/multiclass/report.md)
- [Regression model](models/regression/report.md)
