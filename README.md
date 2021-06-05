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

- [Binary classification model](models/baseline_binary_sdobenchmark/report.md)
- [Multiclass classification model](models/baseline_multiclass_sdobenchmark/report.md)
- [Regression model](models/baseline_regression_sdobenchmark/report.md)
- [Binary classification model on full-disc images](models/baseline_regression_sdodataset/report.md)
- [Self-supervised learning pre-trained model](models/ssl_bz/report.md)
- [Self-supervised learning funetuned model on binary classification](models/ssl_bz_ft_sdobenchmark/report.md)
