# SolarNet

> Deep Learning for Solar Physics Prediction


## Quickstart

### Install environment

Using pip:
```sh
python -m venv venv

source venv/bin/activate  # On Unix
.\env\Scripts\activate  # On Windows

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
