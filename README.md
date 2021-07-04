# SolarNet

> Deep Learning for Solar Physics Prediction

Solar flares release a huge amount of energy and radiation and can affect the Earth in the worst case. Predicting these events is therefore of major importance.
This work aims at applying self-supervised learning (SSL) methods to solar data to learn pattern and structure in the image. This approach permits the use of larger data volumes and overcomes the limitations of supervised learning caused by a low number of labelled samples and class imbalance.
For this task, the SDO-Benchmark is used as a reference. Another curated dataset [Galvez et al.] is processed and refined for SSL and solar flares prediction.
The contributions are summarized as follows: (1) Various conventional deep learning models are trained and show interesting performance. (2) Self-supervised learning is applied to solar images following SimCLR framework and proves to learn a good representation of the data. (3) A dataset is prepared and is now usable for many tasks, including SSL pre-training and flares classification. (4) A library resulting from this work shows exemplary reproducibility ability and permits the use of the pre-trained models.
By combining these findings, we outperform the previous methods applied on the SDO-Benchmark dataset. A classifier fine-tuned using the SSL model obtains a TSS = 0.624 on binary classification (no flare / flare with >=C-class in a 24 hr period). The encoder trained using the SSL method provides a good and useful representation of the input. It could be used as a feature extractor for many downstream tasks with low constraints on the amount of data, time and processing power.

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
