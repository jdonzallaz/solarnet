# Model report
## Training
### Parameters
```yaml
data:
  name: sdo-dataset
  channel: bz
  size: 256
  targets:
    classes:
    - Quiet: < 1e-6
    - '>=C': '>= 1e-6'
  path: data/sdo-dataset-cls-bz-24h-2015-2017
model:
  backbone: simple-cnn
  learning_rate: 0.001
  activation: leakyrelu
  pooling: 2
  dropout: 0.2
  n_hidden: 16
  lr_scheduler: true
trainer:
  epochs: 20
  patience: 12
  batch_size: 128
name: Binary classification model on SDO-Dataset, Bz, split val on month, test on 2015-2017
training_type: train
tune_lr: false
path: models/baseline_binary_sdodataset
seed: 42
tracking: true
system:
  gpus: 1
  workers: 20
tags:
- sdo-dataset
```
### Model architecture
```
ImageClassification(
  (backbone): Sequential(
    (0): SimpleCNN(
      (conv_blocks): Sequential(
        (conv_block_0): Sequential(
          (conv): Conv2d(1, 16, kernel_size=(3, 3), stride=(3, 3))
          (batchnorm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pooling): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (activation): LeakyReLU(negative_slope=0.01)
          (dropout): Dropout2d(p=0.2, inplace=False)
        )
        (conv_block_1): Sequential(
          (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(3, 3))
          (batchnorm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pooling): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (activation): LeakyReLU(negative_slope=0.01)
          (dropout): Dropout2d(p=0.2, inplace=False)
        )
        (conv_block_2): Sequential(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(3, 3))
          (batchnorm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pooling): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (activation): LeakyReLU(negative_slope=0.01)
          (dropout): Dropout2d(p=0.2, inplace=False)
        )
      )
    )
    (1): AdaptiveAvgPool2d(output_size=(1, 1))
    (2): Flatten(start_dim=1, end_dim=-1)
  )
  (classifier): Classifier(
    (classifier): Sequential(
      (0): Flatten(start_dim=1, end_dim=-1)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=64, out_features=16, bias=True)
      (3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (4): ReLU(inplace=True)
      (5): Dropout(p=0.2, inplace=False)
      (6): Linear(in_features=16, out_features=2, bias=True)
    )
  )
  (loss_fn): CrossEntropyLoss()
  (train_accuracy): Accuracy()
  (val_accuracy): Accuracy()
  (test_metrics): MetricCollection(
    (Accuracy): Accuracy()
    (F1): F1()
    (Recall): Recall()
    (StatScores): StatScores()
  )
)
================================================================================
Total parameters: 24626
```
### Loss curve
![Loss curve](train_plots/loss_curve.png 'Loss curve')

### Accuracy curve
![Accuracy curve](train_plots/accuracy_curve.png 'Accuracy curve')

### Metadata
```yaml
machine: 'lambda02 | Linux #113-Ubuntu SMP Thu Jul 9 23:41:39 UTC 2020 | 10 cores @ 4120.00Mhz | RAM 126 GB | 2x TITAN RTX'
training_time: 102.80s
model_size: 323kB
early_stopping_epoch: 0
model_checkpoint_step: 192
model_checkpoint_epoch: 16
tracking_id: SOLN-386
data:
  class-balance:
    train:
      Quiet: 473
      '>=C': 1043
    val:
      Quiet: 41
      '>=C': 112
    test:
      Quiet: 647
      '>=C': 441
  shape: (1, 256, 448)
  tensor-data:
    min: -1.0
    max: 1.0
    mean: -0.00022344599710777402
    std: 0.046560730785131454
  set-sizes:
    train: 1516
    val: 153
    test: 1088
```
## Test
### Metrics
| Path                                           | accuracy   | balanced_accuracy   | csi    | f1     | far    | hss    | pod    | tss    |
|------------------------------------------------|------------|---------------------|--------|--------|--------|--------|--------|--------|
| models/baseline_binary_sdodataset/metrics.yaml | 0.7022     | 0.6431              | 0.3106 | 0.6332 | 0.1657 | 0.3166 | 0.3311 | 0.2862 |

### Confusion matrix
![Confusion matrix](test_plots/confusion_matrix.png 'Confusion matrix')

### ROC Curve
![ROC Curve](test_plots/roc_curve.png 'ROC Curve')

### Test samples
![Test samples](test_plots/test_samples.png 'Test samples')

