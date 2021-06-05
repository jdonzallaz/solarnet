# Model report
## Training
### Parameters
```yaml
data:
  name: sdo-benchmark
  validation_size: 0.1
  channel: 211
  size: 256
  targets: regression
  time_steps:
  - 3
  path: data/sdo-benchmark
model:
  backbone: simple-cnn
  learning_rate: 0.005
  activation: relu6
  pooling: 2
  dropout: 0.2
  n_hidden: 16
  lr_scheduler: true
trainer:
  epochs: 20
  patience: 12
  batch_size: 128
name: Baseline regression model
training_type: train
tune_lr: false
path: models/baseline_regression_sdobenchmark
seed: 42
tracking: true
system:
  gpus: 1
  workers: 20
```
### Model architecture
```
ImageRegression(
  (backbone): Sequential(
    (0): SimpleCNN(
      (conv_blocks): Sequential(
        (conv_block_0): Sequential(
          (conv): Conv2d(1, 16, kernel_size=(3, 3), stride=(3, 3))
          (batchnorm): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pooling): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (activation): ReLU6()
          (dropout): Dropout2d(p=0.2, inplace=False)
        )
        (conv_block_1): Sequential(
          (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(3, 3))
          (batchnorm): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pooling): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (activation): ReLU6()
          (dropout): Dropout2d(p=0.2, inplace=False)
        )
        (conv_block_2): Sequential(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(3, 3))
          (batchnorm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (pooling): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (activation): ReLU6()
          (dropout): Dropout2d(p=0.2, inplace=False)
        )
      )
    )
    (1): AdaptiveAvgPool2d(output_size=(1, 1))
    (2): Flatten(start_dim=1, end_dim=-1)
  )
  (regressor): Classifier(
    (classifier): Sequential(
      (0): Flatten(start_dim=1, end_dim=-1)
      (1): Dropout(p=0.2, inplace=False)
      (2): Linear(in_features=64, out_features=16, bias=True)
      (3): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (4): ReLU(inplace=True)
      (5): Dropout(p=0.2, inplace=False)
      (6): Linear(in_features=16, out_features=1, bias=True)
    )
  )
  (loss_fn): MSELoss()
  (test_metrics): MetricCollection(
    (MeanAbsoluteError): MeanAbsoluteError()
    (MeanSquaredError): MeanSquaredError()
  )
)
================================================================================
Total parameters: 24609
```
### Loss curve
![Loss curve](train_plots/loss_curve.png 'Loss curve')

### Metadata
```yaml
machine: 'lambda02 | Linux #113-Ubuntu SMP Thu Jul 9 23:41:39 UTC 2020 | 10 cores @ 4120.00Mhz | RAM 126 GB | 2x TITAN RTX'
training_time: 92.65s
model_size: 322kB
early_stopping_epoch: 0
model_checkpoint_step: 708
model_checkpoint_epoch: 12
tracking_id: SOLN-328
dataset:
  training_set_size: 7446
  validation_set_size: 827
  test_set_size: 874
```
## Test
### Metrics
| Path                                                 | mae         | mse     |
|------------------------------------------------------|-------------|---------|
| models/baseline_regression_sdobenchmark/metrics.yaml | 1.54241e-05 | 5.1e-09 |

### Regression line
![Regression line](test_plots/regression_line.png 'Regression line')

### Test samples
![Test samples](test_plots/test_samples.png 'Test samples')

