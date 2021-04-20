# Model report
## Training
### Parameters
```yaml
data:
  validation_size: 0.1
  channel: 211
  size: 256
  targets: regression
  time_steps:
  - 3
model:
  activation: relu6
trainer:
  epochs: 20
  batch_size: 128
  learning_rate: 0.005
  patience: 12
name: Baseline regression model
path: models/regression
seed: 42
tracking: true
gpus: 1
```
### Model architecture
```
CNNRegression(
  (cnn): CNNModule(
    (conv_blocks): Sequential(
      (0): Sequential(
        (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(3, 3))
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
        (3): ReLU6()
        (4): Dropout2d(p=0.1, inplace=False)
      )
      (1): Sequential(
        (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(3, 3))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
        (3): ReLU6()
        (4): Dropout2d(p=0.1, inplace=False)
      )
      (2): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(3, 3))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
        (3): ReLU6()
        (4): Dropout2d(p=0.1, inplace=False)
      )
      (3): Flatten(start_dim=1, end_dim=-1)
    )
    (linear_block): Sequential(
      (0): Linear(in_features=64, out_features=16, bias=True)
      (1): ReLU()
      (2): Dropout2d(p=0.2, inplace=False)
    )
    (out): Sequential(
      (0): Linear(in_features=16, out_features=1, bias=True)
    )
  )
  (loss_fn): MSELoss()
  (test_metrics): MetricCollection(
    (MeanAbsoluteError): MeanAbsoluteError()
    (MeanSquaredError): MeanSquaredError()
  )
)
================================================================================
Total parameters: 24577
```
### Loss curve
![Loss curve](history.png 'Loss curve')

## Test
### Metrics
| Path                           | mae   | mse   |
|--------------------------------|-------|-------|
| models\regression\metrics.yaml | 1e-05 | 0.0   |

### Regression line
![Regression line](regression_line.png 'Regression line')

### Test samples
![Test samples](test_samples.png 'Test samples')
