# Model report
## Training
### Parameters
```yaml
data:
  name: sdo-benchmark
  validation_size: 0.1
  channel: 211
  size: 256
  targets:
    classes:
    - Quiet: < 1e-6
    - C: < 1e-5
    - M: < 1e-4
    - X: '>= 1e-4'
  time_steps:
  - 3
  path: data/sdo-benchmark
model:
  activation: relu6
trainer:
  epochs: 20
  batch_size: 128
  learning_rate: 0.005
  patience: 12
name: Baseline multiclass classification model
path: models/multiclass
seed: 42
tracking: true
gpus: 1
```
### Model architecture
```
CNNClassification(
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
      (0): Linear(in_features=16, out_features=4, bias=True)
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
Total parameters: 24628
```
### Loss curve
![Loss curve](train_plots/loss_curve.png 'Loss curve')

### Accuracy curve
![Accuracy curve](train_plots/accuracy_curve.png 'Accuracy curve')

### Metadata
```yaml
machine: eifrpoeisc45 | Windows 10.0.19041 | 4 cores @ 2592.00Mhz | RAM 16 GB | 1x Quadro M2000M
training_time: 233.99s
model_size: 322kB
early_stopping_epoch: 11
model_checkpoint_step: 353
model_checkpoint_epoch: 5
tracking_id: SOLN-125
dataset:
  training_set_size: 7446
  validation_set_size: 827
  test_set_size: 874
```
## Test
### Metrics
| Path                           | accuracy   | balanced_accuracy   | csi     | f1      | far     | hss    | pod     | tss    |
|--------------------------------|------------|---------------------|---------|---------|---------|--------|---------|--------|
| models\multiclass\metrics.yaml | 0.42677    | 0.45713             | 0.27127 | 0.35547 | 0.57323 | 0.2357 | 0.42677 | 0.2357 |

### Confusion matrix
![Confusion matrix](test_plots/confusion_matrix.png 'Confusion matrix')

### Test samples
![Test samples](test_plots/test_samples.png 'Test samples')

