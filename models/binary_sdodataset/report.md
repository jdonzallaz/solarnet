# Model report
## Training
### Parameters
```yaml
data:
  name: sdo-dataset
  channel: 211
  size: 256
  targets:
    classes:
    - Quiet: < 1e-6
    - '>=C': '>= 1e-6'
  path: /data1/data/sdo-dataset
model:
  activation: relu6
trainer:
  epochs: 20
  batch_size: 128
  learning_rate: 0.005
  patience: 12
name: Binary classification model on SDO-Dataset
path: models/binary_sdodataset
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
        (0): Conv2d(4, 16, kernel_size=(3, 3), stride=(3, 3))
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
      (0): Linear(in_features=16, out_features=2, bias=True)
    )
  )
  (loss_fn): CrossEntropyLoss()
  (test_metrics): MetricCollection(
    (Accuracy): Accuracy()
    (F1): F1()
    (Recall): Recall()
    (StatScores): StatScores()
  )
)
================================================================================
Total parameters: 25026
```
### Loss curve
![Loss curve](history.png 'Loss curve')

### Metadata
```yaml
machine: 'lambda02 | Linux #113-Ubuntu SMP Thu Jul 9 23:41:39 UTC 2020 | 10 cores @ 4120.00Mhz | RAM 126 GB | 2x TITAN RTX'
training_time: 315.16s
model_size: 326kB
early_stopping_epoch: 13
model_checkpoint_step: 262
model_checkpoint_epoch: 7
tracking_id: SOLN-104
dataset:
  training_set_size: 4402
  validation_set_size: 672
  test_set_size: 646
```
## Test
### Metrics
| Path                                  | accuracy   | balanced_accuracy   | csi    | f1      | far     | hss    | pod     | tss     |
|---------------------------------------|------------|---------------------|--------|---------|---------|--------|---------|---------|
| models/binary_sdodataset/metrics.yaml | 0.66873    | 0.61404             | 0.6304 | 0.57918 | 0.14319 | 0.1795 | 0.70463 | 0.22807 |

### Confusion matrix
![Confusion matrix](confusion_matrix.png 'Confusion matrix')

### Test samples
![Test samples](test_samples.png 'Test samples')

