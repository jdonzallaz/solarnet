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
    - '>=C': '>= 1e-6'
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
name: Baseline binary classification model
training_type: train
tune_lr: false
path: models/baseline_binary_sdobenchmark
seed: 42
tracking: true
system:
  gpus: 1
  workers: 20
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
training_time: 85.12s
model_size: 323kB
early_stopping_epoch: 0
model_checkpoint_step: 590
model_checkpoint_epoch: 10
tracking_id: SOLN-388
data:
  class-balance:
    train:
      Quiet: 4420
      '>=C': 3026
    val:
      Quiet: 474
      '>=C': 353
    test:
      Quiet: 352
      '>=C': 522
  shape: (1, 256, 256)
  tensor-data:
    min: -0.7960784435272217
    max: 0.8901960849761963
    mean: -0.10380689054727554
    std: 0.28518185019493103
  set-sizes:
    train: 7446
    val: 827
    test: 874
```
## Test
### Metrics
| Path                                             | accuracy   | balanced_accuracy   | csi   | f1     | far    | hss    | pod    | tss    |
|--------------------------------------------------|------------|---------------------|-------|--------|--------|--------|--------|--------|
| models/baseline_binary_sdobenchmark/metrics.yaml | 0.786      | 0.7691              | 0.705 | 0.7733 | 0.2004 | 0.5475 | 0.8563 | 0.5381 |

### Confusion matrix
![Confusion matrix](test_plots/confusion_matrix.png 'Confusion matrix')

### ROC Curve
![ROC Curve](test_plots/roc_curve.png 'ROC Curve')

### Test samples
![Test samples](test_plots/test_samples.png 'Test samples')

