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
  path: data/sdo-dataset-cls-211-24h
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
name: Binary classification model on SDO-Dataset
training_type: train
tune_lr: false
path: models/baseline_binary_sdodataset
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
training_time: 134.19s
model_size: 323kB
early_stopping_epoch: 0
model_checkpoint_step: 323
model_checkpoint_epoch: 17
tracking_id: SOLN-324
dataset:
  training_set_size: 2401
  validation_set_size: 362
  test_set_size: 364
```
## Test
### Metrics
| Path                                           | accuracy   | balanced_accuracy   | csi    | f1     | far   | hss    | pod    | tss    |
|------------------------------------------------|------------|---------------------|--------|--------|-------|--------|--------|--------|
| models/baseline_binary_sdodataset/metrics.yaml | 0.5852     | 0.5886              | 0.2135 | 0.5234 | 0.75  | 0.1161 | 0.5942 | 0.1773 |

### Confusion matrix
![Confusion matrix](test_plots/confusion_matrix.png 'Confusion matrix')

### ROC Curve
![ROC Curve](test_plots/roc_curve.png 'ROC Curve')

### Test samples
![Test samples](test_plots/test_samples.png 'Test samples')

