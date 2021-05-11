# Model report
## Training
### Parameters
```yaml
data:
  name: sdo-dataset
  channel: 171
  size: 256
  targets:
    classes:
    - Quiet: < 1e-6
    - '>=C': '>= 1e-6'
  path: /data1/data/sdo-dataset
model:
  activation: leakyrelu
trainer:
  epochs: 20
  batch_size: 128
  learning_rate: 0.001
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
        (3): LeakyReLU(negative_slope=0.01)
        (4): Dropout2d(p=0.1, inplace=False)
      )
      (1): Sequential(
        (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(3, 3))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
        (3): LeakyReLU(negative_slope=0.01)
        (4): Dropout2d(p=0.1, inplace=False)
      )
      (2): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(3, 3))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
        (3): LeakyReLU(negative_slope=0.01)
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
  (train_accuracy): Accuracy()
  (val_accuracy): Accuracy()
  (test_metrics): MetricCollection(
    (Accuracy): Accuracy()
    (F1): F1()
    (Recall): Recall()
    (StatScores): StatScores()
    (AUROC): AUROC()
  )
)
================================================================================
Total parameters: 25026
```
### Loss curve
![Loss curve](train_plots/loss_curve.png 'Loss curve')

### Accuracy curve
![Accuracy curve](train_plots/accuracy_curve.png 'Accuracy curve')

### Metadata
```yaml
machine: 'lambda02 | Linux #113-Ubuntu SMP Thu Jul 9 23:41:39 UTC 2020 | 10 cores @ 4120.00Mhz | RAM 126 GB | 2x TITAN RTX'
training_time: 687.52s
model_size: 327kB
early_stopping_epoch: 0
model_checkpoint_step: 559
model_checkpoint_epoch: 15
tracking_id: SOLN-136
dataset:
  training_set_size: 4401
  validation_set_size: 672
  test_set_size: 646
```
## Test
### Metrics
| Path                                  | accuracy   | auroc   | balanced_accuracy   | csi     | f1     | far     | hss     | pod     | tss     |
|---------------------------------------|------------|---------|---------------------|---------|--------|---------|---------|---------|---------|
| models/binary_sdodataset/metrics.yaml | 0.68885    | 0.68083 | 0.63541             | 0.65104 | 0.5996 | 0.13395 | 0.21665 | 0.72394 | 0.27081 |

### Confusion matrix
![Confusion matrix](test_plots/confusion_matrix.png 'Confusion matrix')

### ROC Curve
![ROC Curve](test_plots/roc_curve.png 'ROC Curve')

### Test samples
![Test samples](test_plots/test_samples.png 'Test samples')

