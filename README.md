# README

This repository contains an example of the code to load the [SeizeIT2 dataset](https://openneuro.org/datasets/ds005873) and to train the model included in the [dataset paper](https://arxiv.org/abs/2502.01224).

# loader_test.py
Script with an example for loading files from the dataset. The modules `data_loader.data` and `data_loader.annotation` are used to create a data object, containing the signal data and extra information, and an annotation object, containing all information regarding the seizure events of the recording.

# main_net.py
Script to train and evaluate the seizure detection baselines described in the paper. The repository now supports the PyTorch deep-learning models (`ChronoNet`, `EEGnet`, `DeepConvNet`) as well as the feature-based baselines `SVM` and `XGB`. The data generators are likely to take a long time to run (arround 3 hours), hence the option to save the training and validation generators and load them in future runs. By default, `main_net.py` loads cached generators from `data_loader/generators/` when they exist.

`main_net.py` uses command-line arguments. Running without arguments keeps the default paper-style configuration:

```bash
python main_net.py
```

Common examples:

```bash
# Use a custom dataset path
python main_net.py \
  --data-path /data1/zhihao/SeizeIT2/ds005873-1.1.0/ds005873

# Train a different deep-learning model
python main_net.py --model EEGnet --epochs 100 --batch-size 64

# Train the paper-style feature-based SVM baseline
python main_net.py --model SVM

# Train the feature-based XGBoost baseline
python main_net.py --model XGB

# Train a foundation-model adapter
python main_net.py --model STEEGFormer --fs 128
python main_net.py --model BIOT --fs 200
python main_net.py --model BENDR --fs 256
python main_net.py --model CBraMod --fs 200
python main_net.py --model Conformer
python main_net.py --model EEGPT --fs 256

# Fine-tune from an ST-EEGFormer checkpoint
python main_net.py \
  --model STEEGFormer \
  --fs 128 \
  --steegformer-variant small \
  --steegformer-pretrained /path/to/STEEGFormer-small.pth

# Use a specific GPU
python main_net.py --model ChronoNet --device cuda:0

# Force CPU
python main_net.py --model ChronoNet --device cpu

# Save generated datasets for reuse (first run, or when regenerating segments)
python main_net.py --no-load-generators --save-generators

# Regenerate datasets from raw EDF files without saving
python main_net.py --no-load-generators

# Run only prediction and evaluation from an existing checkpoint
python main_net.py --skip-train
```

## ChronoNet warm-up experiments

Neural-network training supports an optional warm-up phase inspired by [Brain-inspired warm-up training with random noise for uncertainty calibration](paper/s42256-026-01215-x.pdf). During the first `--warmup-epochs`, Gaussian noise with fixed standard deviation `1.0` (same units as the preprocessed EEG, µV) is added to the real training inputs: `x = x + N(0, 1)`. Validation always uses clean EEG. If `--random-label` is set, warm-up training uses uniformly random 0/1 labels resampled every batch; otherwise the original segment labels are kept.

Warm-up applies only to PyTorch models trained through `train_net` (for example `ChronoNet`, `EEGnet`, `DeepConvNet`, and the adapter models). It does not affect `SVM` or `XGB`.

Experiment names include a `warmup<N>` suffix when warm-up is enabled, and `randlabel` when random labels are used. The three comparison runs below all use `ChronoNet`, `--epochs 300`, and `--seed 1` so that only the warm-up setting changes:

```bash
# (1) Baseline: no warm-up, train on clean EEG for all 300 epochs
python main_net.py \
  --model ChronoNet \
  --epochs 300 \
  --seed 1

# (2) Warm-up for 30 epochs with noisy EEG and real labels, then 270 epochs on clean EEG
python main_net.py \
  --model ChronoNet \
  --epochs 300 \
  --warmup-epochs 30 \
  --seed 1

# (3) Warm-up for 30 epochs with noisy EEG and random labels, then 270 epochs on clean EEG
python main_net.py \
  --model ChronoNet \
  --epochs 300 \
  --warmup-epochs 30 \
  --random-label \
  --seed 1
```

Saved under `net/save_dir/models/` as:
- `ChronoNet_subsample_factor5/`
- `ChronoNet_subsample_factor5_warmup30/`
- `ChronoNet_subsample_factor5_warmup30_randlabel/`

Cached generators are loaded by default. On a first run, create them once with `--no-load-generators --save-generators` before running these experiments.

During training, the terminal shows per-batch `tqdm` progress bars for both train and validation phases, including current loss, processed samples, samples per second, and per-epoch elapsed time. The training phase reports loss and accuracy only; full overlap/false-alarm metrics are computed on the validation phase.

The command-line arguments are grouped in `python main_net.py --help` as experiment, data/preprocessing, training, runtime/workflow, and method-specific options. Shared arguments include `--data-path`, `--save-dir`, `--model`, `--dataset`, `--sample-type`, `--epochs`, `--batch-size`, `--lr`, `--l2`, `--dropout-rate`, `--seed`, `--warmup-epochs`, `--random-label`, `--device`, `--load-generators` / `--no-load-generators`, `--save-generators`, `--skip-train`, `--skip-predict`, and `--skip-evaluate`. Method-specific options such as `--svm-c`, `--svm-gamma`, and the `--xgb-*` options are declared in their corresponding `methods/` modules.

Evaluation follows the original SeizeIT2 GitHub code and the paper's validation protocol. Predictions are first filtered with the EEG RMS artifact rule (`13-150 uV`) and then post-processed by merging alarms separated by less than 2 seconds and keeping only alarms lasting at least 8 seconds in a 10-second margin. The result file `net/save_dir/results/<experiment>.h5` stores the full threshold sweep for overlap-based and epoch-based metrics. A companion `net/save_dir/results/<experiment>_summary.csv` reports the paper-style summary columns: sensitivity and false alarms per hour at threshold `0.5`, plus AUROC, AUPR, and AUSF from the threshold curves.

The deep-learning code now uses PyTorch modules, `torch.utils.data.Dataset`/`DataLoader`, and PyTorch checkpoints. Existing TensorFlow/Keras `.h5` weights are not compatible with this version and should be regenerated by training the PyTorch models. New deep-learning checkpoints are written as `.pt` files under `net/save_dir/models/<experiment>/Callbacks` and `net/save_dir/models/<experiment>/Weights`. The feature-based `SVM` and `XGB` baselines are stored as pickles under `net/save_dir/models/<experiment>/Weights/<experiment>.pkl`.

For the `SVM` option, the implementation follows the paper description in `paper/SeizelT2.pdf`: 2-second windows with 50% overlap, 1-25 Hz Butterworth filtering, RMS-based rejection below `13 uV` and above `150 uV`, 42 handcrafted features from the two bte-EEG channels, and an RBF-kernel support vector machine. The `XGB` option reuses the same 42 handcrafted features and RMS-based validity filtering, but replaces the classifier with an XGBoost ensemble. By default, both feature-based baselines load cached EEG segment generators for the train/validation splits; use `--no-load-generators` to rebuild preprocessing from raw EDF files instead.

## ST-EEGFormer adapter

`STEEGFormer` follows the downstream-model pattern from the official [ST-EEGFormer repository](https://github.com/LiuyinYang1101/STEEGFormer/tree/main), but is wired into this SeizeIT2 training pipeline instead of replacing it. The existing `data_loader` preprocessing, window generation, cached generator workflow, prediction files, RMS artifact filtering, and paper-style evaluation remain unchanged.

What was added:
- `methods/STEEGFormer.py`: a compact ST-EEGFormer ViT adapter with EEG temporal patch embedding, temporal/channel positional embeddings, optional checkpoint loading, optional frozen-backbone fine-tuning, and an output head for seizure/non-seizure classification.
- `main_net.py`: `--model STEEGFormer` plus `--steegformer-*` command-line arguments.
- `trains/main_func.py` and `trains/config.py`: model registration and experiment names that include the ST-EEGFormer variant and target sampling rate.
- `requirements.txt` and `environment.yml`: `timm`, required by the upstream ViT implementation.

Important notes:
- Use `--fs 128` for the closest match to the released ST-EEGFormer checkpoints. If another `--fs` is used, the adapter resamples the model input internally to `--steegformer-target-fs` while leaving the repository's raw-data preprocessing path intact.
- SeizeIT2 uses behind-the-ear channels (`BTEleft SD`/`BTEright SD` and `CROSStop SD` fallback). These are not standard 10-20 scalp channels in the public ST-EEGFormer channel table. The adapter therefore uses explicit channel embedding indices from `--steegformer-channel-indices` (default `143,144`) rather than silently pretending the channels are standard scalp electrodes.
- If a checkpoint's classification head does not match the two-class seizure task, the adapter skips that head and initializes a fresh two-class head.

## Additional benchmark model adapters

The repository also includes compact adapters based on the upstream ST-EEGFormer benchmark model directory (`benchmark/neural_networks/models`): `BIOT`, `BENDR`, `CBraMod`, `Conformer`, and `EEGPT`. They are registered in the same `main_net.py` interface and keep the existing SeizeIT2 `data_loader`, segment generation, RMS artifact filtering, and paper-style evaluation.

Examples:

```bash
python main_net.py --model BIOT --fs 200 --biot-pretrained /path/to/biot.ckpt
python main_net.py --model BENDR --fs 256 --bendr-pretrained /path/to/bendr_encoder.pt
python main_net.py --model CBraMod --fs 200 --cbramod-pretrained /path/to/pretrained_weights.pth
python main_net.py --model Conformer --fs 250
python main_net.py --model EEGPT --fs 256 --eegpt-pretrained /path/to/eegpt.ckpt
```

Notes:
- `BIOT`, `BENDR`, `CBraMod`, and `EEGPT` expose target sampling-rate arguments and internally resample the model input if needed; this does not change the saved data generators or the paper-style preprocessing path.
- `Conformer` is trained from scratch and does not require a pretrained checkpoint.
- The adapters are deliberately scoped to the two-channel SeizeIT2 seizure/non-seizure task. Checkpoint loading is shape-aware: incompatible classifier heads or upstream task-specific layers are skipped instead of forcing an unsafe load.

Every run now also appends a summary line to `net/save_dir/logs/experiments.csv`, including the experiment name, model, key configuration values, whether cached generators were reused, and a compact training/evaluation summary. This makes it easier to track repeated experiments without manually copying terminal output.

## Project layout
- `data_loader/`: BIDS/EDF loading, annotation loading, preprocessing dataset builders, and patient split TSV files.
- `methods/`: model definitions for ChronoNet, EEGNet, DeepConvNet, and the feature-based SVM/XGB baselines.
- `trains/`: experiment configuration, training loop, prediction, and evaluation orchestration.
- `utils/`: shared EEG preprocessing, losses, and metrics.
- `net/save_dir/`: default output location for checkpoints, predictions, histories, and result files.

## Conda environment setup
The python packages (and corresponding versions) used in the development of the scripts in this repository are gathered in 'environment.yml'. To easily create a conda environment with the same package versions to run the code, follow the instructions below:
```
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels nvidia
conda config --set channel_priority strict
conda env create -n ENV_NAME -f environment.yml
```
