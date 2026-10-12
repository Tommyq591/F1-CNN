# F1 CNN team classifier

An end-to-end PyTorch project for classifying a photo containing one main Formula One car into the teams in the training dataset. The CNN is built explicitly from layers without pretrained weights.

## Use in PyCharm

1. Select `venv\Scripts\python.exe` as the interpreter. For a fresh environment, install `python -m pip install -r requirements.txt`.
2. Run **app.py**. It loads the latest completed experiment. Click **Choose image** to see the predicted team and top three scores. **Load model** selects another `best.pt`.
3. Run **train.py** to train a full suite, or use the commands below. Each epoch prints its duration. CPU training can take substantial time; CUDA is used automatically when available.

```powershell
.\venv\Scripts\python.exe -m f1_cnn audit --data "data/cleaned"
.\venv\Scripts\python.exe -m f1_cnn train --data "data/cleaned" --epochs 20 --image-size 64
.\venv\Scripts\python.exe -m f1_cnn predict "C:/path/to/photo.jpg"
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

The dataset must directly contain one folder per team, e.g. `data/cleaned/Ferrari F1 car/*.jpg`. The default uses populated `data/cleaned`, otherwise `data/raw/Formula One Cars`. Use `--data` for another location. Existing train/test layouts are rejected to prevent accidental leakage. The supplied friend-cleaning summary in `docs/cleaning_report_supplied.json` is not an independently verified audit of the local original data.

## Experiment design

- The read-only audit applies EXIF orientation and RGB conversion, detects exact decoded-pixel duplicates, excludes conflicting-label images, and records unreadable/non-image files. Valid images with unusual extensions are decoded too. Original files are never rewritten or deleted.
- Unique images receive a reproducible stratified 70/15/15 train/validation/test split, with saved paths, labels and pixel hashes. Review and group near-duplicates before final training: hashes do not find related crops or consecutive frames.
- Square resizing and fixed RGB mean/std 0.5 are identical at inference. Square resizing stretches aspect ratio. The normalisation constants are not estimated from held-out images.
- Baseline and augmented models use the same seeds, architecture and maximum epoch budget. Training-only augmentation uses horizontal flips and rotations up to 10 degrees. Early stopping restores the best validation-loss checkpoint.
- Select using validation macro F1, breaking ties by accuracy. Assess the chosen configuration with three fresh development-fold models. This post-selection CV is not nested CV and can be optimistic.
- Evaluate the chosen original checkpoint once on the held-out test set. Save accuracy, balanced accuracy, macro F1, per-team precision/recall/F1, confusion matrix, errors, predictions CSV and a majority-class baseline. CV models do not replace the selected checkpoint.

Architecture: RGB -> two Conv(32)+ReLU -> pool -> two Conv(64)+ReLU -> pool -> two Conv(128)+ReLU -> pool -> global average pool -> dropout -> linear logits. Cross-entropy, Adam, weight decay and Kaiming/Xavier initialisation are explicit. No batch normalisation or transfer learning is used. This is compact VGG-inspired architecture, not standard VGG16.

Seeds improve repeatability, but results can differ across hardware and package versions. Each run records its configuration and environment. Model selection is frozen before the test result is calculated; use a new genuinely untouched holdout if later experiments are guided by that result.

## Files and evidence

### Focused improvement experiment

The recorded original model used **32 × 32** images and achieved **74.79% test accuracy / 67.40% test macro F1**. Its artifacts and checkpoints are preserved under `docs/evidence/baseline/` as well as the original run. See [the experiment log](docs/EXPERIMENT_LOG.md) for the weakness analysis, fixed protocol and measured outcome of the first resolution-only experiment.

Use the separate single-experiment runner to reuse the original split without evaluating the test set or changing the app's default checkpoint:

```powershell
# Estimate CPU cost first; each output path must be new.
.\venv\Scripts\python.exe scripts/benchmark_resolution.py --output .cache/experiments/new_benchmark.json
.\venv\Scripts\python.exe -m f1_cnn.experiments --output runs/new_resolution_run --benchmark .cache/experiments/new_benchmark.json
```

It changes image size to 128 × 128 only, inherits all other settings from the saved original baseline, verifies development-image hashes and disjoint splits, and exports labelled validation figures, CSV comparison tables, per-team metrics and an experiment-log entry to `docs/`. The original `train` command above still runs the full multi-model suite including a test evaluation; use the focused runner for this staged improvement workflow. Load a candidate `best.pt` manually using **Load model** in `app.py`. Checkpoints under `docs/` are also ignored by Git, so include them separately when transferring the evidence archive.

For the separate **longer-training experiment at the original 32px resolution**, change only the epoch cap:

```powershell
.\venv\Scripts\python.exe -m f1_cnn.experiments --epochs 24 --output runs/new_epochs_24_run
```

This starts from fresh initial weights with the saved baseline seed, retains patience 4 and all other original settings, and restores the checkpoint with minimum validation loss. The `--epochs` and `--image-size` options are mutually exclusive. A plan and runtime estimate are saved before training; the report includes per-team recall and a comparison of overlapping epochs with the original history. Existing run/evidence paths cannot be overwritten.

| Path | Purpose |
| --- | --- |
| `app.py` | Desktop image picker |
| `train.py` | PyCharm training entry point |
| `f1_cnn/` | Audit, loaders, CNN, trainer, inference and pipeline |
| `F1_CNN.ipynb` | Standalone implementation and results notebook |
| `F1_CNN.pynb.ipynb` | Updated copy at the original notebook filename |
| `scripts/build_notebook.py` | Regenerate notebooks after source changes; clears outputs |
| `docs/REPORT_CHECKLIST.md` | Assignment requirements mapped to saved evidence |
| `AI_USAGE.md` | Prompt/attribution log requiring team completion |
| `runs/latest.json` | Pointer written only after the complete suite succeeds |
| `runs/<run>/summary.json` | Actual dataset, selected checkpoint and headline results |
| `runs/<run>/figures/` | Dataset, architecture, confusion and error figures |
| `runs/<run>/{baseline,augmented,cv_fold_*}/` | Configurations, histories, learning curves and checkpoints |

The notebooks display the latest completed run by default. Set `RUN_TRAINING=True` to run a new suite, or `RUN_DIR` to inspect a specific result. Implementation source is embedded, so `f1_cnn` imports are not needed. Dataset and saved run folders are still needed for training or evidence display. After moving the project, update `runs/latest.json` or select a checkpoint/run explicitly.

## Limitations and submission

The classifier always chooses a known team, even for unrelated photos. Softmax scores are not calibrated certainty. It does not locate cars, separate multiple cars, or automatically recognise new teams/liveries. The original local dataset contains eight historical labels: AlphaTauri, Ferrari, McLaren, Mercedes, Racing Point, Red Bull Racing, Renault and Williams. Judge recognition quality using measured results; successful execution alone does not imply good accuracy.

The assignment requires your own critical-code explanations, student-written PDF, truthful individual contributions and a complete prompt log. Generated comments cannot stand in for personal understanding. Follow `docs/REPORT_CHECKLIST.md`, insert real names/IDs and confirm the brief's `Assign2` filename discrepancy. References are in `docs/REFERENCES.md`.

Datasets, caches and large weights are ignored by Git but remain in this PyCharm project. Copy the required run/checkpoint when transferring the project.

`requirements-lock.txt` records the installed environment used during verification. CPU-specific PyTorch builds may need the PyTorch wheel index when reproducing this exact lock; `requirements.txt` is the portable dependency entry point.
