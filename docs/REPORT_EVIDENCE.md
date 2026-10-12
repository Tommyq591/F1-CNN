# Evidence guide for the university report

Start with [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). It separates recorded measurements, hypotheses, the predeclared protocol and the measured experiment outcome. The student's report should explain and interpret this evidence in their own words.

[EXPERIMENT_VERIFICATION.md](EXPERIMENT_VERIFICATION.md) records the passing tests, application checks, artifact preservation and visual checks.

## Original results

| Evidence | Suggested use / caption |
| --- | --- |
| [Original validation confusion matrix](evidence/baseline_validation_confusion_matrix.png) | Original 32px CNN, epoch 12, 365 validation images. Rows are actual teams, columns predicted teams. AlphaTauri has zero correct predictions; Mercedes and AlphaTauri are often classified as Williams. |
| [Original historical test confusion matrix](evidence/baseline_historical_test_confusion_matrix.png) | Historical test result from the original completed study, included for baseline documentation. It is not a new evaluation of the 128px candidate. |
| [Original curves](evidence/baseline/baseline/learning_curves.png) | Original training and validation loss/accuracy over 12 epochs. Validation loss is lowest at the final epoch; the curves do not show sustained severe overfitting. |
| [Original per-team table](evidence/baseline_per_team.csv) | Precision, recall, F1 and support, with validation and historical test rows explicitly separated. Metric values are fractions from 0 to 1. |
| [Original configuration and checkpoint](evidence/baseline/baseline/) | Exact saved settings, weights, epoch history and validation metrics. |
| [Historical comparison table](evidence/historical_comparison.csv) | The already completed 32px baseline and flip/rotation experiment. Training times are sums of epoch timers. |
| [Snapshot checksums](evidence/baseline_hashes.json) | SHA-256 checksums proving that original run files and the preserved copies agree. |
| [Resolution benchmark](evidence/resolution_benchmark.json) | Measured synthetic-batch timings used to estimate runtime before training. These are not timings of a complete training run. |

## Resolution-only experiment

The completed candidate evidence is in [evidence/resolution_128_seed42](evidence/resolution_128_seed42/). The run retained epoch 12: validation accuracy 72.88% (unchanged), macro F1 69.24% (+3.37 points), and loss 0.8580 (worse than 0.8101). Training took 24.03 minutes; recorded experiment wall time was 27.92 minutes. No candidate test evaluation was performed.

| File | Suggested use / caption |
| --- | --- |
| `comparison.csv` | Direct comparison on the same 365 validation images: accuracy, macro F1, balanced accuracy, selected-checkpoint loss, best epoch, epochs completed and training time. No candidate test score is included. |
| `comparison_learning_curves.png` | Training and validation loss/accuracy for original 32px and candidate 128px input. Dashed lines are training; solid lines are validation. Colours identify the input resolution. |
| `per_team_validation.csv` | Per-team precision, recall and F1 before/after the resolution change, with supports and F1 differences. |
| `per_team_validation_f1.png` | Comparison of F1 across all eight teams on the same validation split. |
| `validation_confusion_matrix.png` | Candidate confusion counts using the checkpoint with minimum validation loss, with actual labels on rows and predictions on columns. |
| `baseline_validation_confusion_matrix.png` | Original validation matrix with the same label order for direct visual comparison. |
| `validation_predictions.csv` | One row per validation image, with actual and predicted team and whether the prediction is correct. |
| `validation_errors.png` | First 12 validation mistakes in saved manifest order, where available. These examples are illustrative, not a random or representative sample. |
| `history.json` | Per-epoch measured training/validation losses, accuracies and durations. Macro F1 is measured after restoring the selected checkpoint, not at every epoch. |
| `protocol.json` / `config.json` | Original and updated settings, selection rules and split-manifest hash. Only image size changes. |
| `environment.json` | Software versions, CPU thread count and hashes of the implementation used for the run. |
| `result.json` | Final candidate validation metrics, best epoch, time and the predeclared preference decision. |

## Interpretation rules

- Compare validation with validation. The original 74.79% is a historical **test** accuracy; it cannot be directly used as the threshold for claiming a validation improvement.
- Macro F1 weights each team equally and complements accuracy when team counts differ. Report both, plus AlphaTauri's recall and support; do not hide weak classes in an aggregate.
- Distinguish “resolution changed the measured validation result” from the unproven explanation “the model learned logos better.” No saliency study or feature attribution establishes that mechanism here.
- A single seed/split does not establish reliable superiority. The 18 AlphaTauri validation examples make its measured recall especially sensitive to individual images.
- The two runs have the same epoch budget and architecture, but different compute cost. Report that cost alongside predictive performance.
- Training curves include dropout and evolving weights, while validation uses evaluation mode. Interpret their gap with this limitation in mind.
- Further augmentation/optimisation experiments listed in the log are proposals. Do not describe them as completed, or claim that a fresh test evaluation occurred.

Checkpoint files and the `runs/` directory are Git-ignored. Copy them explicitly when transferring the project or submitting an evidence archive. The full baseline backup includes all original suite models. Candidate weights are at `runs/resolution_128_seed42/best.pt`; the app's **Load model** control can open them without replacing its original default.
