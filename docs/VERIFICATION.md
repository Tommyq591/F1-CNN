# Local verification record

Date: 11 October 2026. These are measured engineering results, not student-authored report prose.

## Checks completed

- Five automated tests passed: reproducible disjoint splits, duplicate/conflicting-label handling, training/checkpoint/inference equivalence, full experiment export including CV/test separation, and unusual filename/existing-split handling.
- The real Tk application constructed successfully. A second check exercised its image-selection callback, created the photo preview, ran a real trained checkpoint in the inference worker and displayed three ranked results. The window was hidden during this automated check.
- CLI help was checked. Notebook JSON/schema and Python syntax were validated.
- Dataset distribution, training examples, dimensions, RGB correlation, architecture diagram and both experiment learning curves were visually inspected.

## Local experiment

Run folder: `runs/local_original_32px`. Source: `data/raw/Formula One Cars`, audited read-only. The independent audit retained 2,429 images; this is not a claim that the friend's cleaned JPEG folder was provided.

Configuration: 32x32 RGB, 12 maximum epochs, batch size 32, Adam learning rate 0.001, weight decay 0.0001, dropout 0.3, seed 42, patience 4, CPU with four threads. Six convolutions, 288,040 trainable parameters, no pretrained weights. Split: 1,699 train / 365 validation / 365 test.

| Configuration | Validation accuracy | Balanced accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Baseline | 72.88% | 67.00% | 0.6587 |
| Horizontal flips and rotations | 71.23% | 65.51% | 0.6399 |

Both experiments completed 12 epochs and retained epoch 12 by validation loss. The baseline was selected by validation macro F1 before test evaluation. Its AlphaTauri validation recall was zero; aggregate accuracy masks this weakness. Inspect per-team results and do not promise reliable recognition of every class.

The default training entry point uses 64x64 and 20 maximum epochs. The above measurements apply only to the explicitly configured 32x32 run.
