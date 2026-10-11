# Assignment report preparation

This is a planning checklist, not submission prose. Write the PDF in your own voice. The brief requires student-written critical-code commentary and explicitly prohibits AI improvements to the PDF's English. Do not describe generated code or these explanations as your own work without attribution.

1. **Title page and contents:** supply the project title, module, names, IDs, and accurate date.
2. **Dataset (2 marks):** cite the dataset; verify version, licence, download date and actual team names. Include `figures/class_distribution.png`, `training_examples.png`, `image_dimensions.png` and `rgb_correlation.png`. Explain class imbalance, historical team labels, varied backgrounds, label ambiguity, possible near-duplicates and preprocessing. Mean RGB correlation is descriptive, not proof of predictive relationships. The friend-supplied cleaning counts are in `docs/cleaning_report_supplied.json`; do not attribute them to a training run unless that exact cleaned dataset was used.
3. **Architecture and hyperparameters (3 marks):** use `figures/architecture.png`, the saved `config.json` and `environment.json`. Explain the six explicitly constructed 3x3 convolution layers, ReLU, three max pools, global average pooling, dropout, linear logits, Kaiming/Xavier initialisation and weight decay. This is a compact VGG-inspired architecture, not VGG16. There is no batch normalisation or transfer learning. Discuss these design choices, including information lost by resizing to 64x64 and stretching the aspect ratio.
4. **Loss, optimiser and cross-validation (2 marks):** explain cross-entropy on logits, Adam, the learning rate, batch size, early stopping and best-validation-loss checkpoint. Use `cross_validation.json` and fold histories. The selected configuration is assessed on three development folds; this is not nested CV. The untouched test result is the final estimate. Do not tune on the test set after seeing it.
5. **Results and evaluation (3 marks):** use `summary.json`, `test_metrics.json`, `figures/test_confusion_matrix.png`, `test_predictions.csv` and, when present, `test_errors.png`. Include accuracy, balanced accuracy, per-team precision/recall/F1, macro F1 and the majority-class baseline. Inspect failure cases and train/validation loss divergence. Distinguish train-mode accuracy (with dropout and possibly augmentation) from evaluation accuracy.
6. **Independent experiment (3 marks):** compare baseline and augmented configurations from `selection.json` and each experiment's learning curves. The split, seed, architecture, optimiser and maximum epoch budget are held fixed. Training-only horizontal flips and rotations vary. Early stopping can produce different realised budgets, so report best/stopped epochs. Discuss mirrored text, rotation borders and augmentation limitations. Do not assume augmentation must improve results.
7. **Difficulty (2 marks):** give your own short, reasoned judgement. The teaching team determines the actual mark; a VGG-inspired model does not automatically earn the VGG rubric mark.
8. **Statement of work:** one truthful paragraph per member. Attribute the friend's wrangling, implementation review, experiments and report work accurately.
9. **Generative AI record:** complete `AI_USAGE.md` with every prompt from every member and prior chat, one response summary and how each response was used.
10. **References:** use `docs/REFERENCES.md` as a verified starting list and apply the required Cite It Right style yourself.

## Notebook submission

- Use `F1_CNN.ipynb`, which embeds the implementation and does not require the `f1_cnn` package. The original `F1_CNN.pynb.ipynb` is also updated for continuity.
- Enter true names and IDs on its first comment line. Update the execution comment only after restarting the kernel and running every cell in your submission environment.
- The third comment line lists source URLs and the AI log. Keep and extend attribution when adapting code.
- Replace or expand the generated critical-code comments with explanations demonstrating your own understanding. Be ready for the required walkthrough.
- The supplied brief calls the file `CS4287-Assign2-ID1-ID2-ID3` despite being titled Assignment 1. Confirm this discrepancy with the lecturer; do not invent IDs.
- Re-run with the final cleaned data if the current evidence used a different source. Ensure the notebook points at that completed run and includes measured outputs.
- Submit the student-written PDF, correctly named notebook, complete AI log, and any data/checkpoint attachments allowed by the lecturer. Local datasets and model files are intentionally ignored by Git.

Deadline stated in the supplied brief: 23:59 Saturday 31 October 2026.
