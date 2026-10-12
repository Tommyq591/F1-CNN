# AI assistance log — incomplete starter

The brief requires every prompt, a one-line summary of its response, and how it was used. Add the earlier planning/dataset exchanges and every subsequent prompt. Team members must review this log and supply accurate usage descriptions.

## Foundation implementation

Prompt: “Alright its downloading now. Set up the foundations for the brief, The plan is to show F1 images and it should be able to detect what team is in the photo. Set up the necessary classes you believe will be needed.”

Response summary: Created a PyTorch classification scaffold with dataset auditing, splits, explicit CNN layers, training/evaluation, checkpoint inference, an instructional notebook and synthetic tests.

Use in submission: TODO — record which generated components you retained, adapted and independently explained. No real-data results or student-written report text were generated.

## Project completion request — 11 October 2026

Prompt: "One of my friends did the data wrangling." The user supplied the original/cleaned dataset summary below, followed by: "Anways I need you to do this project fully for me. it should follow the rules on this document. Were creating the F1-CNN where we put in an F1 image and it should be able to tell what team is in the image by training. Update all changes to the files in my pycharm project so I have a full working project."

Supplied document: CS4287-Assign1-CNNs-S1AY2627-v1.docx.

```text
ORIGINAL DATASET
Photos checked:                      2,471
Completely clean photos:             2,143
Photos with warnings:                326
Failed / broken photos:              2
Non-image files skipped:             15
Exact duplicate groups found:        21

CHANGES MADE
Images standardised / fixed:          2,429
Broken images removed:                2
Same-team duplicate copies removed:   2
Cross-team conflicts quarantined:     38
Cleaning operations that failed:      0

CLEANED DATASET
Photos remaining:                     2,429
Completely clean photos:              2,429
Photos still with warnings:           0
Photos still broken:                  0
Exact duplicate groups remaining:     0
Non-JPEG images remaining:            0
Non-RGB images remaining:             0
Wrong file extensions remaining:      0
Team names in filenames remaining:    0
```

Response summary: Extended the scaffold with read-only auditing, baseline/augmentation training, development cross-validation, held-out evaluation, saved evidence, CLI, desktop prediction app, standalone notebooks and integration tests; distinguished supplied cleaning counts from locally audited data.

Use in submission: AI-generated implementation and technical guidance are stored in this project. TODO — each member must record what was retained or changed, verify results, add personal critical-code explanations and disclose how the material was used. PDF prose and statements of personal contribution remain the team's work.

This log covers the visible request and pre-existing starter entry. It does not certify that all prompts from other chats or team members have been recorded.

## 12 October 2026 — controlled improvement and report evidence

User requested preservation of the 74.79% accuracy / 67.40% macro-F1 baseline, analysis of per-team errors and overfitting, one promising bounded experiment initially, validation-only model selection, and automatic report evidence. Constraints included a PyTorch CNN built from scratch, an intact original model/checkpoint, a working `app.py`, and separate tests of resolution, augmentation and optimisation.

AI assistance: inspected saved baseline artifacts, identified the actual 32px input and AlphaTauri/Williams failure pattern, implemented a separate one-factor resolution runner and preservation/isolation tests, benchmarked CPU cost, and started a fresh 128px experiment using the original split, settings and seed. The runner records actual outcomes in `docs/EXPERIMENT_LOG.md`; no improved result should be claimed without those measured artifacts. Augmentation and optimisation changes are staged proposals, not completed experiments. Original model/trainer source and results were preserved. Students should verify, interpret and disclose this assistance in their own report.

Completed outcome: the 128px run used 12 epochs and 24.03 minutes of epoch time. Validation accuracy remained 72.88%; macro F1 increased from 65.87% to 69.24%, while validation loss worsened from 0.8101 to 0.8580. Per-team improvements and regressions, limitations and report figures/tables are documented. No new test-set evaluation was performed and no further full-dataset experiment was started.
