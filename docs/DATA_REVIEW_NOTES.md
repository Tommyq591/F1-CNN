# Observations from the local dataset

These are technical review notes for the team, not PDF submission prose.

- The independent audit retained 2,429 images from the local original directory. It excluded two unreadable image files, two same-label duplicate copies, and 19 conflicting-label pairs (38 images), and skipped 15 non-image files. The removal counts agree with the supplied friend-cleaning summary. This does not establish that the original files equal the friend's standardised JPEGs.
- The local retained classes contain 119 AlphaTauri, 367 Ferrari, 362 McLaren, 310 Mercedes, 285 Racing Point, 335 Red Bull Racing, 315 Renault and 336 Williams images. The smallest class has roughly one third as many images as the largest; inspect macro F1, balanced accuracy and per-team recall.
- The saved training example grid contains photographs, rendered cars, a Ferrari event poster, a Renault toy/model car and images showing multiple cars. Decide whether those examples match the intended use case. Format cleaning does not resolve semantic ambiguity.
- Team labels are historical folder labels. The model's output vocabulary is exactly the checkpoint's `class_names`. Do not promise recognition of teams absent from these labels or all new liveries.
- A 32x32 run is a small-resolution CPU demonstration. Fine lettering, logos and shape detail may disappear. Its performance cannot be transferred to the default 64x64 configuration without running that configuration.
- Exact decoded-pixel hashing prevents exact copies from crossing splits. Related views, crops, edits, and frames from the same scene may remain. A final dataset review should group or remove these before the final split. Preserve a manifest of those decisions and retrain if membership changes.
- Normalisation and augmentation happen in memory. Original files remain untouched, including format warnings. The audit is read-only and therefore is not a replacement for the friend's saved cleaning work.

The generated notes and figures must be attributed in the AI log and interpreted by the students themselves.
