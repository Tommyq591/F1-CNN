"""Build a portable notebook by embedding the project's implementation source."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def standalone_source(path):
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines(keepends=True)
    # Relative package imports are unnecessary because definitions live in earlier cells.
    remove = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level:
            remove.update(range(node.lineno - 1, node.end_lineno))
    return ''.join(line for i, line in enumerate(lines) if i not in remove)


def main():
    cells = []

    def code(source):
        cells.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                      'source': source.splitlines(keepends=True)})

    def markdown(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': source.splitlines(keepends=True)})

    code('# Team members: TODO enter the three names and student IDs truthfully.\n'
         '# Execution: generated notebook; see saved outputs and verify Run All on the submission machine.\n'
         '# Sources: https://arxiv.org/abs/1409.1556 ; https://www.kaggle.com/datasets/vesuvius13/formula-one-cars ; AI_USAGE.md\n')
    markdown('# F1 team image classification\n\n'
             'This notebook includes the implementation layer by layer and can run without the `f1_cnn` package. '
             'It predicts one of the dataset teams for a photo with one principal car. It is not an object detector.\n\n'
             'Generated code and commentary must be reviewed and explained in your own words. Write your PDF yourself '
             'as required by the assignment. Complete names, IDs, contributions and the AI prompt log truthfully.\n\n'
             'Install `requirements.txt` in the notebook kernel environment before running. The default displays a completed '
             'local run when available. Set `RUN_TRAINING = True` to train a fresh full suite on the chosen dataset.')
    code("from pathlib import Path\nimport json\nimport os\nimport torch\nfrom IPython.display import display, Image as DisplayImage\n\n"
         "PROJECT_ROOT = Path.cwd()  # Open the notebook from the project folder.\n"
         "os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.cache' / 'matplotlib'))\n"
         "torch.set_num_threads(4)  # Bound CPU parallelism to keep desktop training responsive.\n"
         "CLEANED_ROOT = PROJECT_ROOT / 'data' / 'cleaned'\n"
         "DATA_ROOT = CLEANED_ROOT if CLEANED_ROOT.is_dir() and any(CLEANED_ROOT.iterdir()) else PROJECT_ROOT / 'data' / 'raw' / 'Formula One Cars'\n"
         "RUN_TRAINING = False  # True runs baseline, augmentation, three development folds and one final test.\n"
         "RUN_DIR = None  # Set to an existing run path, or leave None to find the latest completed run.\n"
         "PHOTO_PATH = None  # Set to a new photo path for optional inference at the end.\n")
    modules = [
        ('config', 'Configuration', 'Immutable configuration records the hyperparameters with each checkpoint.'),
        ('data', 'Data audit and preprocessing', 'The read-only audit detects identical decoded pixels, rejects conflicting labels and keeps class ordering. RGB conversion, EXIF orientation, resizing and fixed normalisation are identical at inference. Augmentations occur only in training. Exact hashing does not find related crops or frames: review these before final experiments.'),
        ('model', 'Explicit CNN architecture', 'Three blocks each contain two explicitly constructed 3x3 convolutions and ReLU activations, followed by max pooling. Global average pooling, dropout and a linear layer return logits. This is a compact VGG-inspired network without batch normalisation or pretrained weights.'),
        ('training', 'Loss optimiser and training', 'Cross-entropy accepts logits and integer class labels. Adam and weight decay update the parameters. Gradients are disabled for validation; the best validation-loss weights are restored after training or early stopping.'),
        ('predict', 'Checkpoint inference', 'The checkpoint stores both preprocessing configuration and class order. Softmax scores rank known teams; they do not measure calibrated certainty or reject non-F1 photos.'),
        ('plots', 'Figures and tables', 'Figures come from actual samples and measured results. Exploratory visualisation uses training images. Test error analysis is produced only after model selection.'),
        ('pipeline', 'Controlled experiments and cross-validation', 'Use a fixed stratified 70/15/15 split. Compare baseline and augmentation using identical seeds and maximum training budgets. Select by validation macro F1, then accuracy. Run three folds on development data with fresh weights, then evaluate the selected checkpoint once on the held-out test set. Development CV follows selection and is not nested CV.')]
    for index, (module, title, description) in enumerate(modules, 1):
        markdown(f'## {index}. {title}\n\n{description}')
        code(standalone_source(ROOT / 'f1_cnn' / f'{module}.py'))
    markdown('## 8. Run or load the experiments\n\nThe cleaned dataset must be a directory of team folders. '
             'The reported friend-cleaning counts do not certify that the current DATA_ROOT is that version. '
             'Every completed run records its actual source path, audit and split hashes. New training uses a unique run directory.')
    code("config = TrainingConfig(epochs=20, image_size=64, patience=6)\n"
         "if RUN_TRAINING:\n"
         "    from datetime import datetime\n"
         "    RUN_DIR = PROJECT_ROOT / 'runs' / datetime.now().strftime('notebook_%Y%m%d_%H%M%S_%f')\n"
         "    summary = run_suite(DATA_ROOT, RUN_DIR, config, folds=3)\n"
         "elif RUN_DIR is None and (PROJECT_ROOT / 'runs' / 'latest.json').exists():\n"
         "    RUN_DIR = Path(json.loads((PROJECT_ROOT / 'runs' / 'latest.json').read_text())['run'])\n"
         "if RUN_DIR is not None:\n"
         "    RUN_DIR = Path(RUN_DIR)\n"
         "    summary = json.loads((RUN_DIR / 'summary.json').read_text())\n"
         "    print(json.dumps(summary, indent=2))\n"
         "    print('These results used:', summary['data_root'])\n"
         "else:\n"
         "    print('No completed run found. Set DATA_ROOT and RUN_TRAINING=True, or run train.py first.')\n")
    markdown('## 9. Dataset and architecture evidence\n\nDescribe class imbalance, resolution, RGB correlations '
             'and possible background shortcuts. Check the actual input size and configuration of this run.')
    code("if RUN_DIR is not None:\n"
         "    audit_result = json.loads((RUN_DIR / 'audit.json').read_text())\n"
         "    print('Class counts:', audit_result['class_counts'])\n"
         "    print('Invalid:', len(audit_result['invalid_files']), 'duplicate copies:', len(audit_result['removed_duplicates']))\n"
         "    for name in ['class_distribution', 'training_examples', 'image_dimensions', 'rgb_correlation', 'architecture']:\n"
         "        display(DisplayImage(filename=str(RUN_DIR / 'figures' / f'{name}.png')))\n")
    markdown('## 10. Experimental results\n\nCompare both learning curves and validation metrics. '
             'Discuss the realised epoch budgets, overfitting or underfitting, and whether augmentation helped. '
             'Use the reported cross-validation standard deviation as descriptive variability, not a confidence interval.')
    code("if RUN_DIR is not None:\n"
         "    print((RUN_DIR / 'selection.json').read_text())\n"
         "    print((RUN_DIR / 'cross_validation.json').read_text())\n"
         "    for name in ['baseline', 'augmented']:\n"
         "        print(name, (RUN_DIR / name / 'config.json').read_text())\n"
         "        display(DisplayImage(filename=str(RUN_DIR / name / 'learning_curves.png')))\n")
    markdown('## 11. Held-out evaluation\n\nThese are the test results of the checkpoint chosen using validation. '
             'Do not select new hyperparameters after inspecting them. Report per-team precision, recall and F1 '
             'alongside overall and balanced accuracy; compare against the majority-class baseline.')
    code("if RUN_DIR is not None:\n"
         "    metrics = json.loads((RUN_DIR / 'test_metrics.json').read_text())\n"
         "    print(json.dumps({k:v for k,v in metrics.items() if k not in ['truth','predicted']}, indent=2))\n"
         "    display(DisplayImage(filename=str(RUN_DIR / 'figures' / 'test_confusion_matrix.png')))\n"
         "    errors = RUN_DIR / 'figures' / 'test_errors.png'\n"
         "    if errors.exists():\n"
         "        display(DisplayImage(filename=str(errors)))\n")
    markdown('## 12. Predict a photo\n\nRun `app.py` for the image picker, or set PHOTO_PATH above. '
             'Historical dataset labels are preserved; the classifier does not automatically support current teams or unseen liveries.')
    code("if PHOTO_PATH is not None:\n"
         "    if RUN_DIR is None:\n"
         "        raise ValueError('Train a model or select a completed RUN_DIR first.')\n"
         "    predictor = TeamPredictor(RUN_DIR / summary['checkpoint'])\n"
         "    print(predictor.predict(PHOTO_PATH))\n"
         "    display(DisplayImage(filename=str(PHOTO_PATH)))\n")
    markdown('## 13. Submission checks\n\nUse `docs/REPORT_CHECKLIST.md` and `docs/REFERENCES.md`. '
             'Write the PDF sections in your own voice, including difficulty and individual contributions. '
             'Complete `AI_USAGE.md` with all prompts. Add your own critical-code explanations and rehearse a walkthrough. '
             'Confirm the brief\'s Assign2 filename discrepancy, insert actual IDs, then restart and Run All in the submission environment.')
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python', 'version': '3.12.10'}}, 'nbformat': 4, 'nbformat_minor': 5}
    for i, cell in enumerate(cells):
        cell['id'] = f'f1-cell-{i:03d}'
    for name in ['F1_CNN.ipynb', 'F1_CNN.pynb.ipynb']:
        (ROOT / name).write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    print(f'Built {len(cells)} cells in both notebook filenames.')


if __name__ == '__main__':
    main()
