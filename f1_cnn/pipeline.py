"""Reproducible baseline, augmentation, development CV and held-out evaluation."""

from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import torch

from .config import TrainingConfig, default_data_root
from .data import DatasetAudit, F1DataModule
from .model import F1TeamCNN
from .training import Trainer, seed_everything
from .plots import dataset_figures, architecture_figure, learning_curves, evaluation_figures


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def train_experiment(records, train, validation, config, names, output, device=None):
    # Reset before construction so both experiments start from the same initial weights.
    seed_everything(config.seed)
    data = F1DataModule(records, config)
    trainer = Trainer(F1TeamCNN(len(names), config.dropout), config, names, device=device)
    history = trainer.fit(data.loader(train, True), data.loader(validation), output)
    metrics = trainer.evaluate(data.loader(validation))
    save_json(Path(output) / 'validation_metrics.json', metrics)
    save_json(Path(output) / 'config.json', config.to_dict())
    learning_curves(history, output)
    return trainer, metrics


def run_suite(data_root, output, config=None, folds=3, device=None):
    config = config or TrainingConfig()
    if folds < 2:
        raise ValueError('At least two development folds are required; the default is three.')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a new output directory to preserve the previous run: {output}')
    audit = DatasetAudit(data_root).scan()
    data = F1DataModule(audit.records, config)
    splits = data.split()
    # Validate all folds before starting any potentially lengthy experiment.
    development = splits['train'] + splits['validation']
    development_folds = list(data.folds(development, folds))
    output.mkdir(parents=True)
    save_json(output / 'audit.json', audit.summary())
    save_json(output / 'split_manifest.json', {
        split: [{'path': r.path.resolve().as_posix(), 'team': r.team, 'label': r.label, 'digest': r.digest}
                for r in records] for split, records in splits.items()})
    versions = {name: importlib.metadata.version(name) for name in
                ['torch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn', 'matplotlib']}
    save_json(output / 'environment.json', {'python': platform.python_version(), 'packages': versions,
        'device': device or ('cuda' if torch.cuda.is_available() else 'cpu'),
        'utc_started': datetime.now(timezone.utc).isoformat(), 'config': config.to_dict()})
    print(f'Audit retained {len(audit.records)} images in {len(audit.class_names)} teams.', flush=True)
    print({name: len(records) for name, records in splits.items()}, flush=True)
    figures = output / 'figures'
    dataset_figures(audit, splits, figures)
    seed_everything(config.seed)
    architecture_figure(F1TeamCNN(len(audit.class_names), config.dropout), config.image_size,
                        len(audit.class_names), figures)
    candidates = {}
    for name, augment in [('baseline', False), ('augmented', True)]:
        print(f'Experiment: {name}', flush=True)
        experiment_config = replace(config, augment=augment)
        trainer, metrics = train_experiment(audit.records, splits['train'], splits['validation'],
                                            experiment_config, audit.class_names, output / name, device)
        candidates[name] = {'trainer': trainer, 'metrics': metrics, 'config': experiment_config}
    # The test set has not been used: selection uses macro F1 on validation only.
    selected = max(candidates, key=lambda name: (candidates[name]['metrics']['macro_f1'],
                                                candidates[name]['metrics']['accuracy']))
    selected_config = candidates[selected]['config']
    save_json(output / 'selection.json', {'selected': selected, 'criterion': 'validation macro F1, then accuracy',
        'experiments': {n: {k: v['metrics'][k] for k in ['accuracy', 'balanced_accuracy', 'macro_f1']}
                        for n, v in candidates.items()}})
    scores = []
    for fold, (train, validation) in enumerate(development_folds, 1):
        print(f'Development cross-validation: {fold}/{folds}, configuration {selected}', flush=True)
        _, metrics = train_experiment(audit.records, train, validation, selected_config,
                                      audit.class_names, output / f'cv_fold_{fold}', device)
        scores.append({k: metrics[k] for k in ['accuracy', 'balanced_accuracy', 'macro_f1']})
        save_json(output / f'cv_fold_{fold}' / 'manifest.json', {
            'train': [r.digest for r in train], 'validation': [r.digest for r in validation]})
    cv = {'configuration': selected, 'folds': scores,
          'mean': {k: float(np.mean([r[k] for r in scores])) for k in scores[0]},
          'std': {k: float(np.std([r[k] for r in scores], ddof=1)) for k in scores[0]},
          'note': 'Development CV after configuration selection; not nested CV or an unbiased final estimate.'}
    save_json(output / 'cross_validation.json', cv)
    # Evaluate the already selected checkpoint once; do not tune against this result.
    best = candidates[selected]['trainer']
    test = best.evaluate(F1DataModule(audit.records, selected_config).loader(splits['test']))
    save_json(output / 'test_metrics.json', test)
    evaluation_figures(test, audit.class_names, splits['test'], figures)
    majority = Counter(r.label for r in splits['train']).most_common(1)[0][0]
    majority_accuracy = sum(r.label == majority for r in splits['test']) / len(splits['test'])
    summary = {'selected_experiment': selected, 'checkpoint': f'{selected}/best.pt',
               'class_names': audit.class_names, 'data_root': str(Path(data_root).resolve()),
               'images': len(audit.records), 'split_sizes': {k: len(v) for k, v in splits.items()},
               'test_accuracy': test['accuracy'], 'test_macro_f1': test['macro_f1'],
               'test_balanced_accuracy': test['balanced_accuracy'],
               'majority_baseline_accuracy': majority_accuracy, 'cv': cv,
               'best_epoch': best.best_epoch, 'completed_utc': datetime.now(timezone.utc).isoformat()}
    save_json(output / 'summary.json', summary)
    # A latest pointer is written only after every experiment and final evaluation succeeds.
    save_json(output.parent / 'latest.json', {'run': str(output.resolve()),
                                            'checkpoint': str((output / summary['checkpoint']).resolve())})
    print(json.dumps(summary, indent=2), flush=True)
    return summary
