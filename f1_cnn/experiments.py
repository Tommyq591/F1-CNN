"""One-factor experiments on the original split, with report-ready evidence.

This path never constructs a test loader and never changes runs/latest.json.
The original full-suite entry point remains available separately.
"""

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import statistics
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from sklearn.metrics import ConfusionMatrixDisplay

from .config import TrainingConfig
from .data import ImageRecord, F1DataModule
from .model import F1TeamCNN
from .pipeline import save_json
from .plots import plt, save_figure, learning_curves, evaluation_figures
from .training import Trainer, seed_everything


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preserve_baseline(baseline, evidence):
    """Copy all original results once, and detect later changes instead of overwriting."""
    baseline, evidence = Path(baseline), Path(evidence)
    snapshot = evidence / 'baseline'
    hashes = {p.relative_to(baseline).as_posix(): file_hash(p)
              for p in baseline.rglob('*') if p.is_file()}
    if snapshot.exists():
        expected = read_json(evidence / 'baseline_hashes.json')
        if hashes != expected or any(file_hash(snapshot / p) != h for p, h in expected.items()):
            raise ValueError('Baseline or its preserved copy changed; investigate before continuing.')
    else:
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copytree(baseline, snapshot)
        save_json(evidence / 'baseline_hashes.json', hashes)
        source = evidence / 'original_source'
        source.mkdir()
        for name in ['model.py', 'config.py', 'data.py', 'training.py', 'predict.py']:
            shutil.copy2(Path(__file__).parent / name, source / name)
    return snapshot


def fixed_development_split(baseline, names):
    """Reuse saved membership/order and verify development pixels; never decode test images."""
    manifest = read_json(Path(baseline) / 'split_manifest.json')
    seen_paths, seen_hashes = set(), set()
    for split in ['train', 'validation', 'test']:
        if not manifest[split]:
            raise ValueError(f'Empty {split} split.')
        for row in manifest[split]:
            path = str(Path(row['path']).resolve()).casefold()
            if path in seen_paths or row['digest'] in seen_hashes:
                raise ValueError('Duplicate image or overlapping original splits.')
            seen_paths.add(path)
            seen_hashes.add(row['digest'])
            if not 0 <= row['label'] < len(names) or names[row['label']] != row['team']:
                raise ValueError('Manifest label order differs from checkpoint.')
    result = {}
    for split in ['train', 'validation']:
        records = []
        for row in manifest[split]:
            record = ImageRecord(Path(row['path']), row['label'], row['team'], row['digest'])
            with Image.open(record.path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if digest != record.digest:
                raise ValueError(f'Development image changed: {record.path}')
            records.append(record)
            if len(records) % 500 == 0:
                print(f'Verified {split}: {len(records)}/{len(manifest[split])} images.', flush=True)
        result[split] = records
        print(f'Verified {split}: {len(records)} images; saved order retained.', flush=True)
    return result


def confusion_figure(metrics, names, title, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(np.asarray(metrics['confusion_matrix']), display_labels=names).plot(
        ax=ax, xticks_rotation=45, colorbar=False, cmap='Blues')
    ax.set_title(title)
    save_figure(fig, path)


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def comparison_report(baseline, run, evidence, result):
    """Export measured values from saved artifacts; keep historical test results separate."""
    names = read_json(baseline / 'summary.json')['class_names']
    protocol = read_json(run / 'protocol.json')
    setting = protocol['changed_setting']
    old_config, new_config = protocol['original_config'], protocol['updated_config']
    old_label = f'Original {old_config["image_size"]}px / {old_config["epochs"]}-epoch cap'
    if setting == 'epochs':
        new_label = f'{new_config["image_size"]}px / {new_config["epochs"]}-epoch cap'
        change = f'Only maximum epochs changed: **{old_config["epochs"]} → {new_config["epochs"]}**.'
        explanation = ('More training may allow continued learning beyond the old epoch cap. '
                       'Any gain applies to this seed and split; class imbalance and background shortcuts remain possible limitations.')
    else:
        old_size, new_size = old_config['image_size'], new_config['image_size']
        new_label = f'Resolution {new_size}px'
        change = f'Only image size changed: **{old_size} × {old_size} → {new_size} × {new_size}**.'
        explanation = ('More input detail may help distinguish liveries, but higher resolution alone '
                       'does not remedy label imbalance, background shortcuts or global pooling.')
    old_metrics = read_json(baseline / 'baseline/validation_metrics.json')
    new_metrics = read_json(run / 'validation_metrics.json')
    old_history = read_json(baseline / 'baseline/history.json')
    new_history = read_json(run / 'history.json')
    folder = evidence / run.name
    folder.mkdir(exist_ok=False)
    for name in ['config.json', 'history.json', 'validation_metrics.json', 'result.json',
                 'environment.json', 'protocol.json', 'benchmark.json']:
        shutil.copy2(run / name, folder / name)
    if (run / 'reproduction_check.json').exists():
        shutil.copy2(run / 'reproduction_check.json', folder / 'reproduction_check.json')
    for path in (run / 'figures').iterdir():
        shutil.copy2(path, folder / path.name)
    rows = []
    for label, metrics, history in [(old_label, old_metrics, old_history),
                                     (run.name, new_metrics, new_history)]:
        best = min(history, key=lambda r: r['validation_loss'])
        rows.append(dict(experiment=label, evaluation_split='validation',
                         accuracy=metrics['accuracy'], macro_f1=metrics['macro_f1'],
                         balanced_accuracy=metrics['balanced_accuracy'], loss=best['validation_loss'],
                         best_epoch=best['epoch'], epochs=len(history),
                         training_seconds=sum(r['seconds'] for r in history)))
    write_csv(folder / 'comparison.csv', rows)
    team_rows = []
    for name in names:
        old, new = old_metrics['classification_report'][name], new_metrics['classification_report'][name]
        team_rows.append(dict(team=name, support=new['support'], baseline_precision=old['precision'],
                             updated_precision=new['precision'], baseline_recall=old['recall'],
                             updated_recall=new['recall'], baseline_f1=old['f1-score'],
                             updated_f1=new['f1-score'], f1_change=new['f1-score']-old['f1-score']))
    write_csv(folder / 'per_team_validation.csv', team_rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, history, colour in [(old_label, old_history, '#235789'),
                                    (new_label, new_history, '#bd3d22')]:
        for split, style in [('train', '--'), ('validation', '-')]:
            for ax, metric in zip(axes, ['loss', 'accuracy']):
                ax.plot([r['epoch'] for r in history], [r[f'{split}_{metric}'] for r in history],
                        style, color=colour, label=f'{label}: {split}')
                ax.set(xlabel='Epoch', ylabel=metric.title())
                ax.legend(fontsize=8)
    fig.suptitle(f'{"Epoch-budget" if setting == "epochs" else "Resolution"}-only comparison | same original development split')
    save_figure(fig, folder / 'comparison_learning_curves.png')
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    ax.bar(x-.2, [r['baseline_f1'] for r in team_rows], .4, label=old_label)
    ax.bar(x+.2, [r['updated_f1'] for r in team_rows], .4, label=new_label)
    ax.set_xticks(x, [n.replace(' F1 car', '') for n in names], rotation=30, ha='right')
    ax.set(ylabel='F1', ylim=(0, 1), title='Per-team F1 on the same validation images')
    ax.legend()
    save_figure(fig, folder / 'per_team_validation_f1.png')
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x-.2, [r['baseline_recall'] for r in team_rows], .4, label=old_label)
    ax.bar(x+.2, [r['updated_recall'] for r in team_rows], .4, label=new_label)
    ax.set_xticks(x, [n.replace(' F1 car', '') for n in names], rotation=30, ha='right')
    ax.set(ylabel='Recall', ylim=(0, 1), title='Per-team recall on the same validation images')
    ax.legend()
    save_figure(fig, folder / 'per_team_validation_recall.png')
    confusion_figure(old_metrics, names, f'{old_label} | validation | epoch {rows[0]["best_epoch"]}',
                     folder / 'baseline_validation_confusion_matrix.png')
    confusion_figure(new_metrics, names, f'{new_label} | validation | epoch {result["best_epoch"]}',
                     folder / 'validation_confusion_matrix.png')
    improved = result['preferred_by_validation']
    lines = [f'## {run.name} — measured outcome', '',
             change + ' All other configuration values, '
             'split membership/order, seed, model and checkpoint-selection rule were retained.', '',
             f'| Validation metric | {old_label} | {new_label} |',
             '| --- | ---: | ---: |']
    for key, title in [('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1'),
                       ('balanced_accuracy', 'Balanced accuracy')]:
        lines.append(f'| {title} | {old_metrics[key]:.2%} | {new_metrics[key]:.2%} |')
    lines += [f'| Best validation loss | {rows[0]["loss"]:.4f} | {rows[1]["loss"]:.4f} |',
              f'| Best epoch / epochs run | {rows[0]["best_epoch"]} / {rows[0]["epochs"]} | '
              f'{rows[1]["best_epoch"]} / {rows[1]["epochs"]} |',
              f'| Training time (sum of epoch timers) | {rows[0]["training_seconds"]/60:.2f} min | '
              f'{rows[1]["training_seconds"]/60:.2f} min |', '',
              f'Experiment wall time including verification/export: {result["wall_seconds"]/60:.2f} min '
              '(excludes the separate preflight benchmark and final comparison rendering).', '',
              f'Validation macro F1 changed by {(new_metrics["macro_f1"]-old_metrics["macro_f1"])*100:+.2f} '
              f'percentage points; accuracy changed by {(new_metrics["accuracy"]-old_metrics["accuracy"])*100:+.2f} points. '
              f'By the predeclared validation macro-F1/accuracy rule, '
              f'{"the candidate is preferred" if improved else "the original baseline remains preferred"}. '
              'The application still defaults to the preserved original checkpoint.', '',
              f'**Measured:** This is one seed on {len(new_metrics["truth"])} validation images. The candidate has no test score; '
              'exceeding the historical 74.79% test accuracy has not been established. '
              'Epoch training metrics include dropout and changing weights, so their gap against '
              'evaluation-mode validation metrics is only an approximate overfitting indicator.', '',
              '**Possible explanation:** ' + explanation + ' '
              'A single run cannot isolate these mechanisms or establish statistical reliability.', '',
              f'Checkpoint: `{run.resolve() / "best.pt"}` (load manually with app.py).', '',
              f'![Learning curves](evidence/{run.name}/comparison_learning_curves.png)', '',
              f'![Per-team validation F1](evidence/{run.name}/per_team_validation_f1.png)', '',
              f'![Per-team validation recall](evidence/{run.name}/per_team_validation_recall.png)', '',
              f'![Validation confusion matrix](evidence/{run.name}/validation_confusion_matrix.png)', '',
              f'Tables: [comparison](evidence/{run.name}/comparison.csv), '
              f'[per-team metrics](evidence/{run.name}/per_team_validation.csv).', '']
    with (evidence.parent / 'EXPERIMENT_LOG.md').open('a', encoding='utf-8') as stream:
        stream.write('\n' + '\n'.join(lines))


def run_resolution_experiment(baseline, output, evidence, image_size=128, threads=4, benchmark=None):
    """Compatibility entry point for the original resolution-only workflow."""
    return run_single_experiment(baseline, output, evidence, 'image_size', image_size, threads, benchmark)


def run_epoch_experiment(baseline, output, evidence, epochs=24, threads=4, benchmark=None):
    return run_single_experiment(baseline, output, evidence, 'epochs', epochs, threads, benchmark)


def run_single_experiment(baseline, output, evidence, setting, value, threads=4, benchmark=None):
    baseline, output, evidence = Path(baseline), Path(output), Path(evidence)
    if output.exists() or (evidence / output.name).exists():
        raise FileExistsError('Choose a fresh run name; previous evidence is never replaced.')
    if threads < 1:
        raise ValueError('threads must be positive.')
    if setting not in {'image_size', 'epochs'}:
        raise ValueError('Only image_size or epochs can change in this workflow.')
    old_config = TrainingConfig(**read_json(baseline / 'baseline/config.json'))
    if value == getattr(old_config, setting):
        raise ValueError('Choose a different value for this one-factor experiment.')
    config = replace(old_config, **{setting: value})
    benchmark_data = read_json(benchmark) if benchmark else {'available': False}
    if setting == 'epochs' and benchmark is None:
        original_history = read_json(baseline / 'baseline/history.json')
        warm = statistics.median(r['seconds'] for r in original_history[1:] or original_history)
        estimate = original_history[0]['seconds'] + (config.epochs - 1) * warm
        benchmark_data = dict(method='Original observed first epoch plus warm median for remaining epochs',
                              first_epoch_seconds=original_history[0]['seconds'], warm_epoch_seconds=warm,
                              estimated_training_minutes=estimate / 60,
                              note='Estimate before training, excluding image hash verification and report export; early stopping may shorten the run.')
    torch.set_num_threads(threads)
    preserve_baseline(baseline, evidence)
    protocol = dict(baseline=str(baseline.resolve()), original_config=old_config.to_dict(),
                    updated_config=config.to_dict(), changed_setting=setting,
                    checkpoint_selection='minimum validation loss',
                    comparison_rule='validation macro F1, then validation accuracy',
                    manifest_sha256=file_hash(baseline / 'split_manifest.json'),
                    fresh_initialisation=True, test_evaluated=False, test_images_opened=False,
                    utc_started=datetime.now(timezone.utc).isoformat())
    # Preserve the declared plan even if verification or training is interrupted.
    with (evidence / f'{output.name}_plan.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(protocol=protocol, runtime_estimate=benchmark_data), stream, indent=2)
    print(json.dumps(dict(changed_setting=setting, original=getattr(old_config, setting),
                          updated=value, runtime_estimate=benchmark_data), indent=2), flush=True)
    started = time.perf_counter()
    names = read_json(baseline / 'summary.json')['class_names']
    splits = fixed_development_split(baseline, names)
    seed_everything(config.seed)
    data = F1DataModule(splits['train'] + splits['validation'], config)
    trainer = Trainer(F1TeamCNN(len(names), config.dropout), config, names, device='cpu')
    # Trainer owns directory creation, preserving its existing refusal to overwrite runs.
    history = trainer.fit(data.loader(splits['train'], True), data.loader(splits['validation']), output)
    metrics = trainer.evaluate(data.loader(splits['validation']))
    save_json(output / 'validation_metrics.json', metrics)
    save_json(output / 'config.json', config.to_dict())
    save_json(output / 'benchmark.json', benchmark_data)
    save_json(output / 'environment.json', dict(python=platform.python_version(), device='cpu',
              threads=threads, packages={name: importlib.metadata.version(name) for name in
              ['torch', 'torchvision', 'numpy', 'Pillow', 'scikit-learn', 'matplotlib']},
              source_hashes={name: file_hash(Path(__file__).parent / name) for name in
                             ['experiments.py', 'model.py', 'data.py', 'training.py', 'config.py']}))
    protocol['utc_completed'] = datetime.now(timezone.utc).isoformat()
    save_json(output / 'protocol.json', protocol)
    if setting == 'epochs':
        original_history = read_json(baseline / 'baseline/history.json')
        compared = min(len(original_history), len(history))
        differences = {key: max(abs(history[i][key] - original_history[i][key]) for i in range(compared))
                       for key in ['train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy']}
        save_json(output / 'reproduction_check.json', dict(epochs_compared=compared,
                  max_absolute_differences=differences, matches_within_1e_6=all(v <= 1e-6 for v in differences.values()),
                  note='Compare overlapping epochs to check reproduction of the original training trajectory; timings excluded.'))
    figures = output / 'figures'
    figures.mkdir()
    learning_curves(history, figures)
    evaluation_figures(metrics, names, splits['validation'], figures, prefix='validation')
    old = read_json(baseline / 'baseline/validation_metrics.json')
    result = dict(best_epoch=trainer.best_epoch, epochs=len(history),
                  training_seconds=sum(r['seconds'] for r in history),
                  wall_seconds=time.perf_counter()-started,
                  validation_accuracy=metrics['accuracy'], validation_macro_f1=metrics['macro_f1'],
                  validation_loss=min(r['validation_loss'] for r in history),
                  preferred_by_validation=(metrics['macro_f1'], metrics['accuracy']) >
                                          (old['macro_f1'], old['accuracy']),
                  test_evaluated=False, app_default_changed=False)
    save_json(output / 'result.json', result)
    comparison_report(baseline, output, evidence, result)
    preserve_baseline(baseline, evidence)  # Assert all original artifacts still match.
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('runs/local_original_32px'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, default=Path('docs/evidence'))
    change = parser.add_mutually_exclusive_group()
    change.add_argument('--image-size', type=int, help='Change only image size (default: 128 when neither change is specified)')
    change.add_argument('--epochs', type=int, help='Change only maximum epochs, retaining original image size')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--benchmark', type=Path)
    args = parser.parse_args()
    if args.epochs is not None:
        run_epoch_experiment(args.baseline, args.output, args.evidence, args.epochs, args.threads, args.benchmark)
    else:
        run_resolution_experiment(args.baseline, args.output, args.evidence,
                                  args.image_size if args.image_size is not None else 128, args.threads, args.benchmark)


if __name__ == '__main__':
    main()
