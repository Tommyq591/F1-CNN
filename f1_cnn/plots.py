"""Save figures and tables from measured data; no invented results."""

from collections import Counter
import csv
from pathlib import Path
import os

# Keep font caches local so restricted Windows/PyCharm environments remain writable.
os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd() / '.cache' / 'matplotlib'))

import matplotlib
matplotlib.use('Agg')  # Saving plots works in PyCharm, notebooks and headless test runs.
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import ConfusionMatrixDisplay


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def dataset_figures(audit, splits, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = audit.class_names
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = np.arange(len(names))
    bottom = np.zeros(len(names))
    for split, records in splits.items():
        counts = Counter(r.team for r in records)
        values = np.array([counts[n] for n in names])
        ax.bar(positions, values, bottom=bottom, label=split)
        bottom += values
    ax.set_xticks(positions, names, rotation=35, ha='right')
    ax.set_ylabel('Images retained after audit')
    ax.legend()
    save_figure(fig, output / 'class_distribution.png')
    fig, axes = plt.subplots(len(names), 3, figsize=(10, 2.4 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        # Use training examples in exploratory figures, keeping the test set unseen.
        records = [r for r in splits['train'] if r.team == name][:3]
        for ax, record in zip(axes[row], records):
            with Image.open(record.path) as source:
                ax.imshow(ImageOps.exif_transpose(source).convert('RGB'))
            ax.set_title(name, fontsize=9)
            ax.axis('off')
    save_figure(fig, output / 'training_examples.png')
    dimensions, means = [], []
    for record in splits['train']:
        with Image.open(record.path) as source:
            img = ImageOps.exif_transpose(source).convert('RGB')
            dimensions.append(img.size)
            means.append(np.asarray(img.resize((16, 16))).mean(axis=(0, 1)) / 255)
    dims = np.asarray(dimensions)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(dims[:, 0], dims[:, 1], s=8, alpha=.4)
    axes[0].set(xlabel='Original width', ylabel='Original height', title='Training image sizes')
    axes[1].hist(dims[:, 0] / dims[:, 1], bins=30)
    axes[1].set(xlabel='Width / height', ylabel='Count', title='Training aspect ratios')
    save_figure(fig, output / 'image_dimensions.png')
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = np.corrcoef(np.asarray(means), rowvar=False)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(3), ['Red', 'Green', 'Blue'])
    ax.set_yticks(range(3), ['Red', 'Green', 'Blue'])
    ax.set_title('Correlation of mean RGB values (training only)')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center')
    fig.colorbar(im, ax=ax)
    save_figure(fig, output / 'rgb_correlation.png')


def architecture_figure(model, image_size, num_classes, output):
    stages = [f'RGB image\n3 x {image_size} x {image_size}',
              f'Conv 3x3 + ReLU (twice)\n32 channels; MaxPool 2x2\n32 x {image_size//2} x {image_size//2}',
              f'Conv 3x3 + ReLU (twice)\n64 channels; MaxPool 2x2\n64 x {image_size//4} x {image_size//4}',
              f'Conv 3x3 + ReLU (twice)\n128 channels; MaxPool 2x2\n128 x {image_size//8} x {image_size//8}',
              'Global average pool + flatten\n128 features',
              f'Dropout + Linear\n{num_classes} logits (one per team)']
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    for i, label in enumerate(stages):
        y = .92 - i * .16
        ax.text(.5, y, label, ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.6', fc='#edf3fb', ec='#345983'))
        if i < len(stages)-1:
            ax.annotate('', xy=(.5, y-.10), xytext=(.5, y-.055), arrowprops={'arrowstyle': '->'})
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ax.set_title(f'Explicit VGG-inspired CNN | {count:,} trainable parameters', pad=20)
    save_figure(fig, Path(output) / 'architecture.png')


def learning_curves(history, output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for prefix, label in [('train', 'Training'), ('validation', 'Validation')]:
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            ax.plot([r['epoch'] for r in history], [r[f'{prefix}_{metric}'] for r in history], label=label)
            ax.set(xlabel='Epoch', ylabel=metric.title())
            ax.legend()
    save_figure(fig, Path(output) / 'learning_curves.png')


def evaluation_figures(metrics, names, records, output, prefix='test'):
    output = Path(output)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(np.array(metrics['confusion_matrix']), display_labels=names).plot(
        ax=ax, xticks_rotation=45, colorbar=False, cmap='Blues')
    save_figure(fig, output / f'{prefix}_confusion_matrix.png')
    with (output / f'{prefix}_predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['path', 'actual_team', 'predicted_team', 'correct'])
        for record, actual, predicted in zip(records, metrics['truth'], metrics['predicted']):
            writer.writerow([str(record.path), names[actual], names[predicted], actual == predicted])
    errors = [(r, p) for r, a, p in zip(records, metrics['truth'], metrics['predicted']) if a != p][:12]
    if errors:
        fig, axes = plt.subplots((len(errors)+2)//3, 3, figsize=(11, 3*((len(errors)+2)//3)), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, (record, predicted) in zip(axes.flat, errors):
            with Image.open(record.path) as source:
                ax.imshow(ImageOps.exif_transpose(source).convert('RGB'))
            ax.set_title(f'True: {record.team}\nPredicted: {names[predicted]}', fontsize=9)
        save_figure(fig, output / f'{prefix}_errors.png')
