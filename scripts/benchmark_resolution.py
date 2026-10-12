"""Estimate the single resolution experiment using disposable synthetic batches."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from f1_cnn.config import TrainingConfig
from f1_cnn.model import F1TeamCNN
from f1_cnn.training import Trainer, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'runs/local_original_32px')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new benchmark output; existing evidence is preserved.')
    if args.threads < 1:
        parser.error('threads must be positive.')
    config = replace(TrainingConfig(**json.loads((args.baseline / 'baseline/config.json').read_text())),
                     image_size=args.image_size)
    summary = json.loads((args.baseline / 'summary.json').read_text())
    torch.set_num_threads(args.threads)
    seed_everything(config.seed)
    names = summary['class_names']
    trainer = Trainer(F1TeamCNN(len(names), config.dropout), config, names, 'cpu')
    x = torch.randn(config.batch_size, 3, config.image_size, config.image_size)
    y = torch.arange(config.batch_size) % len(names)
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        trainer._epoch([(x, y)], True)
        timings.append(time.perf_counter() - start)
    start = time.perf_counter()
    trainer._epoch([(x, y)], False)
    validation_seconds = time.perf_counter() - start
    batches = {k: (v + config.batch_size - 1) // config.batch_size
               for k, v in summary['split_sizes'].items() if k != 'test'}
    estimate = batches['train'] * statistics.median(timings[1:]) + batches['validation'] * validation_seconds
    result = dict(threads=args.threads, device='cpu', image_size=config.image_size,
                  batch_size=config.batch_size, training_batch_seconds=timings,
                  validation_batch_seconds=validation_seconds, warm_epoch_estimate_seconds=estimate,
                  full_budget_estimate_minutes=estimate * config.epochs / 60,
                  note='Synthetic batches; discarded weights. Excludes image verification/decoding, '
                       'caching and export; sustained runtime may differ.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
