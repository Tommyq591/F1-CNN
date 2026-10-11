"""Command line entry point: python -m f1_cnn --help."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from .config import TrainingConfig, default_data_root


def main():
    parser = argparse.ArgumentParser(description='Train and use an F1 team image classifier.')
    commands = parser.add_subparsers(dest='command', required=True)
    audit = commands.add_parser('audit', help='Read-only check of team folders and duplicate pixels')
    audit.add_argument('--data', type=Path, default=default_data_root())
    train = commands.add_parser('train', help='Baseline, augmentation, development CV and final test')
    train.add_argument('--data', type=Path, default=default_data_root())
    train.add_argument('--output', type=Path, default=Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S'))
    train.add_argument('--epochs', type=int, default=20)
    train.add_argument('--image-size', type=int, default=64)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--learning-rate', type=float, default=.001)
    train.add_argument('--dropout', type=float, default=.3)
    train.add_argument('--weight-decay', type=float, default=.0001)
    train.add_argument('--seed', type=int, default=42)
    train.add_argument('--patience', type=int, default=6)
    train.add_argument('--folds', type=int, default=3)
    train.add_argument('--threads', type=int, default=4)
    train.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    predict = commands.add_parser('predict', help='Print ranked scores for a new image')
    predict.add_argument('image', type=Path)
    predict.add_argument('--checkpoint', type=Path)
    predict.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()
    # Keep --help fast; only load ML libraries after parsing a real operation.
    import torch
    from .data import DatasetAudit
    from .pipeline import run_suite
    from .predict import TeamPredictor
    try:
        if args.command == 'audit':
            print(json.dumps(DatasetAudit(args.data).scan().summary(), indent=2))
        elif args.command == 'train':
            if args.threads < 1:
                raise ValueError('--threads must be positive.')
            torch.set_num_threads(args.threads)
            config = TrainingConfig(image_size=args.image_size, batch_size=args.batch_size,
                epochs=args.epochs, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                dropout=args.dropout, seed=args.seed, patience=args.patience)
            run_suite(args.data, args.output, config, args.folds, args.device)
        else:
            torch.set_num_threads(2)
            checkpoint = args.checkpoint
            if checkpoint is None:
                pointer = Path(__file__).resolve().parents[1] / 'runs' / 'latest.json'
                if not pointer.exists():
                    raise FileNotFoundError('Train a model first, or supply --checkpoint PATH.')
                checkpoint = Path(json.loads(pointer.read_text(encoding='utf-8'))['checkpoint'])
            print(json.dumps(TeamPredictor(checkpoint).predict(args.image, args.top_k), indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
