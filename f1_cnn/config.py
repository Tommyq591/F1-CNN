from dataclasses import dataclass, asdict
from pathlib import Path


def default_data_root():
    # Notebook builds define PROJECT_ROOT; package callers resolve from this file.
    project = globals().get('PROJECT_ROOT') or Path(__file__).resolve().parents[1]
    cleaned = project / 'data' / 'cleaned'
    if cleaned.is_dir() and any(cleaned.iterdir()):
        return cleaned
    return project / 'data' / 'raw' / 'Formula One Cars'


@dataclass(frozen=True)
class TrainingConfig:
    """Settings to record alongside every experiment."""

    image_size: int = 64
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.3
    seed: int = 42
    augment: bool = False
    patience: int = 6
    num_workers: int = 0
    cache_images: bool = True

    def __post_init__(self):
        if self.image_size < 16 or self.batch_size < 1 or self.epochs < 1:
            raise ValueError('image_size >= 16, batch_size >= 1 and epochs >= 1 are required.')
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError('Invalid dropout, learning rate or weight decay.')
        if self.patience < 1 or self.num_workers < 0:
            raise ValueError('patience >= 1 and num_workers >= 0 are required.')

    def to_dict(self):
        return asdict(self)
