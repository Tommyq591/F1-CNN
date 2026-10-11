"""Audit images before making reproducible, stratified data splits."""

from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    team: str
    digest: str


class DatasetAudit:
    """Discover classes; exclude all copies of images with conflicting labels."""

    def __init__(self, root):
        self.root = Path(root)
        self.class_names = []
        self.records = []
        self.invalid = []
        self.duplicates = []
        self.conflicts = []
        self.skipped = []
        self.warnings = []

    def scan(self):
        if not self.root.is_dir():
            raise FileNotFoundError(f"Extract data first and point DATA_ROOT at the team folders: {self.root}")
        self.records, self.invalid, self.duplicates = [], [], []
        self.conflicts = []
        self.skipped, self.warnings = [], []
        conflicting_digests = set()
        folders = sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith('.'))
        if len(folders) < 2:
            raise ValueError("DATA_ROOT must directly contain at least two team folders (check ZIP nesting).")
        self.class_names = [p.name for p in folders]
        if {'train', 'test', 'val', 'validation'} & {n.lower() for n in self.class_names}:
            raise ValueError('Existing split folders detected. Set the root to team folders; do not mix an existing test set into training.')
        seen = {}
        for label, folder in enumerate(folders):
            for path in sorted(folder.rglob('*')):
                if not path.is_file() or path.name.startswith('.'):
                    continue
                try:
                    with Image.open(path) as source:
                        problems = []
                        if source.format != 'JPEG':
                            problems.append('non-JPEG')
                        if source.mode != 'RGB':
                            problems.append('non-RGB')
                        if path.suffix.lower() not in {'.jpg', '.jpeg'}:
                            problems.append('non-JPEG extension')
                        image = ImageOps.exif_transpose(source).convert('RGB')
                        # Hash decoded pixels so metadata-only copies are also detected.
                        digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                        if problems:
                            self.warnings.append({'path': str(path), 'issues': problems})
                except UnidentifiedImageError as error:
                    if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif', '.tif', '.tiff'}:
                        self.invalid.append((str(path), str(error)))
                    else:
                        self.skipped.append(str(path))
                    continue
                except (OSError, ValueError) as error:
                    self.invalid.append((str(path), str(error)))
                    continue
                if digest in seen:
                    previous = seen[digest]
                    if previous.label != label:
                        self.conflicts.append((str(previous.path), str(path)))
                        conflicting_digests.add(digest)
                        continue
                    self.duplicates.append((str(path), str(previous.path)))
                    continue
                record = ImageRecord(path, label, folder.name, digest)
                seen[digest] = record
                self.records.append(record)
        self.records = [r for r in self.records if r.digest not in conflicting_digests]
        counts = Counter(r.team for r in self.records)
        if any(counts[name] < 5 for name in self.class_names):
            raise ValueError(f"Need at least five unique readable images per team; found {dict(counts)}. Check folder layout.")
        return self

    def summary(self):
        return {'class_counts': dict(Counter(r.team for r in self.records)),
                'invalid_files': self.invalid, 'removed_duplicates': self.duplicates,
                'excluded_label_conflicts': self.conflicts, 'skipped_non_images': self.skipped,
                'format_warnings': self.warnings, 'photos_retained': len(self.records),
                'root': str(self.root.resolve())}


class F1ImageDataset(Dataset):
    def __init__(self, records, image_size=128, augment=False, cache_images=False):
        self.records = list(records)
        self.image_size = image_size
        self.cache_images = cache_images
        self.cache = {}
        steps = [transforms.Resize((image_size, image_size))]
        if augment:
            steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
        # Fixed scaling avoids estimating preprocessing statistics from test images.
        steps += [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if index in self.cache:
            image = self.cache[index].copy()
        else:
            with Image.open(record.path) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            # Cache resized RGB pixels, never random augmentations, to reduce disk work.
            image = transforms.functional.resize(image, [self.image_size, self.image_size])
            if self.cache_images:
                self.cache[index] = image.copy()
        return self.transform(image), record.label


class F1DataModule:
    """Keep test images separate; augment only training images."""

    def __init__(self, records, config):
        self.records = list(records)
        self.config = config

    def split(self, test_fraction=0.15, validation_fraction=0.15):
        if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1 - test_fraction:
            raise ValueError('Split fractions must be positive and sum to less than one.')
        train_val, test = train_test_split(
            self.records, test_size=test_fraction, random_state=self.config.seed,
            stratify=[r.label for r in self.records])
        train, val = train_test_split(
            train_val, test_size=validation_fraction / (1 - test_fraction),
            random_state=self.config.seed, stratify=[r.label for r in train_val])
        return {'train': train, 'validation': val, 'test': test}

    def folds(self, development_records, n_splits=3):
        """Pass only train + validation records here, never the held-out test set."""
        records = list(development_records)
        labels = [r.label for r in records]
        if min(Counter(labels).values()) < n_splits:
            raise ValueError('Each team needs at least n_splits development images.')
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.seed)
        for train, val in splitter.split(np.zeros(len(records)), labels):
            yield [records[i] for i in train], [records[i] for i in val]

    def loader(self, records, training=False):
        dataset = F1ImageDataset(records, self.config.image_size, training and self.config.augment,
                                 self.config.cache_images)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=training,
                          num_workers=self.config.num_workers,
                          generator=torch.Generator().manual_seed(self.config.seed))
