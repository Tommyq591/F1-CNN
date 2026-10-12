"""Regression checks for holdout isolation, preservation and checkpoint compatibility."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from f1_cnn.config import TrainingConfig
from f1_cnn.data import DatasetAudit, F1DataModule
from f1_cnn.experiments import (file_hash, fixed_development_split,
                                preserve_baseline, run_resolution_experiment)
from f1_cnn.model import F1TeamCNN
from f1_cnn.pipeline import save_json
from f1_cnn.predict import TeamPredictor
from f1_cnn.training import Trainer, seed_everything


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1] / '.cache/tests'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(7)
        for name in ['A', 'B']:
            folder = self.root / 'data' / name
            folder.mkdir(parents=True)
            for i in range(12):
                Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(folder / f'{i}.png')
        self.audit = DatasetAudit(self.root / 'data').scan()
        self.config = TrainingConfig(image_size=16, epochs=1, batch_size=8)
        self.data = F1DataModule(self.audit.records, self.config)
        self.splits = self.data.split()
        self.baseline = self.root / 'original'
        self.baseline.mkdir()
        self.manifest = {split: [dict(path=str(r.path), team=r.team, label=r.label, digest=r.digest)
                                for r in records] for split, records in self.splits.items()}
        save_json(self.baseline / 'split_manifest.json', self.manifest)

    def tearDown(self):
        self.temp.cleanup()

    def test_reject_overlap_changed_pixels_and_modified_snapshot(self):
        fixed_development_split(self.baseline, self.audit.class_names)
        bad = dict(self.manifest, test=[self.manifest['train'][0]])
        save_json(self.baseline / 'split_manifest.json', bad)
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            fixed_development_split(self.baseline, self.audit.class_names)
        save_json(self.baseline / 'split_manifest.json', self.manifest)
        Image.new('RGB', (16, 16), 'red').save(self.splits['train'][0].path)
        with self.assertRaisesRegex(ValueError, 'changed'):
            fixed_development_split(self.baseline, self.audit.class_names)
        evidence = self.root / 'docs/evidence'
        preserve_baseline(self.baseline, evidence)
        (evidence / 'baseline/split_manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'preserved copy changed'):
            preserve_baseline(self.baseline, evidence)

    def test_one_run_never_opens_test_images_or_replaces_original(self):
        torch.set_num_threads(1)
        seed_everything(42)
        trainer = Trainer(F1TeamCNN(2), self.config, self.audit.class_names, 'cpu')
        trainer.fit(self.data.loader(self.splits['train'], True),
                    self.data.loader(self.splits['validation']), self.baseline / 'baseline')
        save_json(self.baseline / 'baseline/config.json', self.config.to_dict())
        save_json(self.baseline / 'baseline/validation_metrics.json',
                  trainer.evaluate(self.data.loader(self.splits['validation'])))
        save_json(self.baseline / 'summary.json', {'class_names': self.audit.class_names})
        old_hash = file_hash(self.baseline / 'baseline/best.pt')
        test_paths = {r.path.resolve() for r in self.splits['test']}
        original_open = Image.open

        def guarded_open(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() in test_paths:
                raise AssertionError('The candidate opened a test image.')
            return original_open(path, *args, **kwargs)

        output, evidence = self.root / 'runs/resolution', self.root / 'docs/evidence'
        pointer = output.parent / 'latest.json'
        pointer.parent.mkdir()
        pointer.write_text('original pointer')
        with patch('PIL.Image.open', side_effect=guarded_open):
            result = run_resolution_experiment(self.baseline, output, evidence, image_size=32, threads=1)
        self.assertFalse(result['test_evaluated'])
        self.assertEqual(file_hash(self.baseline / 'baseline/best.pt'), old_hash)
        self.assertEqual(pointer.read_text(), 'original pointer')
        config = json.loads((output / 'config.json').read_text())
        self.assertEqual({k for k, v in config.items() if v != self.config.to_dict()[k]}, {'image_size'})
        self.assertTrue((evidence / 'resolution/comparison.csv').exists())
        self.assertTrue((evidence / 'resolution/validation_confusion_matrix.png').exists())
        self.assertEqual(len(TeamPredictor(output / 'best.pt').predict(self.splits['validation'][0].path)), 2)
        with self.assertRaises(FileExistsError):
            run_resolution_experiment(self.baseline, output, evidence, image_size=32, threads=1)


if __name__ == '__main__':
    unittest.main()
