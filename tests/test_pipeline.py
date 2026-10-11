import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from f1_cnn.config import TrainingConfig
from f1_cnn.data import DatasetAudit, F1DataModule
from f1_cnn.model import F1TeamCNN
from f1_cnn.training import Trainer, seed_everything
from f1_cnn.predict import TeamPredictor


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # Keep fixtures inside the project so restricted Windows runners can write them.
        temp_root = Path(__file__).resolve().parents[1] / '.cache' / 'tests'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(42)
        for team in ['TeamA', 'TeamB']:
            folder = self.root / 'data' / team
            folder.mkdir(parents=True)
            for index in range(20):
                Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(folder / f'{index}.png')

    def tearDown(self):
        self.temp.cleanup()

    def test_audit_and_disjoint_reproducible_splits(self):
        original = self.root / 'data/TeamA/0.png'
        (original.parent / 'copy.png').write_bytes(original.read_bytes())
        (original.parent / 'bad.jpg').write_bytes(b'not an image')
        audit = DatasetAudit(self.root / 'data').scan()
        self.assertEqual(len(audit.records), 40)
        self.assertEqual(len(audit.duplicates), 1)
        self.assertEqual(len(audit.invalid), 1)
        data = F1DataModule(audit.records, TrainingConfig())
        splits = data.split()
        self.assertEqual(splits, data.split())
        sets = [set(r.digest for r in records) for records in splits.values()]
        self.assertEqual(len(set.union(*sets)), 40)
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for records in splits.values():
            self.assertEqual({r.label for r in records}, {0, 1})
        for train, val in data.folds(splits['train'] + splits['validation']):
            self.assertFalse({r.digest for r in train} & {r.digest for r in val})
            self.assertFalse({r.digest for r in train + val} & sets[2])

    def test_conflicting_duplicate_excluded_from_both_classes(self):
        source = self.root / 'data/TeamA/0.png'
        (self.root / 'data/TeamB/copy.png').write_bytes(source.read_bytes())
        audit = DatasetAudit(self.root / 'data').scan()
        self.assertEqual(len(audit.conflicts), 1)
        self.assertEqual(len(audit.records), 39)
        self.assertNotIn(source, [r.path for r in audit.records])
        self.assertNotIn(self.root / 'data/TeamB/copy.png', [r.path for r in audit.records])

    def test_training_checkpoint_evaluation_and_prediction(self):
        torch.set_num_threads(1)
        config = TrainingConfig(image_size=16, batch_size=8, epochs=1)
        seed_everything(config.seed)
        audit = DatasetAudit(self.root / 'data').scan()
        data = F1DataModule(audit.records, config)
        splits = data.split()
        model = F1TeamCNN(2, config.dropout)
        before = model.classifier[-1].weight.detach().clone()
        trainer = Trainer(model, config, audit.class_names, device='cpu')
        trainer.fit(data.loader(splits['train'], True), data.loader(splits['validation']), self.root / 'run')
        self.assertFalse(torch.equal(before, model.classifier[-1].weight))
        metrics = trainer.evaluate(data.loader(splits['test']))
        self.assertEqual(sum(map(sum, metrics['confusion_matrix'])), len(splits['test']))
        predictor = TeamPredictor(self.root / 'run/best.pt')
        results = predictor.predict(splits['test'][0].path)
        self.assertEqual({r['team'] for r in results}, set(audit.class_names))
        self.assertAlmostEqual(sum(r['score'] for r in results), 1.0, places=5)
        model.eval()
        sample, _ = data.loader(splits['test']).dataset[0]
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(sample[None]), predictor.model(sample[None])))

    def test_unknown_extension_is_decoded_and_existing_split_layout_rejected(self):
        source = self.root / 'data/TeamA/0.png'
        source.rename(source.with_suffix('.123456'))
        audit = DatasetAudit(self.root / 'data').scan()
        self.assertEqual(len(audit.records), 40)
        (self.root / 'existing/train').mkdir(parents=True)
        (self.root / 'existing/test').mkdir()
        with self.assertRaisesRegex(ValueError, 'Existing split'):
            DatasetAudit(self.root / 'existing').scan()

    def test_complete_suite_selects_and_exports_real_metrics(self):
        from f1_cnn.pipeline import run_suite
        import json
        torch.set_num_threads(1)
        output = self.root / 'suite'
        summary = run_suite(self.root / 'data', output,
                            TrainingConfig(image_size=16, epochs=1, batch_size=8), folds=2, device='cpu')
        self.assertEqual(summary['images'], 40)
        self.assertTrue((output / summary['checkpoint']).exists())
        self.assertTrue((output / 'figures/test_confusion_matrix.png').exists())
        manifest = json.loads((output / 'split_manifest.json').read_text())
        test_hashes = {r['digest'] for r in manifest['test']}
        for fold in [1, 2]:
            members = json.loads((output / f'cv_fold_{fold}/manifest.json').read_text())
            self.assertFalse(test_hashes & set(members['train'] + members['validation']))
        with self.assertRaises(FileExistsError):
            run_suite(self.root / 'data', output)


if __name__ == '__main__':
    unittest.main()
