"""Regression coverage for persistence failures and signal edge cases."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eegmem.analyze import analyze_batch, batch_features, train_model
from eegmem.app import file_digest, process_batch
from eegmem.compare import compare_batch
from eegmem.memory import create_db, load_all, save_batch
from eegmem.report import render_report


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.conn = create_db(self.root / 'memory.sqlite3', 'model-a')
        rng = np.random.default_rng(5)
        self.data = rng.normal(size=(8, 3, 161)) * 1e-6
        self.labels = np.array([0, 1] * 4)
        self.train = self.write_batch('train', self.data, self.labels)
        self.artifact = train_model([self.train])
        self.artifact['training_digests'] = [file_digest(self.train)]
        self.incoming = self.write_batch('incoming', self.data * 1.1)

    def tearDown(self):
        self.conn.close()
        self.temp.cleanup()

    def write_batch(self, name, data, labels=None, channels=None):
        path = self.root / f'{name}.npz'
        payload = dict(data=data, ch_names=channels or ['C3', 'C4', 'Cz'], sfreq=160.)
        if labels is not None:
            payload['labels'] = labels
        np.savez(path, **payload)
        return path

    def test_unlabeled_signal_report_and_persistent_idempotency(self):
        report, added = process_batch(self.conn, self.incoming, self.artifact, self.root / 'reports')
        self.assertTrue(added)
        self.assertIn('accuracy cannot be calculated', report.read_text())
        self.assertEqual(len(load_all(self.conn)), 8)
        report.unlink()
        self.conn.close()
        self.conn = create_db(self.root / 'memory.sqlite3', 'model-a')
        report, added = process_batch(self.conn, self.incoming, self.artifact, self.root / 'reports')
        self.assertFalse(added)
        self.assertTrue(report.exists())
        self.assertEqual(len(load_all(self.conn)), 8)

    def test_training_and_changed_batches_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Training'):
            process_batch(self.conn, self.train, self.artifact, self.root)
        process_batch(self.conn, self.incoming, self.artifact, self.root)
        self.write_batch('incoming', self.data * 2)
        with self.assertRaisesRegex(ValueError, 'contents changed'):
            process_batch(self.conn, self.incoming, self.artifact, self.root)

    def test_renamed_duplicate_is_rejected(self):
        process_batch(self.conn, self.incoming, self.artifact, self.root)
        renamed = self.root / 'renamed.npz'
        renamed.write_bytes(self.incoming.read_bytes())
        with self.assertRaisesRegex(ValueError, 'already recorded'):
            process_batch(self.conn, renamed, self.artifact, self.root)

    def test_model_mixing_rejected(self):
        with self.assertRaisesRegex(ValueError, 'different model'):
            create_db(self.root / 'memory.sqlite3', 'model-b')

    def test_batch_insert_rolls_back(self):
        results, evidence = compare_batch([], analyze_batch(self.incoming, self.artifact))
        results[1]['trial'] = results[0]['trial']
        with self.assertRaises(sqlite3.IntegrityError):
            save_batch(self.conn, 'bad', 'digest', results, evidence)
        self.assertEqual(len(load_all(self.conn)), 0)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM batches').fetchone()[0], 0)

    def test_failed_report_can_be_recovered(self):
        with patch('eegmem.app.write_report', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                process_batch(self.conn, self.incoming, self.artifact, self.root)
        report, added = process_batch(self.conn, self.incoming, self.artifact, self.root)
        self.assertFalse(added)
        self.assertTrue(report.exists())
        self.assertEqual(len(load_all(self.conn)), 8)

    def test_zero_variance_and_low_confidence_are_not_hidden(self):
        path = self.write_batch('zero', np.zeros_like(self.data))
        features, _, _, _ = batch_features(path)
        self.assertTrue(np.isfinite(features).all())
        rows = [{'c3': 0., 'c4': 0., 'pred': 0} for _ in range(6)]
        trial = {'trial': 0, 'c3': 0., 'c4': 0., 'pred': 0, 'prob': .51, 'label': None}
        results, evidence = compare_batch(rows, [trial])
        self.assertEqual(results[0]['verdict'], 'Insufficient reference')
        self.assertTrue(results[0]['low_confidence'])
        self.assertIn('Low model confidence', render_report('zero', results, evidence))

    def test_invalid_signal_and_channel_order_rejected(self):
        bad = self.data.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            batch_features(self.write_batch('nan', bad))
        path = self.write_batch('reordered', self.data, channels=['C4', 'C3', 'Cz'])
        with self.assertRaisesRegex(ValueError, 'channel order'):
            analyze_batch(path, self.artifact)
        with self.assertRaisesRegex(ValueError, 'two trials'):
            batch_features(self.write_batch('single', self.data[:1]))

    def test_all_trials_compare_against_pre_batch_history(self):
        process_batch(self.conn, self.incoming, self.artifact, self.root)
        stored = self.conn.execute('SELECT results FROM batches').fetchone()[0]
        self.assertTrue(all(r['verdict'] == 'Cold start' for r in json.loads(stored)))


if __name__ == '__main__':
    unittest.main()
