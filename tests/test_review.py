"""Review provenance, reference eligibility and local HTTP integration."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from eegmem.app import process_batch
from eegmem.compare import compare_batch
from eegmem.memory import create_db, save_batch
from eegmem.review import add_review, detail, records, reference_snapshot
from eegmem.workbench import make_server, workbench_database


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'data' / 'assistant.sqlite3'
        self.conn = create_db(self.path, 'fixed-model')
        for name in ['early', 'later']:
            trials = [{'trial': 0, 'c3': .1, 'c4': -.1, 'pred': 0, 'prob': .51, 'label': 1}]
            results, evidence = compare_batch([], trials)
            save_batch(self.conn, name, name, results, evidence)

    def tearDown(self):
        self.conn.close()
        self.temp.cleanup()

    def review(self, action, label=None, version=0):
        return add_review(self.conn, 'early', 0, action, label, 'Reviewer A', 'Checked source cue', version)

    def test_review_is_append_only_and_truth_is_not_auto_promoted(self):
        sources, _ = reference_snapshot(self.conn, 'reviewed')
        self.assertEqual(sources, [])
        original = self.conn.execute('SELECT results FROM batches WHERE name="early"').fetchone()[0]
        version = self.review('label', 1)
        sources, context = reference_snapshot(self.conn, 'reviewed')
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]['pred'], 1)
        self.assertEqual(sources[0]['model_pred'], 0)
        self.assertEqual(context['sources'][0]['review_id'], version)
        self.assertEqual(self.conn.execute('SELECT pred FROM trials WHERE batch="early"').fetchone()[0], 0)
        self.assertEqual(self.conn.execute('SELECT results FROM batches WHERE name="early"').fetchone()[0], original)
        version = self.review('exclude', version=version)
        self.assertEqual(reference_snapshot(self.conn, 'reviewed')[0], [])
        self.assertEqual([r['batch'] for r in reference_snapshot(self.conn)[0]], ['later'])
        self.review('reopen', version=version)
        self.assertEqual(reference_snapshot(self.conn, 'reviewed')[0], [])
        self.assertEqual(len(detail(self.conn, 'early', 0)['reviews']), 3)

    def test_stale_edit_and_empty_reason_rejected(self):
        self.review('label', 1)
        with self.assertRaisesRegex(ValueError, 'Record changed'):
            self.review('exclude')
        with self.assertRaisesRegex(ValueError, 'reason'):
            add_review(self.conn, 'early', 0, 'label', 0, 'Reviewer', ' ', 1)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0], 1)

    def test_neighbors_only_include_earlier_batches_and_respect_review_policy(self):
        self.assertEqual(detail(self.conn, 'early', 0)['neighbors'], [])
        self.assertEqual(len(detail(self.conn, 'later', 0)['neighbors']), 1)
        self.assertEqual(detail(self.conn, 'later', 0, 'reviewed')['neighbors'], [])
        self.review('label', 1)
        neighbor = detail(self.conn, 'later', 0, 'reviewed')['neighbors'][0]
        self.assertEqual(neighbor['batch'], 'early')
        self.assertEqual(neighbor['pred'], 1)

    def test_review_workspace_is_a_persistent_copy(self):
        workbench = workbench_database(self.root)
        copied = create_db(workbench)
        try:
            add_review(copied, 'early', 0, 'exclude', None, 'Demo reviewer', 'Demo', 0)
            self.assertEqual(len(records(copied)), 2)
        finally:
            copied.close()
        self.assertEqual(workbench_database(self.root), workbench)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM reviews').fetchone()[0], 0)

    def test_future_analysis_keeps_the_review_snapshot_and_does_not_fallback(self):
        path = self.root / 'new.npz'
        path.write_bytes(b'inference is isolated in this integration test')
        artifact = {'training_batches': [], 'training_digests': []}
        trial = {'trial': 0, 'c3': .2, 'c4': -.2, 'pred': 0, 'prob': .7, 'label': None}
        with patch('eegmem.app.analyze_batch', return_value=[trial]):
            report, _ = process_batch(self.conn, path, artifact, self.root / 'reports', 'reviewed')
        stored = self.conn.execute('SELECT context FROM analysis_context WHERE batch="new"').fetchone()[0]
        self.assertEqual(json.loads(stored)['sources'], [])
        before = report.read_text()
        self.review('label', 1)
        process_batch(self.conn, path, artifact, self.root / 'reports', 'reviewed')
        self.assertEqual(report.read_text(), before)
        with self.assertRaisesRegex(ValueError, 'different reference policy'):
            process_batch(self.conn, path, artifact, self.root / 'reports', 'predicted')
        path2 = self.root / 'next.npz'
        path2.write_bytes(b'another batch')
        with patch('eegmem.app.analyze_batch', return_value=[trial]):
            process_batch(self.conn, path2, artifact, self.root / 'reports', 'reviewed')
        context = json.loads(self.conn.execute('SELECT context FROM analysis_context WHERE batch="next"').fetchone()[0])
        self.assertEqual(context['sources'], [{'batch': 'early', 'trial': 0, 'review_id': 1}])

    def test_http_search_review_and_request_protection(self):
        server = make_server(self.path, 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base + '/') as response:
                self.assertIn(b'EEG Memory Workbench', response.read())
            with urlopen(base + '/api/records?q=early') as response:
                data = json.load(response)
            self.assertEqual(len(data['records']), 1)
            self.assertNotIn('label', data['records'][0])
            payload = json.dumps(dict(batch='early', trial=0, action='label', label=1,
                                      reviewer='Reviewer', reason='Source cue', expected_version=0)).encode()
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base + '/api/review', data=payload, headers={'Content-Type': 'application/json'}))
            self.assertEqual(caught.exception.code, 403)
            headers = {'Content-Type': 'application/json', 'X-Review-Token': data['token'], 'Origin': base}
            with urlopen(Request(base + '/api/review', data=payload, headers=headers)) as response:
                self.assertGreater(json.load(response)['review_event'], 0)
            headers['Origin'] = 'https://unrelated.example'
            with self.assertRaises(HTTPError) as caught:
                urlopen(Request(base + '/api/review', data=payload, headers=headers))
            self.assertEqual(caught.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
