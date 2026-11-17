"""Synthetic checks using the shared temporal fixture and a stand-in encoder."""
import csv
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import run_embeddings as runner

spec = importlib.util.spec_from_file_location('starter_fixture', runner.STARTER / 'test_starter.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class ScorerTests(unittest.TestCase):
    def test_normalizes_assets_before_equal_weight_centroid(self):
        vectors = [[10, 0], [0, 1], [3, 4], [-1, 0]]
        actual = runner.centroid_scores(vectors, [[0, 1], [0, 3], []])
        np.testing.assert_allclose(actual[0], np.asarray([1, 1, 1.4, -1]) / math.sqrt(2))
        np.testing.assert_array_equal(actual[1:], np.zeros((2, 4)))
        for invalid in ([[math.nan, 1]], [[0, 0]]):
            with self.assertRaises(ValueError):
                runner.centroid_scores(invalid, [[0]])

    def test_same_value_in_another_field_is_a_different_dimension(self):
        rows = [('Stock', 'Large', 'x', 'y'), ('Large', 'Stock', 'x', 'y')]
        vectors = runner.one_hot_matrix(rows)
        np.testing.assert_allclose(runner.centroid_scores(vectors, [[0]]), [[1, 0.5]])
        fields = runner.feature_values({'assetCategory': ' Stock ', 'sector': 'Not_Available'})
        self.assertEqual(fields, ('Stock', '[MISSING]', '[MISSING]', '[MISSING]'))
        self.assertEqual(len(runner.description(('a\nSubcategory: b', '', '', '')).splitlines()), 4)

    def test_ties_use_popularity_then_isin_with_no_score_rounding(self):
        ids = ['C', 'B', 'D', 'A']
        pop = {'A': 1, 'B': 9, 'C': 9, 'D': 0}
        self.assertEqual(runner.top10(ids, [0.5, 0.5, 0.50000001, 0.5], pop),
                         ['D', 'B', 'C', 'A'])
        with self.assertRaises(ValueError):
            runner.top10(ids, [0.5, 0.5, math.nan, 0.5], pop)

    def test_ivan_reference_rejects_wrong_scores(self):
        reference = runner.IVAN / 'results/table_results.csv'
        with reference.open(newline='') as stream:
            metrics = {row['method']: {key: float(value) for key, value in row.items()
                                       if key != 'method'} for row in csv.DictReader(stream)}
        runner.verify_ivan(metrics, reference)
        metrics['cbf']['ndcg_at_10_nonempty'] += 1e-10
        with self.assertRaisesRegex(ValueError, 'cbf.ndcg_at_10_nonempty'):
            runner.verify_ivan(metrics, reference)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        fixture.write_fixture(self.data)
        self.args = SimpleNamespace(command='evaluate', starter_dir=runner.STARTER,
                                    data_dir=self.data, output_dir=self.root / 'evaluation',
                                    cutoff=runner.CUTOFF, horizon_days=runner.HORIZON_DAYS,
                                    max_users=runner.MAX_USERS, candidate_count=50)
        self.bundle = runner.data_adapter.load(self.args)
        self.encoder_inputs = []

    def fake_encode(self, texts):
        self.assertEqual(len(texts), len(set(texts)))
        self.assertTrue(all(len(text.splitlines()) == 4 for text in texts))
        self.assertFalse(any('XATH' in text or 'Asset A' in text or 'U1' in text for text in texts))
        self.encoder_inputs.append(texts)
        return np.eye(len(texts)), {'encoder': 'synthetic'}

    def test_official_protocol_predictions_and_metrics(self):
        with patch.object(runner, 'MAX_USERS', 2):
            official, baseline_metrics = runner.verify_official(self.args, self.bundle)
            runner.execute(self.args, self.bundle, self.fake_encode,
                           verified=(official, baseline_metrics, None))
        path = self.args.output_dir
        protocol = runner.read_json(path / 'protocol.json')
        self.assertEqual(protocol['users'], self.bundle[-1]['users'])
        self.assertEqual(protocol['eligible_assets'], list('ABCDEFGHIJKL'))
        self.assertEqual(protocol['end_exclusive'], '2021-07-01')
        self.assertTrue(protocol['repeat_allowed'])
        self.assertFalse(protocol['future_labels_used_for_scoring'])
        self.assertEqual(len(self.encoder_inputs[0]), 10)
        metrics = runner.read_json(path / 'metrics.json')
        self.assertEqual(set(metrics), set(runner.METHODS + runner.BASELINES))
        self.assertEqual(metrics['one_hot']['users'], 2)
        self.assertEqual(metrics['one_hot']['nonempty_truth_users'], 1)
        with (path / 'predictions.csv').open(newline='') as stream:
            predictions = list(csv.DictReader(stream))
        self.assertEqual(len(predictions), 2 * 6 * 10)
        for method in metrics:
            for uid in protocol['users']:
                rows = [row for row in predictions if row['user_id'] == uid and row['method'] == method]
                self.assertEqual([int(item['rank']) for item in rows], list(range(1, 11)))
                fixture.st.validate_ranking([item['ISIN'] for item in rows], protocol['eligible_assets'])
        summary = runner.read_json(path / 'short_history.json')
        self.assertEqual(summary['users'], 2)
        self.assertEqual(summary['nonempty_truth_users'], 1)
        self.assertAlmostEqual(summary['mean_paired_qwen3_minus_one_hot'],
                               metrics['qwen3']['ndcg_at_10'] - metrics['one_hot']['ndcg_at_10'])
        case = runner.read_json(path / 'manual_case.json')
        self.assertEqual(case['truth'], ['A', 'E'])
        for name, ranking in case['rankings'].items():
            dcg = sum(1 / math.log2(row['rank'] + 1) for row in ranking if row['hit'])
            self.assertAlmostEqual(dcg / (1 + 1 / math.log2(3)), case['ndcg_at_10'][name])

    def test_protocol_rejects_a_nonmatching_data_hash(self):
        manifest = dict(self.bundle[-1], data_sha256={})
        with patch.object(runner, 'MAX_USERS', 2):
            with self.assertRaisesRegex(ValueError, 'source_sha256'):
                runner.verify_official(self.args, (*self.bundle[:-1], manifest))

    def test_smoke_is_label_free_and_validates_outputs(self):
        self.args.command = 'smoke'
        self.args.output_dir = self.root / 'smoke'
        runner.execute(self.args, self.bundle, self.fake_encode)
        path = self.args.output_dir
        self.assertFalse((path / 'metrics.json').exists())
        self.assertFalse(runner.read_json(path / 'protocol.json')['future_labels_used_for_evaluation'])
        cases = runner.read_json(path / 'rankings.json')
        self.assertEqual(len(cases), 2)
        for case in cases:
            for ranking in case['rankings'].values():
                fixture.st.validate_ranking(ranking, self.bundle[1])
        with patch.object(runner, 'top10', return_value=['A'] * 10):
            with self.assertRaisesRegex(ValueError, 'ten unique IDs'):
                runner.execute(self.args, self.bundle, self.fake_encode)

    def test_future_mutation_does_not_change_predictions(self):
        modified = self.root / 'modified'
        modified.mkdir()
        fixture.write_fixture(modified, future_variant='mutated')
        self.args.data_dir = modified
        alternate = runner.data_adapter.load(self.args)
        self.assertNotEqual(self.bundle[3], alternate[3])
        def predict(bundle):
            starter, assets, history, _, _, _, _, pop, manifest = bundle
            return runner.predict(starter, manifest['users'], assets, history, pop, self.fake_encode)[0]
        self.assertEqual(predict(self.bundle), predict(alternate))


if __name__ == '__main__':
    unittest.main()
