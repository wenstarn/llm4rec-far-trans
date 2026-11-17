"""Synthetic controls: no real transactions, model loading, MPS or downloads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import embedding_reranker as r
import run_embeddings as runner
import torch


def fixture():
    ids = [f'A{i:02}' for i in range(50)]
    assets = {isin: dict(ISIN=isin, assetName=isin, assetCategory='Equity',
                        assetSubCategory='Common', sector='unknown', industry='unknown',
                        marketID='Test', timestamp='2021-05-01') for isin in ids}
    context = {'cutoff': '2021-06-01', 'candidate_assets': [{'id': i} for i in ids],
               'eligible_asset_ids': ids}
    return ids, assets, context


def metadata(revision='test'):
    return {'model': 'synthetic', 'revision': revision, 'settings': {'embedding_dimensions': 2}}


class FakeEncoder:
    def __init__(self):
        self.calls = []
        self.metadata = {'model': 'synthetic'}

    def encode(self, texts):
        self.calls.append(list(texts))
        values = [[1., (sum(map(ord, text)) % 101) / 100] for text in texts]
        return torch.nn.functional.normalize(torch.tensor(values), dim=1)


def synthetic_bundle(truth=None):
    ids, assets, context = fixture()
    users = ['one', 'two', 'five']
    history = {u: dict.fromkeys(ids[:n], 1) for u, n in zip(users, (1, 2, 5))}
    events = {u: [('2021-05-20', i, 'Buy') for i in history[u]] for u in users}
    spec = importlib.util.spec_from_file_location('test_embedding_starter', runner.base.ROOT / 'shared/far-trans/starter.py')
    starter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(starter)
    baseline = {u: {'cbf': ids[:10]} for u in users}
    return (starter, assets, history, truth, {u: context for u in users}, baseline, events, {},
            {'users': users, 'cutoff': '2021-06-01', 'data_sha256': {'synthetic': 'test'}})


class RerankerTests(unittest.TestCase):
    def test_history_filters_future_sell_unknown_and_truncates_distinct(self):
        ids, assets, context = fixture()
        events = [('2021-05-20', i, 'Buy') for i in reversed(ids[:7])]
        events += [('2021-05-30', 'A06', 'Buy'), ('2021-05-29', 'A05', 'Buy'),
                   ('2021-05-31', 'A04', 'Sell'), ('2021-06-01', 'A04', 'Buy'),
                   ('2022-01-01', 'A03', 'Buy'), ('2021-05-31', 'MISSING', 'Buy')]
        case = r.prepare_case(context, assets, events)
        self.assertEqual([a['ISIN'] for a in case['history']], ['A06', 'A05', 'A00', 'A01', 'A02'])
        self.assertEqual([a['ISIN'] for a in case['candidates']], ids)
        self.assertEqual(case['history'][0]['last_buy_date'], '2021-05-30')

    def test_rejects_future_metadata_and_empty_history(self):
        _, assets, context = fixture()
        events = [('2021-05-20', 'A00', 'Buy')]
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            r.prepare_case(context, assets, [('2021-06-01', 'A00', 'Buy')])
        assets['A49']['timestamp'] = '2021-06-01'
        with self.assertRaisesRegex(ValueError, 'metadata'):
            r.prepare_case(context, assets, events)

    def test_rejects_wrong_candidate_pool(self):
        ids, assets, context = fixture()
        events = [('2021-05-20', 'A00', 'Buy')]
        for candidates in (ids[:49], ids[:49] + ['A00'], ids[:49] + ['NOT_ELIGIBLE']):
            with self.subTest(candidates=candidates[-1]), self.assertRaises(ValueError):
                r.prepare_case(dict(context, candidate_assets=[{'id': i} for i in candidates]), assets, events)

    def test_rounding_ties_keep_cbf_order_and_top10_is_valid(self):
        ids, _, _ = fixture()
        scores = [.1] * 50
        scores[8], scores[2] = .9000001, .9000002
        top10 = r.rank_scores(ids, scores)
        self.assertEqual(top10[:2], ['A02', 'A08'])
        self.assertEqual(len(set(top10)), 10)
        self.assertLess(set(top10), set(ids))
        with self.assertRaises(ValueError):
            r.rank_scores(ids, [float('nan')] + scores[1:])

    def test_cache_persists_deduplicates_and_encodes_only_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = FakeEncoder()
            cache = r.EmbeddingCache(directory, metadata())
            first = cache.ensure(['a', 'a', 'b'], encoder.encode)
            self.assertEqual(first['cache_misses'], 2)
            self.assertEqual(encoder.calls, [['a', 'b']])
            reopened = r.EmbeddingCache(directory, metadata())
            self.assertTrue(torch.equal(cache.get(['a', 'b']), reopened.get(['a', 'b'])))
            second = reopened.ensure(['a', 'b', 'c'], encoder.encode)
            self.assertEqual((second['cache_hits'], second['cache_misses']), (2, 1))
            self.assertEqual(encoder.calls[-1], ['c'])
            changed_model = r.EmbeddingCache(directory, metadata('another_revision'))
            self.assertNotEqual(cache.path, changed_model.path)
            self.assertEqual(changed_model.vectors, {})

    def test_cache_rejects_invalid_embeddings_and_does_not_save(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = r.EmbeddingCache(directory, metadata())
            for vector in (torch.zeros((1, 2)), torch.tensor([[float('nan'), 0.]]), torch.ones((1, 3))):
                with self.subTest(vector=vector), self.assertRaises(ValueError):
                    cache.ensure(['a'], lambda texts: vector)
            self.assertEqual(cache.vectors, {})
            self.assertFalse(cache.path.exists())

    def test_both_methods_share_vectors_but_rank_differently(self):
        ids, assets, context = fixture()
        case = r.prepare_case(context, assets, [('2021-05-20', 'A00', 'Buy'), ('2021-05-20', 'A01', 'Buy')])
        def encode(texts):
            values = [[1., 0.] if text.startswith('Asset name: A00\n') else
                      [0., 1.] if text.startswith('Asset name: A01\n') else
                      [1., 1.] if text.startswith('Asset name: A02\n') else [-1., 0.] for text in texts]
            return torch.nn.functional.normalize(torch.tensor(values), dim=1)
        with tempfile.TemporaryDirectory() as directory:
            cache = r.EmbeddingCache(directory, metadata())
            result, timing = r.rerank_cases([case], cache, SimpleNamespace(encode=encode), device='cpu')
        record = result[0]
        self.assertEqual(record['rankings']['MaxSim'][:2], ['A00', 'A01'])
        self.assertEqual(record['rankings']['MeanSim'][0], 'A02')
        self.assertAlmostEqual(record['scores']['MeanSim'][2], 1., places=6)
        self.assertEqual(timing['cache']['cache_misses'], 50)

    def test_single_history_same_scores_and_order(self):
        _, assets, context = fixture()
        case = r.prepare_case(context, assets, [('2021-05-20', 'A00', 'Buy')])
        with tempfile.TemporaryDirectory() as directory:
            result, _ = r.rerank_cases([case], r.EmbeddingCache(directory, metadata()), FakeEncoder(), device='cpu')
        self.assertEqual(result[0]['rankings']['MaxSim'], result[0]['rankings']['MeanSim'])
        torch.testing.assert_close(torch.tensor(result[0]['scores']['MaxSim']),
                                   torch.tensor(result[0]['scores']['MeanSim']), atol=1e-6, rtol=0)

    def test_zero_centroid_and_duplicate_history_fail(self):
        _, assets, context = fixture()
        case = r.prepare_case(context, assets, [('2021-05-20', 'A00', 'Buy'), ('2021-05-20', 'A01', 'Buy')])
        def encode(texts):
            return torch.tensor([[-1., 0.] if text.startswith('Asset name: A01\n') else [1., 0.] for text in texts])
        with tempfile.TemporaryDirectory() as directory:
            cache = r.EmbeddingCache(directory, metadata())
            with self.assertRaisesRegex(ValueError, 'zero norm'):
                r.rerank_cases([case], cache, SimpleNamespace(encode=encode), device='cpu')
            case['history'] = [case['history'][0]] * 2
            with self.assertRaisesRegex(ValueError, 'Duplicate historical'):
                r.rerank_cases([case], cache, FakeEncoder(), device='cpu')

    def test_smoke_never_evaluates_and_cache_reload_does_not_encode(self):
        class PoisonTruth:
            def get(self, *args):
                raise AssertionError('Future labels accessed')
        bundle = synthetic_bundle(PoisonTruth())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            args = SimpleNamespace(command='smoke', output_dir=path, cutoff='2021-06-01', horizon_days=30)
            encoder = FakeEncoder()
            with patch.object(runner.data_adapter, 'save_results', side_effect=AssertionError('Evaluation called')):
                result = runner.execute(args, bundle, r.EmbeddingCache(path / 'cache', metadata()), encoder, device='cpu')
            self.assertEqual(result['scored_users'], 3)
            self.assertEqual(result['valid_top10_count'], 6)
            self.assertFalse(result['future_evaluation_performed'])
            self.assertEqual(len(encoder.calls), 1)
            self.assertEqual(result['timing']['cache_reload_check']['cache']['cache_misses'], 0)
            self.assertFalse((path / 'metrics.json').exists())
            self.assertNotIn('user_id', (path / 'rankings.json').read_text())

    def test_full_evaluation_uses_teacher_metrics_on_synthetic_labels(self):
        bundle = synthetic_bundle({'one': {'A00'}, 'two': set(), 'five': {'A49'}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            args = SimpleNamespace(command='evaluate', output_dir=path, cutoff='2021-06-01', horizon_days=30)
            with patch.object(runner.data_adapter, 'save_results', wraps=runner.data_adapter.save_results) as save:
                result = runner.execute(args, bundle, r.EmbeddingCache(path / 'cache', metadata()), FakeEncoder(), device='cpu')
            save.assert_called_once()
            metrics = json.loads((path / 'metrics.json').read_text())
            self.assertTrue(result['future_evaluation_performed'])
            for name in r.METHODS:
                self.assertEqual(metrics[name]['users'], 3)
                self.assertEqual(metrics[name]['nonempty_truth_users'], 2)
                rankings = json.loads((path / 'rankings.json').read_text())
                expected = [bundle[0].ndcg10(row['rankings'][name], bundle[3].get(uid, set()))
                            for uid, row in zip(bundle[-1]['users'], rankings)]
                self.assertAlmostEqual(metrics[name]['ndcg_at_10'], sum(expected) / 3)
                self.assertAlmostEqual(metrics[name]['ndcg_at_10_nonempty'], (expected[0] + expected[2]) / 2)


if __name__ == '__main__':
    unittest.main()
