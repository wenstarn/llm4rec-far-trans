"""Padding, cosine aggregation and feature-isolation controls on synthetic data."""
import unittest

import torch

from experiments import benchmark_embeddings as benchmark


class EmbeddingTests(unittest.TestCase):
    def test_pool_ignores_left_and_right_padding(self):
        hidden = torch.tensor([[[99.], [1.], [2.]], [[3.], [4.], [99.]]])
        mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
        self.assertEqual(benchmark.last_token_pool(hidden, mask).tolist(), [[2.], [4.]])

    def test_single_history_methods_agree(self):
        history = torch.tensor([[1., 0.]])
        candidates = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        pairs, max_scores, mean_scores = benchmark.similarity_scores(history, candidates)
        self.assertEqual(pairs[:, 0].tolist(), [1., 0., -1.])
        self.assertTrue(torch.equal(max_scores, mean_scores))

    def test_centroid_is_renormalized_and_methods_can_disagree(self):
        history = torch.eye(2)
        candidates = torch.nn.functional.normalize(torch.tensor([[1., 0.], [1., 1.]]), dim=1)
        _, maximum, mean = benchmark.similarity_scores(history, candidates)
        self.assertGreater(maximum[0], maximum[1])
        self.assertGreater(mean[1], mean[0])
        self.assertAlmostEqual(mean[1].item(), 1., places=6)

    def test_identifiers_and_dates_do_not_enter_features(self):
        a = dict.fromkeys(benchmark.FIELDS, 'known')
        a.update(ISIN='ID1', timestamp='2020-01-01', last_buy_date='2020-02-01')
        b = dict(a, ISIN='ID2', timestamp='2021-01-01', last_buy_date='2021-02-01')
        self.assertEqual(benchmark.asset_text(a), benchmark.asset_text(b))


if __name__ == '__main__':
    unittest.main()
