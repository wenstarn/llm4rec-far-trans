"""Numerical and batching contract tests; synthetic tensors, no model loading."""
import ast
from pathlib import Path
import unittest

import torch
from experiments import benchmark_ab_logits as benchmark


class ABLogitTests(unittest.TestCase):
    def test_softmax_is_restricted_to_labels(self):
        logits = torch.tensor([[1000., 2., 1., -1000.], [1000., 0., 0., -1000.]])
        result = benchmark.ab_probabilities(logits, 1, 2)
        self.assertAlmostEqual(result[0, 0].item(), 0.73105858, places=6)
        self.assertEqual(result[1].tolist(), [0.5, 0.5])
        self.assertTrue(torch.allclose(result.sum(-1), torch.ones(2)))

    def test_fp16_logits_are_normalized_in_fp32(self):
        result = benchmark.ab_probabilities(torch.tensor([[100., -100.]], dtype=torch.float16), 0, 1)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all())
        with self.assertRaises(ValueError):
            benchmark.ab_probabilities(torch.zeros((1, 2)), 0, 0)

    def test_left_padding_positions(self):
        mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]])
        self.assertEqual(benchmark.position_ids_for_mask(mask).tolist(),
                         [[0, 0, 0, 1], [0, 0, 1, 2], [0, 1, 2, 3]])

    def test_max_aggregation_and_stable_ties(self):
        pairs, values = [], []
        for ci in range(10):
            for hi in range(2):
                pairs.append({'case_id': 'test', 'candidate_index': ci, 'history_index': hi})
                score = .9 if ci in (0, 2) and hi == 1 else (.8 if ci == 1 else .2)
                values.append([0., 0., score, 1-score, 1.])
        result = benchmark.aggregate_max(pairs, values, 'test')
        self.assertEqual(result['ranked_candidate_indices'][:3], [0, 2, 1])
        self.assertEqual(result['best_history_indices'][0], 1)
        with self.assertRaises(ValueError):
            benchmark.aggregate_max(pairs[:-2], values[:-2], 'test')

    def test_distribution_and_no_generation(self):
        summary = benchmark.distribution([.2, .2, .2])
        self.assertEqual(summary['span'], 0)
        self.assertEqual(summary['std'], 0)
        self.assertEqual(summary['unique'], 1)
        tree = ast.parse(Path(benchmark.__file__).read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        self.assertFalse(any(isinstance(n.func, ast.Attribute) and n.func.attr == 'generate' for n in calls))


if __name__ == '__main__':
    unittest.main()
