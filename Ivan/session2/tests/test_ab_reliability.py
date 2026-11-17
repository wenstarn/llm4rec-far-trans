"""Synthetic controls for semantic label remapping and paired comparisons."""
import unittest

from experiments import benchmark_ab_reliability as benchmark


class ReliabilityTests(unittest.TestCase):
    def test_only_label_suffix_changes(self):
        prefix = 'Metadata A and B remain untouched.\n'
        original = prefix + benchmark.ORIGINAL_SUFFIX
        self.assertEqual(benchmark.swapped_prompt(original), prefix + benchmark.SWAPPED_SUFFIX)
        with self.assertRaises(ValueError):
            benchmark.swapped_prompt('unexpected')

    def test_probability_and_margin_follow_semantics(self):
        row = [8., 2., .9975, .0025, 1.]
        self.assertEqual(benchmark.semantic_values([row], 'original')[0][2], .9975)
        self.assertEqual(benchmark.semantic_values([row], 'swapped')[0][2], .0025)
        self.assertEqual(benchmark.semantic_margin(row,'original'), 6.)
        self.assertEqual(benchmark.semantic_margin(row,'swapped'), -6.)
        self.assertEqual(row, [8.,2.,.9975,.0025,1.])

    def test_remap_before_max_over_history(self):
        pairs, values = [], []
        for ci in range(10):
            for hi, pa in enumerate((.8,.2)):
                pairs.append({'case_id':'case','candidate_index':ci,'history_index':hi})
                values.append([0.,0.,pa,1-pa,1.])
        mapped = benchmark.semantic_values(values,'swapped')
        result = benchmark.scorer.aggregate_max(pairs,mapped,'case')
        self.assertAlmostEqual(result['scores_cbf_order'][0], .8)
        self.assertEqual(result['best_history_indices'][0], 1)

    def test_order_comparison(self):
        identity = list(range(10))
        self.assertEqual(benchmark.order_agreement(identity,identity)['kendall_order_tau'], 1)
        reversed_result = benchmark.order_agreement(identity,identity[::-1])
        self.assertEqual(reversed_result['kendall_order_tau'], -1)
        self.assertFalse(reversed_result['same_top1'])
        self.assertEqual(reversed_result['top5_overlap'], 0)

    def test_ties_and_semantic_agreement(self):
        self.assertEqual(benchmark.midranks([2,1,1]), [2, .5, .5])
        self.assertIsNone(benchmark.pearson([1,1],[2,3]))
        self.assertAlmostEqual(benchmark.pearson([1,2,3],[3,2,1]), -1)
        agreement = benchmark.score_agreement([.9,.99],[.1,.01])
        self.assertEqual(agreement['semantic_decision_agreement'], 0)
        self.assertAlmostEqual(agreement['mean_absolute_delta'], .89)


if __name__ == '__main__':
    unittest.main()
