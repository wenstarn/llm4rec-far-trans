"""Contract checks with synthetic IDs; no model or real customer data."""
import copy
import json
import unittest

from experiments import benchmark_best_match as benchmark


class ResponseContractTests(unittest.TestCase):
    def setUp(self):
        self.payload = {'candidates': [{'ISIN': f'C{i}'} for i in range(10)],
                        'history': [{'ISIN': 'H1'}]}
        self.response = {'matches': [{'candidate_id': f'C{i}', 'history_id': 'H1', 'score': i % 5}
                                     for i in range(10)]}

    def check(self, response):
        return benchmark.validate_response(json.dumps(response), self.payload)

    def test_valid_single_history_and_all_score_values(self):
        self.assertTrue(self.check(self.response)['valid'])

    def test_enclosing_fence_only(self):
        raw = json.dumps(self.response)
        result = benchmark.validate_response('```json\n' + raw + '\n```', self.payload)
        self.assertTrue(result['valid'])
        self.assertTrue(result['fence_removed'])
        self.assertFalse(benchmark.validate_response('Answer: ' + raw, self.payload)['valid'])

    def test_duplicate_candidate_and_missing_coverage(self):
        self.response['matches'][1]['candidate_id'] = 'C0'
        checks = self.check(self.response)['checks']
        self.assertFalse(checks['candidate_unique'])
        self.assertFalse(checks['candidate_coverage'])

    def test_unknown_ids(self):
        for key, check in [('candidate_id', 'candidate_membership'), ('history_id', 'history_membership')]:
            with self.subTest(key=key):
                response = copy.deepcopy(self.response)
                response['matches'][0][key] = 'UNKNOWN'
                self.assertFalse(self.check(response)['checks'][check])

    def test_score_range_and_type(self):
        for score in (-1, 5, 1.0, True, '2', None):
            with self.subTest(score=score):
                self.response['matches'][0]['score'] = score
                self.assertFalse(self.check(self.response)['checks']['score_range_integer'])

    def test_count_fields_root_and_duplicates(self):
        mutations = [self.response['matches'], {'matches': self.response['matches'][:9]},
                     {'matches': self.response['matches'], 'extra': 1}]
        for response in mutations:
            self.assertFalse(self.check(response)['valid'])
        del self.response['matches'][0]['score']
        self.assertFalse(self.check(self.response)['valid'])
        self.assertFalse(benchmark.validate_response('{"matches": [], "matches": []}', self.payload)['valid'])

    def test_downstream_checks_are_unassessed_after_bad_structure(self):
        result = self.check(self.response['matches'])
        self.assertFalse(result['checks']['root_schema'])
        self.assertIsNone(result['checks']['history_membership'])
        self.assertIsNone(result['checks']['score_range_integer'])


if __name__ == '__main__':
    unittest.main()
