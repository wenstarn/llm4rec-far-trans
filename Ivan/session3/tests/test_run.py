from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, Mock, patch

import run


spec = importlib.util.spec_from_file_location('hw3_test_starter', run.STARTER / 'starter.py')
starter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(starter)


class SmokeTest(unittest.TestCase):
    def setUp(self):
        self.pool = [f'asset{i:02}' for i in range(50)]
        self.users = ['five', 'one', 'two', 'another_one']
        self.assets = {isin: {'timestamp': '2021-05-01', 'assetName': isin} for isin in self.pool}
        context = {'cutoff': run.CUTOFF, 'eligible_asset_ids': self.pool,
                   'candidate_assets': [{'id': isin} for isin in self.pool]}
        self.contexts = dict.fromkeys(self.users, context)
        self.baseline = {uid: {'cbf': self.pool[:10]} for uid in self.users}
        self.history = {uid: dict.fromkeys(self.pool[:count], 1)
                        for uid, count in zip(self.users, (5, 1, 2, 1), strict=True)}
        self.events = {uid: [('2021-05-15', isin, 'Buy') for isin in self.history[uid]]
                       for uid in self.users}
        self.manifest = {'users': self.users, 'end_exclusive': '2021-07-01',
                         'repeat_allowed': True, 'data_sha256': {}}

    def bundle(self, truth):
        return (starter, self.assets, self.history, truth, self.contexts, self.baseline,
                self.events, {}, self.manifest)

    def prepare(self, truth):
        with patch.object(run, 'COHORT_SIZE', len(self.users)), patch.object(
            run.data_adapter, 'load', return_value=self.bundle(truth)
        ):
            return run.prepare(run.ROOT / 'data')

    def test_prepare_uses_fixed_past_strata_and_ignores_truth(self):
        _, protocol, cases, prompts, pools = self.prepare({'one': {'asset01'}})
        _, second_protocol, _, second_prompts, _ = self.prepare({'two': {'asset02'}})
        self.assertEqual(protocol, second_protocol)
        self.assertEqual(prompts, second_prompts)
        self.assertEqual(protocol['smoke']['users_sha256'], run.digest(['one', 'two', 'five']))
        self.assertEqual([len(case['history']) for case in cases], [1, 2, 5])
        self.assertEqual(pools, [self.pool] * 3)
        expected_hash = protocol.pop('protocol_sha256')
        self.assertEqual(expected_hash, run.digest(protocol))

    def test_prepare_rejects_ranker_prompt_with_different_candidate_order(self):
        original = run.ranker.build_prompt

        def reversed_prompt(*args):
            instruction, payload = original(*args).split('\n', 1)
            data = json.loads(payload)
            data['candidate_assets'].reverse()
            return instruction + '\n' + json.dumps(data)

        with patch.object(run.ranker, 'build_prompt', side_effect=reversed_prompt):
            with self.assertRaisesRegex(ValueError, 'same ordered CBF pool'):
                self.prepare(None)

    def test_embedding_validation_rejects_invalid_top10(self):
        record = {'candidate_ids': self.pool, 'rankings': {
            'MaxSim': self.pool[:10], 'MeanSim': self.pool[:9] + ['outside'],
        }}
        with self.assertRaisesRegex(ValueError, 'candidate list'):
            run.validate_embeddings(starter, [record], [self.pool])

    def test_llmrank_keeps_valid_model_order_at_context_limit(self):
        generator = Mock()
        generator.prompt_token_count.return_value = run.CONTEXT_LIMIT - run.MAX_NEW_TOKENS
        ranking = self.pool[10:20][::-1]
        raw = json.dumps(ranking)
        generator.generate.return_value = {'text': raw}
        [row] = run.rank_with_llm(starter, generator, ['prompt'], [self.pool])
        self.assertEqual(row['ranking'], ranking)
        self.assertEqual(row['text'], raw)
        self.assertFalse(row['fallback'])
        self.assertIsNone(row['error'])
        generator.generate.assert_called_once_with('prompt', max_new_tokens=512)

    def assert_invalid_response_fallback(self, raw, reason):
        generator = Mock()
        generator.prompt_token_count.return_value = 500
        generator.generate.return_value = {'text': raw}
        [row] = run.rank_with_llm(starter, generator, ['prompt'], [self.pool])
        self.assertEqual(row['ranking'], self.pool[:10])
        self.assertEqual(row['text'], raw)
        self.assertTrue(row['fallback'])
        self.assertEqual(row['error']['phase'], 'validation')
        self.assertIn(reason, row['error']['message'])

    def test_llmrank_falls_back_for_out_of_pool_isin(self):
        self.assert_invalid_response_fallback(json.dumps(self.pool[:9] + ['outside']), 'outside candidate set')

    def test_llmrank_falls_back_for_duplicate_isins(self):
        self.assert_invalid_response_fallback(json.dumps(self.pool[:9] + [self.pool[0]]), 'Duplicate ISINs')

    def test_llmrank_falls_back_for_wrong_length_or_format(self):
        for raw, reason in (
            (json.dumps(self.pool[:9]), 'exactly ten'),
            ('malformed', 'No JSON array'),
            (f'```json\n{json.dumps(self.pool[:10])}\n```', 'without fences'),
        ):
            with self.subTest(reason=reason):
                self.assert_invalid_response_fallback(raw, reason)

    def test_llmrank_falls_back_before_generation_when_context_overflows(self):
        generator = Mock()
        generator.prompt_token_count.return_value = run.CONTEXT_LIMIT - run.MAX_NEW_TOKENS + 1
        [row] = run.rank_with_llm(starter, generator, ['too long'], [self.pool])
        self.assertEqual(row['ranking'], self.pool[:10])
        self.assertTrue(row['fallback'])
        self.assertIsNone(row['text'])
        self.assertEqual(row['error']['phase'], 'context')
        generator.generate.assert_not_called()

    def test_llmrank_reports_unexpected_generation_errors_and_keeps_every_case(self):
        generator = Mock()
        generator.prompt_token_count.return_value = 500
        generator.generate.side_effect = [
            RuntimeError('MPS generation failed'), ValueError('Unexpected tokenizer failure'),
            {'text': json.dumps(self.pool[:10])},
        ]
        responses = run.rank_with_llm(starter, generator, ['first', 'second', 'third'], [self.pool] * 3)
        self.assertEqual(len(responses), 3)
        for row, error_type in zip(responses[:2], ('RuntimeError', 'ValueError'), strict=True):
            self.assertIsNone(row['ranking'])
            self.assertFalse(row['fallback'])
            self.assertEqual(row['error']['phase'], 'generation')
            self.assertEqual(row['error']['type'], error_type)
        self.assertEqual(responses[2]['ranking'], self.pool[:10])

    def test_smoke_connects_all_three_methods_without_grading(self):
        prepared = self.prepare(None)
        records = [{'candidate_ids': pool, 'rankings': {name: pool[:10] for name in run.reranker.METHODS}}
                   for pool in prepared[-1]]
        generator = Mock(metadata={})
        generator.prompt_token_count.side_effect = (500, 500, run.CONTEXT_LIMIT)
        generator.generate.side_effect = ({'text': json.dumps(self.pool[:10])}, {'text': 'invalid'})
        fake_torch = MagicMock()
        fake_torch.backends.mps.is_built.return_value = True
        fake_torch.backends.mps.is_available.return_value = True
        probe = fake_torch.ones.return_value.__add__.return_value
        probe.device.type = 'mps'
        probe.item.return_value = 2
        grader = Mock(side_effect=AssertionError('Smoke must not grade'))
        with (
            TemporaryDirectory() as folder,
            patch.object(run, 'prepare', return_value=prepared),
            patch.object(run, 'torch', fake_torch),
            patch.multiple(run.reranker, require_mps=Mock(), EmbeddingCache=Mock(), QwenEmbedder=Mock(),
                           rerank_cases=Mock(return_value=(records, {}))),
            patch.object(run, 'MPSGenerator', return_value=generator),
            patch.object(starter, 'ndcg10', grader),
            redirect_stdout(io.StringIO()),
        ):
            output = Path(folder) / 'smoke'
            self.assertEqual(run.main(['smoke', '--output-dir', str(output)]), 0)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['valid_top10'], {'MaxSim': 3, 'MeanSim': 3, 'llmrank': 3})
            self.assertEqual(summary['llmrank'], {'valid_responses': 1, 'fallbacks': 2})
            self.assertEqual(summary['errors'], [])
            results = json.loads((output / 'llmrank.json').read_text())['responses']
            self.assertEqual([row['fallback'] for row in results], [False, True, True])
            self.assertEqual([row['ranking'] for row in results], [self.pool[:10]] * 3)
            self.assertFalse((output / 'metrics.json').exists())
            grader.assert_not_called()

    def test_evaluate_interrupts_and_resumes_in_original_order(self):
        with TemporaryDirectory() as folder:
            output = Path(folder) / 'run'
            checkpoints = output / 'llmrank_checkpoints'
            valid = {'text': json.dumps(self.pool[10:20])}
            interrupted, resumed = Mock(), Mock()
            interrupted.prompt_token_count.return_value = 500
            resumed.prompt_token_count.return_value = 500
            resumed.generate.return_value = valid

            def interrupt_after_two(prompt, *, max_new_tokens):
                count = interrupted.generate.call_count
                if count == 3:
                    self.assertTrue((checkpoints / '001.json').is_file())
                    self.assertTrue((checkpoints / '002.json').is_file())
                    raise KeyboardInterrupt
                return valid if count == 1 else {'text': 'invalid JSON'}

            interrupted.generate.side_effect = interrupt_after_two
            factory = Mock(side_effect=[interrupted, resumed])
            records = [{'candidate_ids': self.pool, 'rankings': {name: self.pool[:10]
                        for name in run.reranker.METHODS}} for _ in self.users]
            fake_rerank = Mock(return_value=(records, {}))
            save_results = run.data_adapter.save_results

            def grade(args, bundle, predictions, failures, method, metadata):
                self.assertEqual(list(predictions), self.users)
                self.assertEqual(failures[0]['user_id'], self.users[1])
                self.assertEqual(metadata['fallbacks'], 1)
                self.assertFalse((output / 'evaluation').exists())
                save_results(args, bundle, predictions, failures, method, metadata)

            args = ['evaluate', '--output-dir', str(output)]
            with (
                patch.object(run, 'COHORT_SIZE', len(self.users)),
                patch.object(run, 'load_bundle', return_value=self.bundle({})),
                patch.object(run, 'MPSGenerator', factory),
                patch.multiple(run.reranker, EmbeddingCache=Mock(), QwenEmbedder=Mock(), rerank_cases=fake_rerank),
                patch.object(run.data_adapter, 'save_results', side_effect=grade) as grade_call,
                patch.object(starter, 'ndcg10', return_value=0.0) as scorer,
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
            ):
                prompts = run.prepare_inputs(self.bundle({}), full=True)[3]
                self.assertEqual(run.main(args), 130)
                grade_call.assert_not_called()
                scorer.assert_not_called()
                fake_rerank.assert_not_called()
                self.assertFalse((output / 'evaluation').exists())
                completed_paths = sorted(checkpoints.glob('*.json'))
                self.assertEqual([p.name for p in completed_paths], ['001.json', '002.json'])
                original = [p.read_bytes() for p in completed_paths]
                fallback = json.loads(original[1])
                self.assertEqual(fallback['status'], 'fallback')
                self.assertTrue(fallback['fallback'])
                self.assertEqual(fallback['text'], 'invalid JSON')
                self.assertEqual(fallback['ranking'], self.pool[:10])
                self.assertEqual(fallback['error']['phase'], 'validation')
                self.assertGreaterEqual(fallback['wall_seconds'], 0)

                self.assertEqual(run.main([*args, '--resume']), 0)
                self.assertEqual([call.args[0] for call in resumed.generate.call_args_list], prompts[2:])
                self.assertEqual(original, [p.read_bytes() for p in completed_paths])
                self.assertEqual([json.loads(p.read_text())['user_id'] for p in sorted(checkpoints.glob('*.json'))],
                                 self.users)
                self.assertTrue((output / 'evaluation/metrics.json').exists())
                self.assertEqual(scorer.call_count, len(self.users) * 4)
                self.assertEqual(run.main([*args, '--resume']), 0)
                self.assertEqual(factory.call_count, 2)
                grade_call.assert_called_once()

    def test_technical_error_stops_after_saving_and_remains_retryable(self):
        with TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            checkpoints = Path(folder)
            first, second = Mock(), Mock()
            first.prompt_token_count.return_value = second.prompt_token_count.return_value = 500
            valid = {'text': json.dumps(self.pool[:10])}
            first.generate.side_effect = [valid, RuntimeError('MPS failure')]
            second.generate.return_value = valid
            with patch.object(run, 'MPSGenerator', side_effect=[first, second]):
                args = (starter, checkpoints, 'protocol', self.users, self.users, [self.pool] * len(self.users))
                with self.assertRaisesRegex(RuntimeError, 'case 2/4.*MPS failure'):
                    run.complete_llm(*args)
                error = json.loads((checkpoints / '002.json').read_text())
                self.assertEqual(error['status'], 'technical_error')
                self.assertFalse(error['fallback'])
                self.assertEqual(error['error']['message'], 'MPS failure')
                self.assertFalse((checkpoints / '003.json').exists())
                responses = run.complete_llm(*args)
                self.assertEqual([row['status'] for row in responses], ['valid'] * len(self.users))
                self.assertEqual([call.args[0] for call in second.generate.call_args_list], self.users[1:])

    def test_resume_rejects_changed_protocol_before_generating(self):
        with (
            TemporaryDirectory() as folder,
            patch.object(run, 'COHORT_SIZE', len(self.users)),
            patch.object(run, 'load_bundle', return_value=self.bundle({})),
            patch.object(run, 'MPSGenerator', side_effect=KeyboardInterrupt) as factory,
        ):
            output = Path(folder) / 'run'
            with self.assertRaises(KeyboardInterrupt):
                run.evaluate(run.ROOT / 'data', output, resume=False)
            factory.reset_mock()
            for field, value in (('CUTOFF', '2021-05-31'), ('LLM_REVISION', 'other'), ('MAX_NEW_TOKENS', 511)):
                with self.subTest(field=field), patch.object(run, field, value):
                    with self.assertRaisesRegex(ValueError, 'Incompatible protocol'):
                        run.evaluate(run.ROOT / 'data', output, resume=True)
            candidates = self.contexts[self.users[0]]['candidate_assets']
            candidates[10:] = reversed(candidates[10:])
            with self.assertRaisesRegex(ValueError, 'Incompatible protocol'):
                run.evaluate(run.ROOT / 'data', output, resume=True)
            factory.assert_not_called()

    def test_atomic_checkpoint_preserves_old_file_if_replacement_is_interrupted(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / '001.json'
            run.atomic_json(path, {'status': 'technical_error'})
            with patch.object(run.os, 'replace', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run.atomic_json(path, {'status': 'valid'})
            self.assertEqual(json.loads(path.read_text()), {'status': 'technical_error'})
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_final_evaluation_rejects_incomplete_cohort(self):
        with patch.object(run.data_adapter, 'save_results') as grade:
            with self.assertRaisesRegex(ValueError, 'requires all 500 completed users'):
                run.publish_evaluation(Path('/unused'), self.bundle({}), {}, [], [], [])
            grade.assert_not_called()


if __name__ == '__main__':
    unittest.main()
