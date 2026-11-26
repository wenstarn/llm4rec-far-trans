"""Smoke-test or evaluate FAR-Trans rerankers on the fixed HW3 cohort."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
HW2 = ROOT / 'Ivan/session2'
STARTER = ROOT / 'shared/far-trans'
OUTPUTS = Path(__file__).resolve().parent / 'outputs'
CUTOFF = '2021-06-01'
HORIZON_DAYS = 30
COHORT_SIZE = 500
CANDIDATE_COUNT = 50
LLM_MODEL = 'Qwen/Qwen3-1.7B'
LLM_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
HISTORY_EVENTS = 15
MAX_NEW_TOKENS = 512
CONTEXT_LIMIT = 8192
EVALUATION_FILES = ('predictions.csv', 'metrics.json', 'per_user.json', 'run_manifest.json')

sys.dont_write_bytecode = True
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
sys.path.insert(0, str(HW2))
import run_embeddings as hw2
from run_embeddings import data_adapter, reranker
import runtime

ranker = reranker.teacher_ranker
torch = reranker.torch
digest = hw2.base.digest


class MPSGenerator(runtime.LocalGenerator):
    """Load the pinned ranker on MPS and inherit the teacher's decoding path."""

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

        reranker.require_mps()
        self.torch = torch
        self.device = 'mps'
        options = dict(revision=LLM_REVISION, local_files_only=True, trust_remote_code=False)
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, **options)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL, dtype=torch.float16, **options
        ).to(self.device).eval()
        torch.mps.synchronize()
        if any(p.device.type != self.device or p.dtype != torch.float16 for p in self.model.parameters()):
            raise RuntimeError('Expected all ranker parameters on MPS in float16')
        self.metadata = {
            'model': LLM_MODEL, 'revision': LLM_REVISION, 'device': self.device,
            'dtype': 'float16', 'torch_version': torch.__version__, 'transformers_version': __version__,
        }

    def prompt_token_count(self, prompt):
        # Match LocalGenerator.generate's chat formatting and tokenization before moving to MPS.
        text = self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        return len(self.tokenizer(text)['input_ids'])


def load_bundle(data_dir):
    args = SimpleNamespace(data_dir=data_dir, starter_dir=STARTER, cutoff=CUTOFF,
                           horizon_days=HORIZON_DAYS, max_users=COHORT_SIZE,
                           candidate_count=CANDIDATE_COUNT)
    return data_adapter.load(args)


def prepare(data_dir):
    return prepare_inputs(load_bundle(data_dir), full=False)


def prepare_inputs(bundle, *, full):
    starter, assets, history, _truth, contexts, baseline, events, _pop, manifest = bundle
    users = manifest['users']
    if len(set(users)) != COHORT_SIZE or len(users) != COHORT_SIZE:
        raise ValueError(f'Expected {COHORT_SIZE} distinct warm users')
    pools = {uid: ranker.candidate_ids(contexts[uid]) for uid in users}
    if any(len(pool) != CANDIDATE_COUNT or pool[:10] != baseline[uid]['cbf']
           for uid, pool in pools.items()):
        raise ValueError('Expected CBF top-50 in its original order')

    selected = users
    if not full:
        try:
            selected = hw2.select_smoke_users(manifest, history)
        except StopIteration as exc:
            raise ValueError('Smoke requires users with 1, 2 and 5 distinct past Buys') from exc
    cases, prompts = [], []
    selected_pools = [pools[uid] for uid in selected]
    for uid, pool in zip(selected, selected_pools, strict=True):
        case = reranker.prepare_case(contexts[uid], assets, events[uid])
        prompt = ranker.build_prompt(contexts[uid], assets, history[uid], events[uid], HISTORY_EVENTS)
        prompt_assets = json.loads(prompt.split('\n', 1)[1])['candidate_assets']
        embedding_ids = [asset['ISIN'] for asset in case['candidates']]
        prompt_ids = [asset['ISIN'] for asset in prompt_assets]
        if embedding_ids != pool or prompt_ids != pool:
            raise ValueError('Embedding and LLMRank inputs must use the same ordered CBF pool')
        cases.append(case)
        prompts.append(prompt)

    protocol = {
        'cohort': 'warm', 'cohort_size': COHORT_SIZE, 'selection_seed': 42,
        'cutoff': CUTOFF, 'horizon_days': HORIZON_DAYS, 'end_exclusive': manifest['end_exclusive'],
        'candidate_count': CANDIDATE_COUNT, 'candidate_order': 'starter CBF', 'top_k': 10,
        'repeat_allowed': manifest['repeat_allowed'], 'data_sha256': manifest['data_sha256'],
        'cohort_sha256': digest(users), 'candidate_pools_sha256': digest(list(pools.items())),
        'embeddings': {'model': reranker.MODEL, 'revision': reranker.REVISION, 'history_limit': 5},
        'llmrank': {
            'model': LLM_MODEL, 'revision': LLM_REVISION, 'history_events': HISTORY_EVENTS,
            'shuffle_seed': None, 'max_new_tokens': MAX_NEW_TOKENS, 'context_limit': CONTEXT_LIMIT,
            'do_sample': False, 'thinking': False,
            'fallback_policy': 'original CBF top-10 on invalid response or context overflow',
        },
        'source_sha256': {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve(), Path(starter.__file__), Path(data_adapter.__file__),
                Path(ranker.__file__), Path(runtime.__file__),
                Path(hw2.__file__), Path(reranker.__file__), Path(reranker.pilot.__file__),
                Path(hw2.base.__file__),
            )
        },
    }
    if full:
        protocol['evaluate'] = {
            'prompts_sha256': digest(prompts), 'embedding_cases_sha256': digest(cases),
            'device': 'mps', 'llmrank_dtype': 'float16', 'cpu_fallback': False,
            'versions': {name: version(name) for name in ('torch', 'transformers', 'tokenizers')},
        }
    else:
        protocol['smoke'] = {
            'selection': 'first user in cohort order per distinct Buy count 1, 2, 5',
            'users_sha256': digest(selected), 'candidate_pools_sha256': digest(selected_pools),
            'prompts_sha256': digest(prompts), 'uses_future_labels': False,
        }
    protocol['protocol_sha256'] = digest(protocol)
    return starter, protocol, cases, prompts, selected_pools


def validate_embeddings(starter, records, pools):
    for record, pool in zip(records, pools, strict=True):
        if record['candidate_ids'] != pool or set(record['rankings']) != set(reranker.METHODS):
            raise ValueError('Unexpected embedding methods or candidate order')
        for ranking in record['rankings'].values():
            starter.validate_ranking(ranking, pool)


def rank_one(starter, generator, prompt, pool):
    started = time.perf_counter()
    result = {'ranking': None, 'text': None, 'fallback': False, 'error': None}
    phase = 'tokenization'
    try:
        input_tokens = generator.prompt_token_count(prompt)
        result['input_tokens'] = input_tokens
        if input_tokens + MAX_NEW_TOKENS > CONTEXT_LIMIT:
            result['error'] = {
                'phase': 'context',
                'message': f'{input_tokens} prompt + {MAX_NEW_TOKENS} reserved tokens exceeds {CONTEXT_LIMIT}',
            }
        else:
            phase = 'generation'
            response = generator.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
            result.update(response)
            raw = response['text']
            phase = 'validation'
            try:
                ranking, repaired = ranker.parse_ranking(raw, set(pool))
                result['parse_repaired'] = repaired
                if repaired:
                    raise ValueError('Expected only a JSON array, without fences or surrounding text')
                result['ranking'] = ranking
            except ValueError as exc:
                result['error'] = {'phase': phase, 'message': str(exc)}

        if result['error'] is not None:
            result.update(ranking=pool[:10], fallback=True)
        phase = 'validation'
        starter.validate_ranking(result['ranking'], pool)
    except Exception as exc:
        result.update(ranking=None, fallback=False,
                      error={'phase': phase, 'type': type(exc).__name__, 'message': str(exc)})
    result['wall_seconds'] = time.perf_counter() - started
    return result


def rank_with_llm(starter, generator, prompts, pools):
    return [rank_one(starter, generator, prompt, pool) for prompt, pool in zip(prompts, pools, strict=True)]


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.', suffix='.tmp',
                            delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def read_checkpoint(path, identity, starter, pool):
    if not path.exists():
        return None
    row = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(row, dict) or any(row.get(key) != value for key, value in identity.items()):
        raise ValueError(f'Incompatible checkpoint {path.name}')
    if 'text' not in row or type(row.get('wall_seconds')) not in (int, float):
        raise ValueError(f'Incomplete checkpoint {path.name}')
    status = row['status']
    if status == 'technical_error' and row['ranking'] is None and not row['fallback'] and row['error']:
        return None
    if status == 'fallback':
        if row['ranking'] != pool[:10] or not row['fallback'] or not row['error']:
            raise ValueError(f'Invalid fallback checkpoint {path.name}')
    elif status != 'valid' or row['fallback'] or row['error'] is not None:
        raise ValueError(f'Invalid status in checkpoint {path.name}')
    starter.validate_ranking(row['ranking'], pool)
    return row


def complete_llm(starter, directory, protocol_hash, users, prompts, pools):
    jobs = []
    for index, (uid, prompt, pool) in enumerate(zip(users, prompts, pools, strict=True), 1):
        checkpoint = directory / f'{index:03}.json'
        identity = {'position': index, 'user_id': uid, 'protocol_sha256': protocol_hash,
                    'candidate_pool_sha256': digest(pool)}
        saved = read_checkpoint(checkpoint, identity, starter, pool)
        jobs.append((checkpoint, identity, prompt, pool, saved))

    responses = []
    generator = None
    for checkpoint, identity, prompt, pool, saved in jobs:
        if saved is not None:
            responses.append(saved)
            continue
        if generator is None:
            generator = MPSGenerator()
        row = rank_one(starter, generator, prompt, pool)
        status = 'fallback' if row['fallback'] else 'valid'
        if row['ranking'] is None:
            status = 'technical_error'
        row.update(identity, status=status)
        atomic_json(checkpoint, row)
        if status == 'technical_error':
            error = row['error']
            raise RuntimeError(f"LLMRank stopped at case {identity['position']}/{len(users)} "
                               f"during {error['phase']}: {error['type']}: {error['message']}")
        print(f"LLMRank {identity['position']}/{len(users)}: {status} ({row['wall_seconds']:.2f}s)", flush=True)
        responses.append(row)
    return responses


def publish_evaluation(output, bundle, protocol, cases, pools, responses):
    starter, *_, manifest = bundle
    users = manifest['users']
    if len(responses) != COHORT_SIZE or [row['user_id'] for row in responses] != users or any(
        row['status'] not in ('valid', 'fallback') for row in responses
    ):
        raise ValueError(f'Final evaluation requires all {COHORT_SIZE} completed users in cohort order')

    final = output / 'evaluation'
    if final.exists():
        if any(not (final / name).is_file() for name in EVALUATION_FILES):
            raise ValueError('Incomplete evaluation directory')
        finished = json.loads((final / 'run_manifest.json').read_text(encoding='utf-8'))
        if finished['runtime']['protocol_sha256'] != protocol['protocol_sha256']:
            raise ValueError('Incompatible saved evaluation')
        print(f'Completed evaluation: {final}')
        return
    embeddings_path = output / 'embeddings.json'
    if embeddings_path.exists():
        embeddings = json.loads(embeddings_path.read_text(encoding='utf-8'))
        if embeddings['protocol_sha256'] != protocol['protocol_sha256']:
            raise ValueError('Incompatible saved embeddings')
        records, timing = embeddings['records'], embeddings['timing']
        validate_embeddings(starter, records, pools)
    else:
        cache = reranker.EmbeddingCache(output / 'embedding_cache')
        records, timing = reranker.rerank_cases(cases, cache, reranker.QwenEmbedder(), device='mps')
        validate_embeddings(starter, records, pools)
        atomic_json(embeddings_path, {'protocol_sha256': protocol['protocol_sha256'],
                                      'records': records, 'timing': timing})

    predictions = {uid: dict(record['rankings'], llmrank=response['ranking'])
                   for uid, record, response in zip(users, records, responses, strict=True)}
    failures = [{'user_id': row['user_id'], **row['error']} for row in responses if row['fallback']]
    metadata = dict(protocol['llmrank'], protocol_sha256=protocol['protocol_sha256'],
                    valid_responses=len(users) - len(failures), fallbacks=len(failures),
                    llmrank_wall_seconds=sum(row['wall_seconds'] for row in responses),
                    embeddings_timing=timing)
    with TemporaryDirectory(dir=output, prefix='.evaluation-') as folder:
        staging = Path(folder) / 'results'
        data_adapter.save_results(SimpleNamespace(output_dir=staging), bundle, predictions, failures,
                                  'hw3_evaluate', metadata)
        for name in EVALUATION_FILES:
            with (staging / name).open('rb+') as stream:
                os.fsync(stream.fileno())
        sync_directory(staging)
        os.replace(staging, final)
        sync_directory(output)


def evaluate(data_dir, output, *, resume):
    if not resume:
        output.mkdir(parents=True)
        sync_directory(output.parent)
    elif not output.is_dir():
        raise ValueError('--resume requires the existing evaluate output directory')

    with (output / '.lock').open('a', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process is using this output directory') from exc
        bundle = load_bundle(data_dir)
        starter, protocol, cases, prompts, pools = prepare_inputs(bundle, full=True)
        manifest_path = output / 'protocol.json'
        checkpoints = output / 'llmrank_checkpoints'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding='utf-8')) != protocol:
                raise ValueError('Incompatible protocol; use the original inputs/settings or a new output directory')
        else:
            if any(checkpoints.glob('*.json')) or any(
                (output / name).exists() for name in ('embeddings.json', 'evaluation')
            ):
                raise ValueError('Missing protocol for saved results')
            checkpoints.mkdir(exist_ok=True)
            atomic_json(manifest_path, protocol)
        if not checkpoints.is_dir():
            raise ValueError('Missing LLMRank checkpoint directory')

        users = bundle[-1]['users']
        responses = complete_llm(starter, checkpoints, protocol['protocol_sha256'], users, prompts, pools)
        publish_evaluation(output, bundle, protocol, cases, pools, responses)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'evaluate'])
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true', help='Continue an existing evaluate run')
    args = parser.parse_args(argv)
    default_name = 'full500' if args.command == 'evaluate' else 'smoke'
    output = (args.output_dir or OUTPUTS / default_name).resolve()
    if output.is_relative_to(HW2) or output.is_relative_to(ROOT / 'shared'):
        parser.error('Choose an output directory outside HW2 and shared')
    if args.command == 'evaluate':
        try:
            evaluate(args.data_dir, output, resume=args.resume)
        except KeyboardInterrupt:
            print('Interrupted. Continue the same run with --resume.', file=sys.stderr)
            return 130
        except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
            return 1
        return 0
    if args.resume:
        parser.error('--resume requires evaluate')
    if output.exists():
        parser.error('Choose a fresh output directory')

    started = time.perf_counter()
    summary = {'status': 'failed', 'seconds': {}, 'valid_top10': {}, 'errors': [], 'mps': {}}
    output.mkdir(parents=True)
    phase, phase_started = 'prepare', started
    try:
        starter, protocol, cases, prompts, pools = prepare(args.data_dir)
        atomic_json(output / 'protocol.json', protocol)
        summary.update(prepared_users=len(cases), identical_candidate_pools=len(pools),
                       protocol_sha256=protocol['protocol_sha256'])

        summary['seconds'][phase] = time.perf_counter() - started
        phase = 'mps'
        phase_started = time.perf_counter()
        summary['mps'] = {'built': torch.backends.mps.is_built(),
                          'available': torch.backends.mps.is_available(), 'cpu_fallback': False}
        reranker.require_mps()
        probe = torch.ones(1, device='mps') + 1
        torch.mps.synchronize()
        summary['mps']['tensor_probe_passed'] = probe.device.type == 'mps' and probe.item() == 2
        if not summary['mps']['tensor_probe_passed']:
            raise RuntimeError('MPS tensor operation failed')
        summary['seconds'][phase] = time.perf_counter() - phase_started

        phase = 'embeddings'
        phase_started = time.perf_counter()
        cache = reranker.EmbeddingCache(output / 'embedding_cache')
        encoder = reranker.QwenEmbedder()
        records, timing = reranker.rerank_cases(cases, cache, encoder, device='mps')
        validate_embeddings(starter, records, pools)
        atomic_json(output / 'embeddings.json', {'records': records, 'timing': timing})
        summary['valid_top10'].update({name: len(records) for name in reranker.METHODS})
        summary['seconds'][phase] = time.perf_counter() - phase_started
        del encoder
        torch.mps.empty_cache()

        phase = 'llmrank'
        phase_started = time.perf_counter()
        generator = MPSGenerator()
        responses = rank_with_llm(starter, generator, prompts, pools)
        atomic_json(output / 'llmrank.json', {'runtime': generator.metadata, 'responses': responses})
        fallbacks = sum(row['fallback'] for row in responses)
        valid_top10 = sum(row['ranking'] is not None for row in responses)
        summary['llmrank'] = {'valid_responses': valid_top10 - fallbacks, 'fallbacks': fallbacks}
        summary['valid_top10']['llmrank'] = valid_top10
        summary['errors'].extend(dict(row['error'], method='llmrank', case=index)
                                 for index, row in enumerate(responses, 1)
                                 if row['error'] and not row['fallback'])
        summary['seconds'][phase] = time.perf_counter() - phase_started
        if not summary['errors']:
            summary['status'] = 'passed'
    except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
        summary['seconds'][phase] = time.perf_counter() - phase_started
        summary['errors'].append({'phase': phase, 'type': type(exc).__name__, 'message': str(exc)})
    summary['seconds']['total'] = time.perf_counter() - started
    atomic_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
