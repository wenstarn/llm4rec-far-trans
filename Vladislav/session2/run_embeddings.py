"""Compare four-field one-hot and frozen Qwen3 on the FAR-Trans homework cohort."""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STARTER = ROOT / 'shared/far-trans'
IVAN = ROOT / 'Ivan/session2'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'shared/far-trans-llm'))
import data_adapter

CUTOFF = '2021-06-01'
HORIZON_DAYS = 30
MAX_USERS = 100
METHODS = ('one_hot', 'qwen3')
BASELINES = ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell')
FIELDS = {
    'assetCategory': 'Category', 'assetSubCategory': 'Subcategory',
    'sector': 'Sector', 'industry': 'Industry',
}
MISSING = '[MISSING]'
TEMPLATE = 'far-trans-four-fields-v1'


def feature_values(asset):
    values = (asset.get(field, '').strip() for field in FIELDS)
    return tuple(MISSING if value.casefold() in ('', 'not_available', 'nan') else value
                 for value in values)


def description(values):
    # JSON string literals keep the labeled lines unambiguous even for embedded newlines.
    return '\n'.join(f'{label}: {json.dumps(value, ensure_ascii=False)}'
                     for label, value in zip(FIELDS.values(), values, strict=True))


def one_hot_matrix(feature_rows):
    # Field indices keep a label in one field distinct from that label in any other field.
    vocabulary = sorted({(field, value) for row in feature_rows for field, value in enumerate(row)})
    columns = {key: index for index, key in enumerate(vocabulary)}
    vectors = np.zeros((len(feature_rows), len(columns)))
    for index, row in enumerate(feature_rows):
        vectors[index, [columns[field, value] for field, value in enumerate(row)]] = 1
    return vectors


def centroid_scores(vectors, history_indices):
    """One cosine implementation for both arms, with one vote per history ISIN."""
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError('Expected a finite asset-by-feature matrix')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError('Expected nonzero asset vectors with finite norms')
    unit = vectors / norms
    centroids = np.stack([
        unit[indices].mean(axis=0) if indices else np.zeros(unit.shape[1])
        for indices in history_indices
    ])
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    # Zero centroids score every asset equally, resolving through the popularity tie break.
    normalized = np.divide(centroids, norms, out=np.zeros_like(centroids), where=norms > 1e-12)
    return normalized @ unit.T


def top10(ids, scores, popularity):
    if len(ids) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected one finite score per eligible asset')
    order = sorted(range(len(ids)), key=lambda i: (-scores[i], -popularity[ids[i]], ids[i]))
    return [ids[i] for i in order[:10]]


def encode_qwen(texts, cache_dir):
    # Ivan's benchmark sets HF_HOME on import. Set it first to keep his tree untouched
    # and use the ordinary Hugging Face cache unless the caller supplies HF_HOME.
    cache_home = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    os.environ.setdefault('HF_HOME', str(cache_home / 'huggingface'))
    sys.path.insert(0, str(IVAN))
    import embedding_reranker as ivan

    metadata = dict(ivan.cache_metadata(), features=list(FIELDS.items()), template=TEMPLATE)
    # The cache round-trips with torch.save, while protocol JSON represents these as arrays.
    cache = ivan.EmbeddingCache(cache_dir, metadata)
    encoder = ivan.QwenEmbedder()
    stats = cache.ensure(texts, encoder.encode)
    return cache.get(texts).numpy(), {
        'cache': stats, 'cache_file': str(cache.path), 'cache_metadata': metadata,
        'encoder': encoder.metadata,
    }


def predict(starter, users, assets, history, popularity, encode):
    """The predictor receives only as-of assets, distinct Buy histories and popularity."""
    ids = sorted(assets)
    feature_rows = [feature_values(assets[isin]) for isin in ids]
    unique_rows = list(dict.fromkeys(feature_rows))
    row_index = {row: i for i, row in enumerate(unique_rows)}
    asset_rows = [row_index[row] for row in feature_rows]
    index = dict(zip(ids, asset_rows, strict=True))
    # Equal metadata share one score, but each distinct historical ISIN keeps its vote.
    histories = [[index[isin] for isin in sorted(history[uid])] for uid in users]
    predictions = {uid: {} for uid in users}
    timings = {}

    def rank_arm(name, matrix):
        started = time.perf_counter()
        if len(matrix) != len(unique_rows):
            raise ValueError(f'{name} must return one vector per unique description')
        scores = centroid_scores(matrix, histories)[:, asset_rows]
        for uid, user_scores in zip(users, scores, strict=True):
            predictions[uid][name] = starter.validate_ranking(
                top10(ids, user_scores, popularity), ids)
        timings[f'{name}_scoring_seconds'] = time.perf_counter() - started

    started = time.perf_counter()
    one_hot = one_hot_matrix(unique_rows)
    timings['one_hot_representation_seconds'] = time.perf_counter() - started
    rank_arm('one_hot', one_hot)

    started = time.perf_counter()
    qwen, runtime = encode([description(row) for row in unique_rows])
    timings['qwen3_representation_seconds'] = time.perf_counter() - started
    rank_arm('qwen3', qwen)

    coverage = {field: sum(row[i] != MISSING for row in feature_rows)
                for i, field in enumerate(FIELDS)}
    return predictions, dict(runtime, timing=timings, unique_descriptions=len(unique_rows),
                             metadata_nonmissing_counts=coverage)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_ivan(metrics, reference_csv):
    source = reference_csv.read_bytes()
    reference = {row['method']: row for row in csv.DictReader(source.decode('utf-8').splitlines())}
    fields = ('users', 'nonempty_truth_users', 'empty_truth_users',
              'ndcg_at_10', 'ndcg_at_10_nonempty')
    for name in BASELINES:
        for field in fields:
            expected = float(reference[name][field])
            if metrics[name][field] != expected:
                raise ValueError(f'Ivan baseline mismatch for {name}.{field}: '
                                 f'{metrics[name][field]} != {expected}')
    return hashlib.sha256(source).hexdigest()


def verify_official(args, bundle):
    """Run the unmodified starter CLI and compare its complete baseline outputs."""
    starter, assets, history, truth, contexts, baseline, _, _, manifest = bundle
    users = manifest['users']
    if len(users) != MAX_USERS:
        raise ValueError(f'Expected {MAX_USERS} warm users, found {len(users)}')
    eligible = sorted(assets)
    if any(contexts[uid]['cutoff'] != CUTOFF
           or contexts[uid]['eligible_asset_ids'] != eligible
           or not set(history[uid]) <= set(eligible) for uid in users):
        raise ValueError('Starter contexts differ from the full repeat-eligible universe')

    with tempfile.TemporaryDirectory(prefix='far-trans-official-') as directory:
        output = Path(directory)
        subprocess.run([sys.executable, '-B', str(STARTER / 'starter.py'), 'run',
                        '--data-dir', str(args.data_dir), '--output-dir', str(output)],
                       check=True)
        official = read_json(output / 'protocol.json')
        payload = {key: value for key, value in official.items() if key != 'protocol_hash'}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if official['protocol_hash'] != digest:
            raise ValueError('Invalid official starter protocol hash')
        expected = {
            'users': users, 'cutoff': CUTOFF, 'end_exclusive': manifest['end_exclusive'],
            'eligible_assets': eligible, 'exclude_seen': False,
            'source_sha256': manifest['data_sha256'], 'k': 10, 'cohort_selection': 'warm',
        }
        end = (date.fromisoformat(CUTOFF) + timedelta(days=HORIZON_DAYS)).isoformat()
        if not manifest['repeat_allowed'] or manifest['end_exclusive'] != end:
            raise ValueError('Adapter repeat policy or label window differs from homework')
        for key, value in expected.items():
            if official[key] != value:
                raise ValueError(f'Official starter protocol mismatch for {key}')
        with (output / 'predictions.csv').open(newline='', encoding='utf-8') as stream:
            actual_rows = list(csv.reader(stream))
        expected_rows = [['method', 'user_id', 'rank', 'ISIN']] + [
            [method, uid, str(rank), isin] for uid in users
            for method in BASELINES for rank, isin in enumerate(baseline[uid][method], 1)
        ]
        if actual_rows != expected_rows:
            raise ValueError('Baseline rankings differ from the official starter')

        result = read_json(output / 'metrics.json')
        if result['protocol_hash'] != digest or set(result['methods']) != set(BASELINES):
            raise ValueError('Unexpected official starter metrics')
        metrics = result['methods']
        for method in BASELINES:
            records = [{'ndcg_at_10': starter.ndcg10(baseline[uid][method], truth.get(uid, set())),
                        'relevant_count': len(truth.get(uid, set()))} for uid in users]
            if starter.aggregate(records) != metrics[method]:
                raise ValueError(f'Official starter metric mismatch for {method}')
    return official, metrics


def paired_summary_and_case(bundle, predictions, records):
    starter, assets, history, truth, contexts, _, events, _, manifest = bundle
    indexed = {(row['user_id'], row['method']): row for row in records}
    short = [uid for uid in manifest['users'] if 1 <= len(history[uid]) <= 3]
    with_truth = [uid for uid in short if indexed[uid, METHODS[0]]['relevant_count']]
    pairs = [(indexed[uid, METHODS[0]]['ndcg_at_10'], indexed[uid, METHODS[1]]['ndcg_at_10'])
             for uid in short]
    deltas = [qwen - one_hot for one_hot, qwen in pairs]
    summary = {
        'users': len(short), 'nonempty_truth_users': len(with_truth),
        'users_by_history_size': {str(n): sum(len(history[uid]) == n for uid in short)
                                  for n in (1, 2, 3)},
        'methods': {name: starter.aggregate([indexed[uid, name] for uid in short])
                    for name in METHODS},
        'mean_paired_qwen3_minus_one_hot': sum(deltas) / len(deltas) if deltas else None,
        'qwen3_wins': sum(delta > 0 for delta in deltas),
        'one_hot_wins': sum(delta < 0 for delta in deltas),
        'ties': sum(delta == 0 for delta in deltas),
        'interpretation': 'Descriptive paired result on users with 1 to 3 distinct past buys.',
    }

    # Prefer a short-history user with truth and a differential hit for hand calculation.
    users = manifest['users']
    eligible = [uid for uid in users if indexed[uid, METHODS[0]]['relevant_count']]
    uid = next((uid for uid in with_truth
                if indexed[uid, 'one_hot']['ndcg_at_10'] != indexed[uid, 'qwen3']['ndcg_at_10']),
               (with_truth or eligible or users)[0])
    relevant = sorted(truth.get(uid, set()) & set(contexts[uid]['eligible_asset_ids']))
    bought = sorted(history[uid])
    latest = {}
    for day, isin, kind in events.get(uid, ()):
        if kind == 'Buy' and isin in history[uid]:
            latest[isin] = max(day, latest.get(isin, ''))
    case = {
        'case_id': 'MANUAL-01', 'cutoff': manifest['cutoff'],
        'end_exclusive': manifest['end_exclusive'], 'truth': relevant,
        'distinct_past_buys': len(bought),
        'history': [{'ISIN': isin, 'last_buy_date': latest[isin],
                     'metadata_timestamp': assets[isin]['timestamp'],
                     'fields': dict(zip(FIELDS, feature_values(assets[isin]), strict=True))}
                    for isin in bought],
        'rankings': {name: [{'rank': rank, 'ISIN': isin, 'hit': isin in relevant}
                            for rank, isin in enumerate(predictions[uid][name], 1)]
                     for name in METHODS},
        'ndcg_at_10': {name: indexed[uid, name]['ndcg_at_10'] for name in METHODS},
    }
    return summary, case


def execute(args, bundle, encode, *, verified=None, timings=None):
    starter, assets, history, _, _, _, _, popularity, manifest = bundle
    all_users = manifest['users']
    smoke = args.command == 'smoke'
    if not all_users:
        raise ValueError('No warm users selected')
    # Three ordered users, preferring the first histories of sizes 1, 2, and 3.
    if smoke:
        strata = [next((uid for uid in all_users if len(history[uid]) == n), None)
                  for n in (1, 2, 3)]
        users = list(dict.fromkeys([uid for uid in strata if uid is not None] + all_users))[:3]
    else:
        users = all_users
    predictions, runtime = predict(starter, users, assets, history, popularity, encode)
    runtime['timing'].update(timings or {})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        write_json(args.output_dir / 'rankings.json', [
            {'case_id': f'case_{i}', 'distinct_past_buys': len(history[uid]),
             'rankings': predictions[uid]} for i, uid in enumerate(users, 1)
        ])
    else:
        official, baseline_metrics, reference_hash = verified
        data_adapter.save_results(args, bundle, predictions, [], 'asset_representation', runtime)
        metrics = read_json(args.output_dir / 'metrics.json')
        if any(metrics[name] != baseline_metrics[name] for name in BASELINES):
            raise ValueError('Saved baseline metrics differ from the verified starter metrics')
        records = read_json(args.output_dir / 'per_user.json')
        summary, case = paired_summary_and_case(bundle, predictions, records)
        write_json(args.output_dir / 'short_history.json', summary)
        write_json(args.output_dir / 'manual_case.json', case)
        print('Short-history paired result')
        print(json.dumps(summary, indent=2))

    protocol = {
        'mode': args.command, 'cutoff': CUTOFF, 'horizon_days': HORIZON_DAYS,
        'end_exclusive': manifest['end_exclusive'], 'cohort_users': len(all_users),
        'scored_users': len(users), 'repeat_allowed': manifest['repeat_allowed'],
        'candidate_policy': 'full starter eligible universe for both encoder arms',
        'eligible_asset_count': len(assets),
        'data_sha256': manifest['data_sha256'], 'methods': list(METHODS),
        'future_labels_used_for_scoring': False, 'future_labels_used_for_evaluation': not smoke,
        'features': list(FIELDS), 'description_template': TEMPLATE, 'missing_value': MISSING,
        'history': 'all distinct eligible pre-cutoff Buy ISINs retained by the starter',
        'scoring': 'normalize asset vectors, mean history vectors, cosine to centroid in FP64',
        'tie_break': 'descending full-precision cosine, descending starter popularity, ascending ISIN',
        'zero_centroid_policy': 'equal scores, then popularity and ISIN',
        'invalid_output_policy': 'error', 'runtime': runtime,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'numpy': np.__version__},
        'source_sha256': {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), STARTER / 'starter.py', Path(data_adapter.__file__),
                         IVAN / 'embedding_reranker.py', IVAN / 'experiments/benchmark_embeddings.py')
        },
    }
    if not smoke:
        protocol.update(users=users, eligible_assets=sorted(assets), official_starter=official,
                        ivan_baselines_match=reference_hash is not None,
                        ivan_baseline_csv_sha256=reference_hash)
    write_json(args.output_dir / 'protocol.json', protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'evaluate'])
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=HERE / 'outputs/four_field_embeddings')
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = HERE / 'outputs' / args.command
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error('Output directory must be empty. Choose another --output-dir.')
    args.starter_dir = STARTER
    args.cutoff, args.horizon_days, args.max_users = CUTOFF, HORIZON_DAYS, MAX_USERS
    args.candidate_count = 50  # Official CBF context only. Both encoder arms use all eligible assets.
    started = time.perf_counter()
    bundle = data_adapter.load(args)
    timings = {'shared_loader_seconds': time.perf_counter() - started}
    verified = None
    if args.command == 'evaluate':
        verification_started = time.perf_counter()
        official, baseline_metrics = verify_official(args, bundle)
        reference_hash = verify_ivan(baseline_metrics, IVAN / 'results/table_results.csv')
        verified = official, baseline_metrics, reference_hash
        timings['baseline_verification_seconds'] = time.perf_counter() - verification_started
    execute(args, bundle, lambda texts: encode_qwen(texts, args.cache_dir),
            verified=verified, timings=timings)
    print(f'{args.command} completed in {time.perf_counter() - started:.2f}s. Outputs: {args.output_dir}')


if __name__ == '__main__':
    main()
