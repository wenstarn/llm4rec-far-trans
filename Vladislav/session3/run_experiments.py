"""Run the FAR-Trans HW3 paired comparison and its independent controls."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'session2'))
import run_embeddings as hw2
sys.path.insert(0, str(hw2.STARTER))
import starter
import encoder as teacher

COHORTS = {'regression100': 100, 'development2000': 2000}
CSV_NAMES = ('transactions.csv', 'asset_information.csv', 'customer_information.csv', 'close_prices.csv')


@dataclass
class Evaluation:
    assets: dict
    profiles: dict
    history: dict
    truth: dict
    end: str
    excluded: dict
    users: list
    popularity: dict
    data_hashes: dict
    preparation_seconds: float

    @classmethod
    def load(cls, data_dir):
        started = perf_counter()
        assets, profiles, history, truth, end, excluded = starter.load_data(
            data_dir, hw2.CUTOFF, hw2.HORIZON_DAYS)
        users = starter.select_users(profiles, history, 'warm', 2000, None)
        if len(users) != 2000:
            raise ValueError(f'Expected 2,000 warm users, found {len(users)}')
        popularity = starter.popularity_scores(history, assets)
        hashes = {name: sha256_file(data_dir / name) for name in CSV_NAMES}
        return cls(assets, profiles, history, truth, end, excluded, users, popularity, hashes,
                   perf_counter() - started)


def sha256_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fresh_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f'Use an empty output directory: {path}')


def protocol_for(official100, data, users):
    protocol = {k: v for k, v in official100.items() if k != 'protocol_hash'}
    protocol['users'] = users
    if (protocol['source_sha256'] != data.data_hashes or protocol['eligible_assets'] != sorted(data.assets)
            or protocol['cutoff'] != hw2.CUTOFF or protocol['end_exclusive'] != data.end
            or protocol['exclude_seen'] or protocol['k'] != 10):
        raise ValueError('Dataset or protocol differs from the verified starter')
    return dict(protocol, protocol_hash=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest())


def evaluate(out, data, protocol, predictions):
    """Write complete rankings and full-precision metrics after scoring finishes."""
    users = protocol['users']
    eligible = set(protocol['eligible_assets'])
    truth = {uid: data.truth.get(uid, set()) & eligible for uid in users}
    records = []
    metrics = {}
    with (out / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['method', 'user_id', 'rank', 'ISIN'])
        for method, rankings in predictions.items():
            if list(rankings) != users:
                raise ValueError(f'{method} did not score the protocol cohort in order')
            rows = []
            for uid, ranking in rankings.items():
                starter.validate_ranking(ranking, eligible)
                rows.append({'user_id': uid, 'method': method, 'relevant_count': len(truth[uid]),
                             'ndcg_at_10': starter.ndcg10(ranking, truth[uid])})
                writer.writerows((method, uid, rank, isin) for rank, isin in enumerate(ranking, 1))
            records.extend(rows)
            metrics[method] = starter.aggregate(rows)
    hw2.write_json(out / 'protocol.json', protocol)
    hw2.write_json(out / 'per_user.json', records)
    hw2.write_json(out / 'metrics.json', metrics)
    print(f'{out}: {json.dumps(metrics)}', flush=True)
    return records, metrics


def transpose(predictions, users, methods):
    return {name: {uid: predictions[uid][name] for uid in users} for name in methods}


def reference_control(data, users, original):
    """Use the instructor's history scorer and ISIN tie break for original text."""
    eligible = sorted(data.assets)
    rankings, fallbacks = {}, 0
    for uid in users:
        scores = teacher.history_scores(data.history[uid], original)
        if not any(scores.values()):
            scores = data.popularity
            fallbacks += 1
        rankings[uid] = sorted(eligible, key=lambda i: (-scores.get(i, 0.0), i))[:10]
    return rankings, fallbacks


def controls(args, data):
    fresh_dir(args.output_dir)
    timings = {'data_load_hashes_selection_popularity_seconds': data.preparation_seconds}
    started = perf_counter()
    features = {i: starter.tokens(a) for i, a in data.assets.items()}
    events = starter.scan_recent_events(args.data_dir, hw2.CUTOFF, data.assets)
    timings['starter_features_and_event_scan_seconds'] = perf_counter() - started
    started = perf_counter()
    contexts, baselines = starter.contexts_for(data.users, data.assets, data.profiles, data.history,
                                              features, data.popularity, hw2.CUTOFF, 50, False)
    timings['popularity_and_cbf_with_context_construction_seconds'] = perf_counter() - started
    started = perf_counter()
    recent = starter.recency_rankings(contexts, data.popularity, events)
    for uid in data.users:
        baselines[uid].update(recent[uid])
    timings['both_recency_baselines_seconds'] = perf_counter() - started
    args.cutoff, args.horizon_days = hw2.CUTOFF, hw2.HORIZON_DAYS
    users100 = data.users[:100]
    manifest = {'users': users100, 'end_exclusive': data.end, 'data_sha256': data.data_hashes,
                'repeat_allowed': True}
    bundle = (starter, data.assets, data.history, data.truth, contexts, baselines, events,
              data.popularity, manifest)
    started = perf_counter()
    official, verified = hw2.verify_official(args, bundle)
    reference_hash = hw2.verify_ivan(verified, hw2.IVAN / 'results/table_results.csv')
    hw2.verify_ivan(verified, HERE.parent / 'session2/table_results.csv')
    timings['independent_100_user_starter_verification_seconds'] = perf_counter() - started
    started = perf_counter()
    original = teacher.tfidf({i: teacher.source_text(data.assets[i]) for i in sorted(data.assets)})
    timings['original_text_preparation_seconds'] = perf_counter() - started
    started = perf_counter()
    control, cold = reference_control(data, data.users, original)
    timings['original_text_scoring_seconds'] = perf_counter() - started
    for cohort, size in COHORTS.items():
        users = data.users[:size]
        out = args.output_dir / cohort
        fresh_dir(out)
        protocol = protocol_for(official, data, users)
        if size == 100 and protocol != official:
            raise ValueError('Regression protocol differs from the starter run')
        predictions = transpose(baselines, users, hw2.BASELINES)
        if size == 2000:
            predictions['original_text_cbf'] = control
        evaluate(out, data, protocol, predictions)
    coverage = {'classical_invalid_rankings': 0, 'original_text_invalid_rankings': 0,
                'cbf_popularity_fallback_users': sum(context['cold_start'] for context in contexts.values()),
                'original_text_popularity_fallback_users': cold,
                'recency_users_with_popularity_fill': {
                    'latest_buy': sum(len(data.history[uid]) < 10 for uid in data.users),
                    'latest_buy_or_sell': sum(len({isin for _, isin, _ in events[uid]}) < 10
                                              for uid in data.users)}}
    hw2.write_json(args.output_dir / 'run.json', {'timings': timings, 'coverage': coverage,
                                                 'excluded': data.excluded,
                                                 'ivan_reference_csv_sha256': reference_hash})


def paired(records, data, users):
    indexed = {(r['user_id'], r['method']): r for r in records}
    short = [uid for uid in users if 1 <= len(data.history[uid]) <= 3]
    nonempty = [uid for uid in short if indexed[uid, 'one_hot']['relevant_count']]
    deltas = {uid: indexed[uid, 'qwen3']['ndcg_at_10'] - indexed[uid, 'one_hot']['ndcg_at_10']
              for uid in users}

    def summarize(sample):
        values = [deltas[uid] for uid in sample]
        return {'users': len(values), 'mean_paired_difference': sum(values) / len(values) if values else None,
                'wins': sum(x > 0 for x in values), 'losses': sum(x < 0 for x in values),
                'ties': sum(x == 0 for x in values)}

    summary = {'short_history': summarize(short), 'short_history_nonempty': summarize(nonempty),
               'methods': {name: starter.aggregate([indexed[uid, name] for uid in short]) for name in hw2.METHODS},
               'history_size_counts': {n: sum(len(data.history[uid]) == n for uid in short) for n in (1, 2, 3)}}
    return summary, deltas, nonempty, indexed


def save_cases(out, data, users, ids, rows, scores, records):
    summary, deltas, short_nonempty, indexed = paired(records, data, users)
    hw2.write_json(out / 'short_history.json', summary)
    with_truth = [uid for uid in users if data.truth.get(uid)]
    if not with_truth:
        return
    user_index = {uid: i for i, uid in enumerate(users)}
    asset_index = {isin: i for i, isin in enumerate(ids)}
    ordered = list(dict.fromkeys(short_nonempty + with_truth))

    def pick_delta(sign):
        return next((uid for uid in ordered if sign * deltas[uid] > 0), None)

    def full_order(values):
        return sorted(range(len(ids)), key=lambda i: (-values[i], -data.popularity[ids[i]], ids[i]))

    miss = None
    for uid in ordered:
        values = scores['qwen3'][user_index[uid]]
        ranking = full_order(values)
        # Prefer an observable tie spanning the Top-10 boundary.
        boundary_rows = {rows[i] for i in ranking[:10] if values[i] == values[ranking[9]]}
        outside = [i for i in ranking[10:] if ids[i] in data.truth[uid]]
        if outside and miss is None:
            miss = uid
        if any(rows[i] in boundary_rows for i in outside):
            miss = uid
            break

    hand = next((uid for uid in short_nonempty if deltas[uid]), ordered[0])
    cases = {}
    aliases = {}
    for label, uid in (('qwen_helps', pick_delta(1)), ('qwen_hurts', pick_delta(-1)),
                       ('qwen_misses', miss), ('hand_check', hand)):
        if uid is None:
            continue
        pseudonym = aliases.setdefault(uid, f'Case {len(aliases) + 1:02d}')
        truth = sorted(data.truth[uid])
        method_lists = {}
        for method, matrix in scores.items():
            values = matrix[user_index[uid]]
            ranking = full_order(values)
            score_rows = [{'rank': rank, 'ISIN': ids[i], 'score': float(values[i]),
                           'past_buyers': data.popularity[ids[i]],
                           'fields': dict(zip(hw2.FIELDS, rows[i], strict=True))}
                          for rank, i in enumerate(ranking, 1)]
            ranks = {row['ISIN']: row['rank'] for row in score_rows}
            score_path = f'{pseudonym.lower().replace(" ", "-")}-{method}-scores.json'
            hw2.write_json(out / score_path, score_rows)
            method_lists[method] = {'top10': [r['ISIN'] for r in score_rows[:10]],
                                    'truth': [score_rows[ranks[isin] - 1] for isin in truth],
                                    'rank10': score_rows[9],
                                    'ndcg_at_10': indexed[uid, method]['ndcg_at_10'],
                                    'score_file': score_path}
        cases[label] = {'pseudonym': pseudonym, 'truth': truth, 'methods': method_lists,
                        'history': [{'ISIN': isin, 'metadata_timestamp': data.assets[isin]['timestamp'],
                                     'fields': dict(zip(hw2.FIELDS, rows[asset_index[isin]], strict=True))}
                                    for isin in sorted(data.history[uid])]}
    hw2.write_json(out / 'cases.json', cases)
    hw2.write_json(out / 'case_customer_ids.json', {alias: uid for uid, alias in aliases.items()})


def main_comparison(args, data):
    fresh_dir(args.output_dir)
    timings = {'data_load_hashes_selection_popularity_seconds': data.preparation_seconds}
    ids = sorted(data.assets)
    started = perf_counter()
    rows = [hw2.feature_values(data.assets[isin]) for isin in ids]
    unique = list(dict.fromkeys(rows))
    row_index = {row: i for i, row in enumerate(unique)}
    asset_rows = [row_index[row] for row in rows]
    index = dict(zip(ids, asset_rows, strict=True))
    histories = [[index[i] for i in sorted(data.history[uid])] for uid in data.users]
    timings['four_field_tuple_and_history_preparation_seconds'] = perf_counter() - started
    started = perf_counter()
    one_hot = hw2.one_hot_matrix(unique)
    timings['one_hot_representation_seconds'] = perf_counter() - started
    matrices = {'one_hot': one_hot}
    runtime = {}
    if not args.one_hot_only:
        started = perf_counter()
        qwen, runtime = hw2.encode_qwen([hw2.description(row) for row in unique], args.cache_dir)
        timings['qwen_representation_wall_seconds'] = perf_counter() - started
        matrices['qwen3'] = qwen
    official = hw2.read_json(args.controls_dir / 'regression100/protocol.json')
    for cohort, size in COHORTS.items():
        out = args.output_dir / cohort
        fresh_dir(out)
        users = data.users[:size]
        protocol = protocol_for(official, data, users)
        if protocol != hw2.read_json(args.controls_dir / cohort / 'protocol.json'):
            raise ValueError(f'Control protocol mismatch for {cohort}')
        scores, predictions = {}, {}
        for method, matrix in matrices.items():
            started = perf_counter()
            values = hw2.centroid_scores(matrix, histories[:size])[:, asset_rows]
            rankings = {uid: hw2.top10(ids, row, data.popularity) for uid, row in zip(users, values, strict=True)}
            timings[f'{cohort}_{method}_scoring_seconds'] = perf_counter() - started
            scores[method], predictions[method] = values, rankings
        records, metrics = evaluate(out, data, protocol, predictions)
        if size == 100:
            with (HERE.parent / 'session2/table_results.csv').open(newline='') as stream:
                prior = next(row for row in csv.DictReader(stream) if row['method'] == 'one_hot')
            if any(metrics['one_hot'][key] != float(prior[key]) for key in metrics['one_hot']):
                raise ValueError('One-hot differs from the saved HW2 row at full precision')
        if 'qwen3' in scores and size == 2000:
            save_cases(out, data, users, ids, rows, scores, records)
        if size == 2000 and args.one_hot_only:
            short = [r for r in records if 1 <= len(data.history[r['user_id']]) <= 3]
            hw2.write_json(out / 'short_history.json', starter.aggregate(short))
    hw2.write_json(args.output_dir / 'run.json', dict(runtime, timings=timings, unique_tuples=len(unique),
                   asset_count=len(ids), invalid_rankings=0, model_fallbacks=0,
                   metadata_nonmissing_counts={field: sum(row[i] != hw2.MISSING for row in rows)
                                               for i, field in enumerate(hw2.FIELDS)},
                   one_hot_hw2_full_precision_match=True))


def embedding_smoke(args):
    fresh_dir(args.output_dir)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    vectors, runtime = hw2.encode_qwen([hw2.description(('Equity', 'Stock', 'Energy', 'Oil & Gas')),
                                        hw2.description(('Bond', hw2.MISSING, hw2.MISSING, hw2.MISSING))],
                                       args.output_dir / 'smoke_cache')
    if not runtime['encoder']['model_loaded'] or vectors.shape != (2, 1024):
        raise RuntimeError('Smoke test must encode two descriptions with the real model')
    hw2.write_json(args.output_dir / 'forward_pass.json', dict(runtime, shape=list(vectors.shape),
                   norms=np.linalg.norm(vectors, axis=1).tolist(), cosine=float(vectors[0] @ vectors[1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('controls', 'main', 'embedding-smoke'))
    parser.add_argument('--data-dir', type=Path, default=hw2.ROOT / 'data/FAR-Trans')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--controls-dir', type=Path, default=HERE / 'outputs/controls')
    parser.add_argument('--cache-dir', type=Path, default=HERE / 'outputs/four_field_embeddings')
    parser.add_argument('--one-hot-only', action='store_true')
    args = parser.parse_args()
    if args.stage == 'embedding-smoke':
        embedding_smoke(args)
    elif args.stage == 'controls':
        controls(args, Evaluation.load(args.data_dir))
    else:
        main_comparison(args, Evaluation.load(args.data_dir))


if __name__ == '__main__':
    main()
