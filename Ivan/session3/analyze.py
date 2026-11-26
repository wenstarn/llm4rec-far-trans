"""Audit a completed HW3 run and summarize failures without model inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from math import isclose
from pathlib import Path
from time import perf_counter

import run


METHODS = ('cbf', 'MaxSim', 'MeanSim', 'llmrank', 'popularity', 'latest_buy', 'latest_buy_or_sell')
GROUPS = {'1': (1,), '2–3': (2, 3), '4–5': (4, 5)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_predictions(path):
    rankings = defaultdict(list)
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            ranking = rankings[row['user_id'], row['method']]
            require(int(row['rank']) == len(ranking) + 1, 'Unexpected rank order in predictions.csv')
            ranking.append(row['ISIN'])
    return rankings


def load_verified(run_dir, data_dir):
    protocol = read_json(run_dir / 'protocol.json')
    evaluation = run_dir / 'evaluation'
    manifest = read_json(evaluation / 'run_manifest.json')
    runtime = manifest['runtime']
    embeddings = read_json(run_dir / 'embeddings.json')
    protocol_hash = protocol['protocol_sha256']
    require(protocol_hash == run.digest({k: v for k, v in protocol.items() if k != 'protocol_sha256'}),
            'Invalid protocol digest')
    require(protocol_hash == embeddings['protocol_sha256'] == runtime['protocol_sha256'],
            'Embeddings and evaluation refer to different protocols')
    require(all(runtime[key] == value for key, value in protocol['llmrank'].items()),
            'Different LLMRank settings in the evaluation manifest')

    started = perf_counter()
    bundle = run.load_bundle(data_dir)
    starter, reconstructed, cases, _, pools = run.prepare_inputs(bundle, full=True)
    reconstruction_seconds = perf_counter() - started
    # The local archive predates the directory rename. The runner bytes must still match.
    archived_runner, current_runner = 'Ivan/hw3/run.py', 'Ivan/session3/run.py'
    sources = reconstructed['source_sha256']
    if archived_runner in protocol['source_sha256'] and current_runner in sources:
        sources[archived_runner] = sources.pop(current_runner)
        reconstructed.pop('protocol_sha256')
        reconstructed['protocol_sha256'] = run.digest(reconstructed)
    require(reconstructed == protocol, 'Source, data or reconstructed protocol differs from the archived run')
    _, _, history, truth, contexts, baselines, _, _, source_manifest = bundle
    require(all(manifest[key] == value for key, value in source_manifest.items()),
            'Evaluation manifest differs from the reconstructed cohort or data policy')
    users = manifest['users']
    records = embeddings['records']
    run.validate_embeddings(starter, records, pools)
    checkpoint_dir = run_dir / 'llmrank_checkpoints'
    require(len(list(checkpoint_dir.glob('*.json'))) == len(users), 'Unexpected number of LLMRank checkpoints')

    saved_scores = read_json(evaluation / 'per_user.json')
    grades = {(row['user_id'], row['method']): row for row in saved_scores}
    predictions = read_predictions(evaluation / 'predictions.csv')
    expected_keys = {(uid, method) for uid in users for method in METHODS}
    require(len(saved_scores) == len(expected_keys) and grades.keys() == predictions.keys() == expected_keys,
            'Predictions and per-user metrics must cover every method and user exactly once')

    rows, checkpoints = [], []
    for index, (uid, case, pool, record) in enumerate(zip(users, cases, pools, records, strict=True), 1):
        require(record['history_count'] == len(case['history']), 'Different history length in embeddings archive')
        identity = {'position': index, 'user_id': uid, 'protocol_sha256': protocol_hash,
                    'candidate_pool_sha256': run.digest(pool)}
        checkpoint = run.read_checkpoint(checkpoint_dir / f'{index:03}.json', identity, starter, pool)
        require(checkpoint is not None, 'Incomplete LLMRank checkpoint')
        checkpoints.append(checkpoint)
        rankings = dict(baselines[uid], **record['rankings'], llmrank=checkpoint['ranking'])
        relevant = truth.get(uid, set()) & set(contexts[uid]['eligible_asset_ids'])
        scores = {}
        for method in METHODS:
            ranking = rankings[method]
            grade = grades[uid, method]
            require(predictions[uid, method] == ranking, 'Saved prediction disagrees with method output')
            require(grade['relevant_count'] == len(relevant) and isclose(
                grade['ndcg_at_10'], starter.ndcg10(ranking, relevant), abs_tol=1e-12
            ), 'Saved nDCG or target count disagrees with data and prediction')
            scores[method] = grade['ndcg_at_10']
        rows.append({'history_count': record['history_count'], 'prior_buys': set(history[uid]),
                     'relevant': relevant, 'pool': pool, 'rankings': rankings, 'scores': scores})

    failures = [{'user_id': row['user_id'], **row['error']} for row in checkpoints if row['fallback']]
    require(failures == manifest['failures'], 'LLMRank fallback log disagrees with checkpoints')
    require(runtime['fallbacks'] == len(failures) and runtime['valid_responses'] == len(users) - len(failures),
            'LLMRank counts in the manifest disagree with checkpoints')
    require(runtime['embeddings_timing'] == embeddings['timing'], 'Conflicting embedding times')
    require(isclose(runtime['llmrank_wall_seconds'], sum(row['wall_seconds'] for row in checkpoints)),
            'Conflicting LLMRank times')
    require(isclose(embeddings['timing']['scoring_end_to_end_seconds'],
                    sum(row['scoring_end_to_end_seconds'] for row in records)), 'Conflicting scoring times')
    return protocol, rows, checkpoints, embeddings['timing'], reconstruction_seconds


def comparison(rows):
    counts = dict(better=0, worse=0, equal=0)
    for row in rows:
        if not row['relevant']:
            continue
        left, right = (row['scores'][method] for method in ('MaxSim', 'MeanSim'))
        outcome = 'equal' if isclose(left, right, abs_tol=1e-12) else 'better' if left > right else 'worse'
        counts[outcome] += 1
    return counts


def group_metrics(rows):
    nonempty = [row for row in rows if row['relevant']]
    return {'users': len(rows), 'nonempty_users': len(nonempty),
            'maxsim_vs_meansim_nonempty': comparison(nonempty),
            'maxsim_meansim_identical_top10_users': sum(
                row['rankings']['MaxSim'] == row['rankings']['MeanSim'] for row in rows
            ),
            'ndcg_at_10': {
                method: {'all_users': sum(row['scores'][method] for row in rows) / len(rows) if rows else None,
                         'nonempty': sum(row['scores'][method] for row in nonempty) / len(nonempty)
                         if nonempty else None}
                for method in METHODS
            }}


def candidate_coverage(rows):
    covered = [row['relevant'] & set(row['pool']) for row in rows]
    missed = [row['relevant'] - set(row['pool']) for row in rows]
    return {'relevant_user_asset_pairs': sum(len(row['relevant']) for row in rows),
            'covered_pairs': sum(map(len, covered)), 'missing_pairs': sum(map(len, missed)),
            'unique_assets_missing_for_at_least_one_user': len(set().union(*missed)),
            'users_with_any_missing': sum(map(bool, missed)),
            'users_with_all_missing': sum(bool(miss and not hit) for miss, hit in zip(missed, covered, strict=True)),
            'users_with_any_relevant_candidate': sum(map(bool, covered)),
            'users_with_relevant_candidate_but_both_embeddings_ndcg_zero': sum(
                bool(hit) and row['scores']['MaxSim'] == row['scores']['MeanSim'] == 0
                for hit, row in zip(covered, rows, strict=True)
            )}


def repeats(rows):
    by_method = {}
    for method in METHODS:
        recommended = [set(row['rankings'][method]) & row['prior_buys'] for row in rows]
        count = sum(map(len, recommended))
        slots = sum(len(row['rankings'][method]) for row in rows)
        by_method[method] = {
            'repeat_slots': count, 'total_slots': slots, 'repeat_share': count / slots,
            'users_with_repeat': sum(map(bool, recommended)),
            'relevant_repeat_hits': sum(len(items & row['relevant'])
                                        for items, row in zip(recommended, rows, strict=True)),
        }
    return {'definition': 'ISIN with an eligible Buy before cutoff in the full user history',
            'relevant_repeat_pairs': sum(len(row['relevant'] & row['prior_buys']) for row in rows),
            'methods': by_method}


def llm_summary(checkpoints):
    statuses = Counter(row['status'] for row in checkpoints)
    reasons = Counter('Context budget exceeded' if row['error']['phase'] == 'context' else row['error']['message']
                      for row in checkpoints if row['fallback'])
    return {'users': len(checkpoints), 'generation_attempts': sum(row['text'] is not None for row in checkpoints),
            'valid_responses': statuses['valid'], 'fallbacks': statuses['fallback'],
            'accepted_after_parse_repair': sum(row['status'] == 'valid' and bool(row.get('parse_repaired'))
                                               for row in checkpoints),
            'parse_repairs_flagged': sum(bool(row.get('parse_repaired')) for row in checkpoints),
            'fallback_reasons': dict(sorted(reasons.items())), 'technical_errors_in_saved_run': statuses['technical_error']}


def anonymized_examples(rows):
    relevant = [row for row in rows if row['relevant']]

    def delta(row):
        return row['scores']['MaxSim'] - row['scores']['MeanSim']

    choices = {
        'A_MaxSim_better': max(relevant, key=delta),
        'B_MeanSim_better': min(relevant, key=delta),
        'C_no_metric_change_vs_CBF': next(row for row in relevant if row['scores']['cbf'] == row['scores']['MaxSim']
                                          == row['scores']['MeanSim'] and row['relevant'].intersection(row['pool'])
                                          and row['rankings']['MaxSim'] == row['rankings']['MeanSim']),
        'D_all_relevant_outside_candidates': next(row for row in relevant
                                                  if row['relevant'].isdisjoint(row['pool'])),
    }
    require(delta(choices['A_MaxSim_better']) > 0 and delta(choices['B_MeanSim_better']) < 0,
            'No examples for both directions of paired difference')

    examples = {}
    methods = ('cbf', 'MaxSim', 'MeanSim')
    for label, row in choices.items():
        candidate_ranks = {isin: rank for rank, isin in enumerate(row['pool'], 1)}
        ranks = {method: {isin: rank for rank, isin in enumerate(row['rankings'][method], 1)} for method in methods}
        examples[label] = {
            'used_history_assets': row['history_count'],
            'ndcg_at_10': {method: row['scores'][method] for method in methods},
            'embedding_top10s_identical': row['rankings']['MaxSim'] == row['rankings']['MeanSim'],
            'maxsim_top10_identical_to_cbf': row['rankings']['MaxSim'] == row['rankings']['cbf'],
            'relevant_assets': [
                {'asset': f'T{number}', 'bought_before_cutoff': isin in row['prior_buys'],
                 'candidate_rank': candidate_ranks.get(isin),
                 'top10_ranks': {method: ranks[method].get(isin) for method in methods}}
                for number, isin in enumerate(sorted(row['relevant']), 1)
            ],
        }
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=run.OUTPUTS / 'full500')
    parser.add_argument('--data-dir', type=Path, default=run.ROOT / 'data')
    args = parser.parse_args()
    protocol, rows, checkpoints, timing, reconstruction_seconds = load_verified(args.run_dir, args.data_dir)
    summary = {
        'protocol': {key: protocol[key] for key in ('protocol_sha256', 'cohort_sha256', 'cutoff',
                     'end_exclusive', 'candidate_count', 'candidate_pools_sha256')},
        'verification': {'ordered_pools_matched_for_users': len(rows), 'prediction_and_grade_rows_checked':
                         len(rows) * len(METHODS), 'reconstructed_prompt_and_embedding_case_hashes_matched': True},
        'overall': group_metrics(rows),
        'by_used_history_assets': {label: group_metrics([row for row in rows if row['history_count'] in counts])
                                   for label, counts in GROUPS.items()},
        'coverage': candidate_coverage(rows), 'repeated_recommendations': repeats(rows),
        'llmrank': llm_summary(checkpoints),
        'timing_seconds': {
            'original_data_preparation': None, 'analysis_data_preparation_replay': reconstruction_seconds,
            'embedding_cache_fill_including_model_load': timing['cache_seconds'], 'embedding_cache': timing['cache'],
            'maxsim_and_meansim_combined_scoring': timing['scoring_end_to_end_seconds'],
            'llmrank_per_user_wall_sum': sum(row['wall_seconds'] for row in checkpoints),
            'llmrank_generation_sum': sum(row.get('seconds', 0) for row in checkpoints),
            'llmrank_model_load': None,
        },
    }
    examples = anonymized_examples(rows)
    results = Path(__file__).resolve().parent / 'results'
    results.mkdir(exist_ok=True)
    run.atomic_json(results / 'failure_analysis.json', summary)
    run.OUTPUTS.mkdir(exist_ok=True)
    run.atomic_json(run.OUTPUTS / 'failure_examples.json', {'protocol_sha256': protocol['protocol_sha256'],
                    'selection': 'Extreme paired differences, then first qualifying tie and coverage miss',
                    'examples': examples})


if __name__ == '__main__':
    main()
