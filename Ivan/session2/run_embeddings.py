"""Run a three-user smoke test or explicitly evaluate MaxSim/MeanSim for HW2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace

import embedding_reranker as reranker
import data_adapter

base = reranker.pilot.base


def select_smoke_users(manifest, history):
    return [next(u for u in manifest['users'] if len(history.get(u, {})) == count)
            for count in (1, 2, 5)]


def predict(bundle, cache, encoder, *, smoke, device='mps'):
    starter, assets, history, _truth, contexts, _baseline, events, _pop, manifest = bundle
    if _truth is not None:
        raise ValueError('Remove evaluation labels before prediction')
    users = select_smoke_users(manifest, history) if smoke else list(manifest['users'])
    cases = [reranker.prepare_case(contexts[u], assets, events.get(u, ())) for u in users]
    records, timing = reranker.rerank_cases(cases, cache, encoder, device=device)
    predictions = {}
    for uid, record in zip(users, records, strict=True):
        for ranking in record['rankings'].values():
            starter.validate_ranking(ranking, record['candidate_ids'])
            starter.validate_ranking(ranking, contexts[uid]['eligible_asset_ids'])
        predictions[uid] = record['rankings']
    return users, cases, predictions, records, timing


def execute(args, bundle, cache, encoder, *, device='mps'):
    """Keep real future labels outside the predictor and outside smoke evaluation."""
    smoke = args.command == 'smoke'
    manifest = bundle[-1]
    if smoke:
        # Drop labels even though the teacher loader always constructs them.
        bundle = (*bundle[:3], None, *bundle[4:])
    prediction_bundle = (*bundle[:3], None, *bundle[4:])
    users, cases, predictions, records, timing = predict(prediction_bundle, cache, encoder, smoke=smoke, device=device)
    if smoke:
        # Reopen the on-disk cache and prove it serves a second pass without inference.
        class NoInference:
            def encode(self, texts):
                raise RuntimeError('Unexpected model call on the cache-only pass')
        reopened = reranker.EmbeddingCache(cache.path.parent, cache.metadata)
        cached, cache_timing = reranker.rerank_cases(cases, reopened, NoInference(), device=device)
        if any(a['scores'] != b['scores'] or a['rankings'] != b['rankings']
               for a, b in zip(records, cached, strict=True)):
            raise RuntimeError('Cache reload changed scores or rankings')
        timing['cache_reload_check'] = dict(cache_timing, identical_scores_and_top10=True)
    else:
        # Uses starter.validate_ranking, starter.ndcg10 and starter.aggregate unchanged.
        data_adapter.save_results(args, bundle, predictions, [], 'embedding_rerankers',
                                  dict(encoder.metadata, methods=list(reranker.METHODS), timing=timing,
                                       history_limit=5, cache_metadata=cache.metadata))
    # Customer IDs are omitted from smoke artifacts. Full evaluation's standard
    # per-user IDs are written only by data_adapter to the ignored output directory.
    public_records = [dict(case_id=f'case_{i+1}', **record) for i, record in enumerate(records)]
    base.write_json(args.output_dir / 'rankings.json', public_records)
    base.write_json(args.output_dir / 'inputs.json', cases)
    base.write_json(args.output_dir / 'protocol.json', {
        'mode': args.command, 'cutoff': args.cutoff, 'horizon_days': args.horizon_days,
        'cohort_users': len(manifest['users']), 'scored_users': len(users),
        'cohort_sha256': base.digest(manifest['users']), 'candidate_count': 50,
        'history_limit': 5, 'history_counts': [len(c['history']) for c in cases],
        'history_order': 'latest Buy date descending, then ISIN ascending; day-level dates',
        'repeat_allowed': True, 'methods': list(reranker.METHODS),
        'same_candidates_and_metadata_for_both_methods': True,
        'future_labels_used_for_scoring': False, 'future_labels_used_for_evaluation': not smoke,
        'tie_break': reranker.pilot.SETTINGS['tie_break'],
        'cases_sha256': base.digest(cases), 'data_sha256': manifest['data_sha256'],
        'cache_metadata': cache.metadata, 'cache_file': str(cache.path),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__), Path(reranker.__file__), Path(reranker.pilot.__file__), Path(base.__file__))}})
    return {'mode': args.command, 'scored_users': len(users),
            'history_counts': [len(c['history']) for c in cases],
            'valid_top10_count': len(users) * len(reranker.METHODS),
            'future_evaluation_performed': not smoke, 'timing': timing, 'errors': []}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['smoke', 'evaluate'])
    parser.add_argument('--data-dir', type=Path, default=base.ROOT / 'data')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=reranker.DEFAULT_CACHE)
    args = parser.parse_args(argv)
    args.starter_dir = base.ROOT / 'shared/far-trans'
    args.cutoff, args.horizon_days, args.max_users, args.candidate_count = '2021-06-01', 30, 100, 50
    if args.output_dir is None:
        args.output_dir = base.DEFAULT_OUTPUT.parent / ('embedding_smoke' if args.command == 'smoke' else 'embedding_full')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Output directory must be empty; choose another --output-dir')
    reranker.require_mps()
    before = base.shared_hashes()
    started = time.perf_counter()
    bundle = data_adapter.load(args)
    data_load_seconds = time.perf_counter() - started
    if args.command == 'smoke':
        bundle = (*bundle[:3], None, *bundle[4:])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = reranker.EmbeddingCache(args.cache_dir)
    encoder = reranker.QwenEmbedder()
    summary = execute(args, bundle, cache, encoder)
    after = base.shared_hashes()
    if before != after:
        raise RuntimeError('Shared source hashes changed')
    summary.update(data_load_seconds=data_load_seconds, shared_unchanged=True,
                   shared_sha256=after, runtime=encoder.metadata,
                   environment={'python': sys.version, 'platform': platform.platform(),
                                'device': 'mps', 'dtype': 'float16', 'cpu_fallback': False,
                                'versions': cache.metadata['versions']})
    base.write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('shared_sha256', 'environment')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
