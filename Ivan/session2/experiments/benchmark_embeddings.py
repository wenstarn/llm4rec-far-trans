"""Bounded, past-only embedding pilot: three histories and 50 CBF candidates each."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

if __package__:
    from . import benchmark_mps as base
else:
    import benchmark_mps as base

OUTPUT = base.DEFAULT_OUTPUT.parent / 'embedding_pilot'
MODEL = 'Qwen/Qwen3-Embedding-0.6B'
FIELDS = {'assetName': 'Asset name', 'assetCategory': 'Category',
          'assetSubCategory': 'Subcategory', 'sector': 'Sector',
          'industry': 'Industry', 'marketID': 'Market'}
SETTINGS = {'device': 'mps', 'dtype': 'float16', 'attn_implementation': 'sdpa',
            'batch_size': 8, 'torch_threads': 4, 'seed': 42,
            'padding_side': 'left', 'pooling': 'last nonpadding token',
            'normalization_dtype': 'float32', 'similarity_dtype': 'float32',
            'embedding_dimensions': 1024, 'context_limit': 8192,
            'warmup_full_passes': 1, 'measured_full_passes': 3,
            'instruction': None, 'chat_template': False,
            'embedding_cache_key': 'exact feature text; shared across both methods',
            'mean_history': 'unweighted mean of unit vectors, then L2 normalization',
            'tie_break': 'scores rounded to 6 decimals; then original CBF order',
            'canonical_scores': 'first measured pass', 'use_cache': False}


def asset_text(asset):
    # Identical feature representation on both sides; identifiers and dates are not features.
    return '\n'.join(f'{label}: {asset[field]}' for field, label in FIELDS.items())


def prepare(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory')
    before = base.shared_hashes()
    sys.path.insert(0, str(base.ROOT / 'shared/far-trans-llm'))
    import data_adapter
    import ranker
    args = SimpleNamespace(starter_dir=base.ROOT / 'shared/far-trans', data_dir=base.ROOT / 'data',
                           cutoff='2021-06-01', horizon_days=30, max_users=100, candidate_count=50)
    starter, assets, history, unused_truth, contexts, baseline, events, pop, manifest = data_adapter.load(args)
    del unused_truth  # Teacher loader returns labels; never use them for selection or scoring.
    old_output = base.DEFAULT_OUTPUT.parent / 'ab_logits'
    old_cases = json.loads((old_output / 'cases.json').read_text())
    old_protocol = json.loads((old_output / 'protocol.json').read_text())
    assert manifest['data_sha256'] == old_protocol['data_sha256']
    assert base.digest(manifest['users']) == old_protocol['cohort_sha256']
    cases, unique = [], {}
    for count, old in zip((1, 2, 5), old_cases, strict=True):
        uid = next(u for u in manifest['users'] if len(history.get(u, {})) == count)
        recent = base.last_distinct_buys(events.get(uid, ()), args.cutoff, limit=5)
        bought = [dict(ranker.asset_record(isin, assets, args.cutoff), last_buy_date=day)
                  for isin, day in recent]
        candidates = [ranker.asset_record(row['id'], assets, args.cutoff)
                      for row in contexts[uid]['candidate_assets'][:50]]
        assert len(bought) == count and len(candidates) == 50
        assert len({a['ISIN'] for a in candidates}) == 50
        assert set(a['ISIN'] for a in candidates) <= set(contexts[uid]['eligible_asset_ids'])
        assert all(a['timestamp'][:10] < args.cutoff for a in bought + candidates)
        assert all(a['last_buy_date'] < args.cutoff for a in bought)
        assert bought == old['history'] and candidates[:10] == old['candidates']
        for asset in bought + candidates:
            record = {'ISIN': asset['ISIN'], 'text': asset_text(asset)}
            assert asset['ISIN'] not in unique or unique[asset['ISIN']] == record
            unique[asset['ISIN']] = record
        cases.append({'case_id': f'history_{count}', 'history_count': count,
                      'history': bought, 'candidates': candidates})
    records = list(unique.values())  # One encoding per distinct asset, reused by both methods.
    assert before == base.shared_hashes()
    output.mkdir(parents=True)
    protocol = {'model': MODEL, 'settings': SETTINGS, 'feature_fields': list(FIELDS),
                'cutoff': args.cutoff, 'horizon_days': 30, 'cohort_users': 100,
                'cohort_sha256': base.digest(manifest['users']),
                'selection': 'same deterministic users as ab_logits; exactly 1, 2, 5 distinct past assets',
                'candidate_policy': 'first 50 teacher CBF candidates, repeat purchases allowed',
                'future_labels_used': False, 'ndcg_computed': False,
                'same_histories_and_first_10_candidates_as_ab_pilot': True,
                'metadata_and_purchase_dates_strictly_before_cutoff': True,
                'candidate_memberships': 150, 'pair_count': 400,
                'unique_assets': len(records), 'unique_texts': len({r['text'] for r in records}),
                'data_sha256': manifest['data_sha256'], 'cases_sha256': base.digest(cases),
                'assets_sha256': base.digest(records), 'shared_sha256': before,
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__), Path(base.__file__))}}
    base.write_json(output / 'cases.json', cases)
    base.write_json(output / 'assets.json', records)
    base.write_json(output / 'protocol.json', protocol)
    print(json.dumps({k: protocol[k] for k in ('unique_assets', 'unique_texts', 'pair_count',
                     'same_histories_and_first_10_candidates_as_ab_pilot')}), flush=True)


def download(output):
    from huggingface_hub import HfApi, snapshot_download
    path = output / 'model.json'
    info = json.loads(path.read_text()) if path.exists() else {
        'model': MODEL, 'revision': HfApi().model_info(MODEL, token=False).sha}
    base.write_json(path, info)
    print(f"Downloading {MODEL}@{info['revision']}", flush=True)
    info['snapshot'] = snapshot_download(MODEL, revision=info['revision'], token=False,
                                        allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja', 'README.md'])
    base.write_json(path, info)
    print(f"Ready: {MODEL}@{info['revision']}", flush=True)


def last_token_pool(hidden, mask):
    import torch
    # Works with either padding side, with no padded token selected.
    assert bool(mask.any(dim=1).all())
    indices = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = indices.masked_fill(mask == 0, -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def similarity_scores(history, candidates):
    import torch.nn.functional as F
    # Both inputs already contain L2-normalized vectors.
    pairs = candidates @ history.T
    centroid = history.mean(dim=0, keepdim=True)
    assert centroid.norm().item() > 1e-12
    mean_scores = (candidates @ F.normalize(centroid, dim=1).T)[:, 0]
    return pairs, pairs.max(dim=1).values, mean_scores


def describe(values):
    import numpy as np
    a = np.asarray(values, dtype=float).reshape(-1)
    return {'n': int(a.size), 'min': float(a.min()), 'median': float(np.median(a)),
            'p10': float(np.quantile(a, .1)), 'p90': float(np.quantile(a, .9)),
            'max': float(a.max()), 'mean': float(a.mean()), 'std': float(a.std()),
            'span': float(a.max() - a.min()), 'unique_6dp': len(set(a.round(6))),
            'unique_3dp': len(set(a.round(3)))}


def ranking(scores, case):
    order = sorted(range(len(scores)), key=lambda i: (-round(scores[i], 6), i))
    known = {a['ISIN'] for a in case['history']}
    top10 = [{'candidate': f'C{i+1:02}', 'ISIN': case['candidates'][i]['ISIN'],
              'score': scores[i], 'already_in_history': case['candidates'][i]['ISIN'] in known}
             for i in order[:10]]
    assert len(top10) == len({r['ISIN'] for r in top10}) == 10
    return {'scores_cbf_order': scores, 'order': order, 'top10': top10,
            'distribution': describe(scores),
            'top10_history_memberships': sum(row['already_in_history'] for row in top10),
            'score_gap_rank10_minus_rank11': scores[order[9]] - scores[order[10]]}


def encode(torch, model, tokenizer, texts):
    import torch.nn.functional as F
    vectors, batches = [], []
    start = time.perf_counter()
    unique_texts = list(dict.fromkeys(texts))
    text_index = {text: i for i, text in enumerate(unique_texts)}
    for offset in range(0, len(unique_texts), SETTINGS['batch_size']):
        inputs = tokenizer(unique_texts[offset:offset + SETTINGS['batch_size']], padding=True,
                           truncation=False, return_tensors='pt')
        assert inputs['input_ids'].shape[1] <= SETTINGS['context_limit']
        assert bool(inputs['attention_mask'][:, -1].all())
        positions = inputs['attention_mask'].long().cumsum(-1) - 1
        inputs['position_ids'] = positions.masked_fill(inputs['attention_mask'] == 0, 0)
        inputs = inputs.to('mps')
        torch.mps.synchronize()
        compute_start = time.perf_counter()
        with torch.inference_mode():
            hidden = model(**inputs, use_cache=False, return_dict=True).last_hidden_state
            pooled = last_token_pool(hidden, inputs['attention_mask']).float()
            embeddings = F.normalize(pooled, p=2, dim=1)
        torch.mps.synchronize()
        compute_seconds = time.perf_counter() - compute_start
        assert embeddings.device.type == 'mps' and torch.isfinite(embeddings).all().item()
        assert torch.allclose(embeddings.norm(dim=1), torch.ones(len(embeddings), device='mps'), atol=1e-6)
        vectors.append(embeddings)
        batches.append({'assets': len(embeddings), 'padded_tokens': inputs['input_ids'].shape[1],
                        'compute_seconds': compute_seconds})
        del hidden, pooled, inputs
    vectors = torch.cat(vectors)[[text_index[text] for text in texts]]
    torch.mps.synchronize()
    return vectors, {'unique_texts_encoded': len(unique_texts),
                     'compute_seconds': sum(b['compute_seconds'] for b in batches),
                     'end_to_end_seconds': time.perf_counter() - start, 'batches': batches}


def score_cases(torch, vectors, assets, cases):
    index = {a['ISIN']: i for i, a in enumerate(assets)}
    results, compute_seconds = {}, 0.
    for case in cases:
        history = vectors[[index[a['ISIN']] for a in case['history']]]
        candidates = vectors[[index[a['ISIN']] for a in case['candidates']]]
        torch.mps.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            pairs, max_scores, mean_scores = similarity_scores(history, candidates)
        torch.mps.synchronize()
        compute_seconds += time.perf_counter() - start
        assert torch.isfinite(pairs).all().item() and pairs.abs().max().item() <= 1 + 1e-6
        assert torch.isfinite(mean_scores).all().item() and mean_scores.abs().max().item() <= 1 + 1e-6
        matrix, max_values, mean_values = pairs.cpu().tolist(), max_scores.cpu().tolist(), mean_scores.cpu().tolist()
        same, different = [], []
        for ci, candidate in enumerate(case['candidates']):
            for hi, asset in enumerate(case['history']):
                (same if candidate['ISIN'] == asset['ISIN'] else different).append(matrix[ci][hi])
        assert all(abs(v - 1) < 1e-6 for v in same)
        if case['history_count'] == 1:
            assert max(abs(a - b) for a, b in zip(max_values, mean_values)) < 1e-6
        max_rank, mean_rank = ranking(max_values, case), ranking(mean_values, case)
        results[case['case_id']] = {'history_count': case['history_count'],
            'pairwise_cosines': matrix, 'pair_distribution': describe(matrix),
            'nonidentical_pair_distribution': describe(different),
            'identical_isin_pairs': len(same), 'identical_isin_cosines': same,
            'max': max_rank, 'mean': mean_rank,
            'top10_overlap': len(set(max_rank['order'][:10]) & set(mean_rank['order'][:10])),
            'same_top1': max_rank['order'][0] == mean_rank['order'][0],
            'valid_top10': True}
    return results, compute_seconds


def run(output):
    import torch
    import psutil
    from transformers import AutoModel, AutoTokenizer
    protocol = json.loads((output / 'protocol.json').read_text())
    cases = json.loads((output / 'cases.json').read_text())
    assets = json.loads((output / 'assets.json').read_text())
    info = json.loads((output / 'model.json').read_text())
    assert protocol['settings'] == SETTINGS and info['model'] == MODEL
    assert base.digest(cases) == protocol['cases_sha256'] and base.digest(assets) == protocol['assets_sha256']
    assert base.shared_hashes() == protocol['shared_sha256']
    for name, expected in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; CPU fallback is forbidden')
    if (output / 'runs.jsonl').exists():
        raise FileExistsError('Measured log already exists')
    torch.set_num_threads(SETTINGS['torch_threads'])
    torch.manual_seed(SETTINGS['seed'])
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'cpu': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                   'ram_bytes': psutil.virtual_memory().total, 'mps_available': True, 'cpu_fallback': False,
                   'versions': {p: importlib.metadata.version(p) for p in
                                ('torch', 'transformers', 'tokenizers', 'huggingface-hub', 'numpy', 'psutil')}}
    base.write_json(output / 'environment.json', environment)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(info['snapshot'], local_files_only=True,
                                               trust_remote_code=False, padding_side='left')
    model = AutoModel.from_pretrained(info['snapshot'], local_files_only=True, trust_remote_code=False,
                                     dtype=torch.float16, attn_implementation='sdpa').to('mps').eval()
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - started
    assert all(p.device.type == 'mps' and p.dtype == torch.float16 for p in model.parameters())
    assert model.config.hidden_size == SETTINGS['embedding_dimensions']
    texts = [a['text'] for a in assets]
    encoded = tokenizer(texts, padding=False, truncation=False)
    lengths = list(map(len, encoded['input_ids']))
    assert max(lengths) <= SETTINGS['context_limit']
    base.write_json(output / 'tokenization.json', {'lengths': lengths, 'input_ids_sha256': base.digest(encoded['input_ids']),
                                                 'truncation': False, 'add_special_tokens': True})
    print(f'Loaded {MODEL} on MPS FP16 in {load_seconds:.3f}s; {len(assets)} unique assets', flush=True)
    records, canonical, canonical_vectors = [], None, None
    with (output / 'runs.jsonl').open('x') as log:
        for repetition in range(SETTINGS['measured_full_passes'] + 1):
            with base.MemorySampler(torch) as memory:
                start = time.perf_counter()
                vectors, encoding = encode(torch, model, tokenizer, texts)
                results, scoring_seconds = score_cases(torch, vectors, assets, cases)
                end_to_end = time.perf_counter() - start
            record = {'phase': 'warmup' if repetition == 0 else 'measured', 'repetition': repetition,
                      'encoding': encoding, 'similarity_compute_seconds': scoring_seconds,
                      'end_to_end_seconds': end_to_end, 'sampled_peak_memory': memory.peak}
            if repetition == 0:
                base.write_json(output / 'warmup.json', record)
            else:
                if canonical is None:
                    canonical, canonical_vectors = results, vectors.cpu()
                    torch.save({'asset_ids': [a['ISIN'] for a in assets], 'embeddings': canonical_vectors}, output / 'embeddings.pt')
                    base.write_json(output / 'rankings.json', canonical)
                record['max_embedding_delta_from_first_pass'] = (vectors.cpu() - canonical_vectors).abs().max().item()
                record['top10_order_matches_first_pass'] = {
                    key: {method: results[key][method]['order'][:10] == canonical[key][method]['order'][:10]
                          for method in ('max', 'mean')} for key in results}
                records.append(record)
                log.write(json.dumps(record) + '\n')
                log.flush()
            print(json.dumps({k: record[k] for k in ('phase', 'repetition', 'end_to_end_seconds', 'similarity_compute_seconds')}), flush=True)
            del vectors
    assert base.shared_hashes() == protocol['shared_sha256']
    summary = {'model': MODEL, 'revision': info['revision'], 'settings': SETTINGS,
               'unique_assets': len(assets), 'pair_count': 400, 'load_seconds': load_seconds,
               'median_encoding_compute_seconds': statistics.median(r['encoding']['compute_seconds'] for r in records),
               'median_encoding_end_to_end_seconds': statistics.median(r['encoding']['end_to_end_seconds'] for r in records),
               'median_similarity_compute_seconds': statistics.median(r['similarity_compute_seconds'] for r in records),
               'median_end_to_end_seconds': statistics.median(r['end_to_end_seconds'] for r in records),
               'sampled_peak_memory_gib': {k: max(r['sampled_peak_memory'][k] for r in records) / 2**30
                                           for k in records[0]['sampled_peak_memory']},
               'max_repeat_embedding_delta': max(r['max_embedding_delta_from_first_pass'] for r in records),
               'all_repeat_top10_orders_identical': all(all(all(v.values()) for v in r['top10_order_matches_first_pass'].values()) for r in records),
               'shared_unchanged': True, 'errors': [],
               'cases': {key: {k: v for k, v in result.items() if k != 'pairwise_cosines'}
                         for key, result in canonical.items()}}
    base.write_json(output / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('cases', 'settings')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'download', 'run'])
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    {'prepare': prepare, 'download': download, 'run': run}[args.command](args.output_dir)
