"""Fixed-pair label-swap control for the existing Qwen3 A/B scorer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import time

if __package__:
    from . import benchmark_ab_logits as scorer
else:
    import benchmark_ab_logits as scorer

OUTPUT = scorer.OUTPUT.parent / 'ab_reliability'
ORIGINAL_SUFFIX = 'A = related. B = unrelated. Answer only A or B.'
SWAPPED_SUFFIX = 'A = unrelated. B = related. Answer only A or B.'
CONFIGS = [('original', 1), ('swapped', 1), ('original', 8), ('swapped', 8)]
REPETITIONS = 3


def swapped_prompt(prompt):
    if not prompt.endswith(ORIGINAL_SUFFIX):
        raise ValueError('Original prompt does not end with the expected label definition')
    return prompt[:-len(ORIGINAL_SUFFIX)] + SWAPPED_SUFFIX


def semantic_values(values, variant):
    if variant not in ('original', 'swapped'):
        raise ValueError('Unknown prompt variant')
    # Preserve raw logits; column 2 always means semantic P(related) for aggregation.
    return [list(v) if variant == 'original' else [v[0], v[1], v[3], v[2], v[4]] for v in values]


def semantic_margin(value, variant):
    delta = value[0] - value[1]
    return delta if variant == 'original' else -delta


def pearson(x, y):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError('Correlation requires paired observations')
    mx, my = statistics.mean(x), statistics.mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx == 0 or vy == 0:
        return None
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / math.sqrt(vx*vy)


def midranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for j in range(start, end):
            ranks[order[j]] = (start + end - 1) / 2
        start = end
    return ranks


def score_agreement(first, second):
    deltas = [abs(a-b) for a, b in zip(first, second)]
    return {'n': len(first), 'mean_absolute_delta': statistics.mean(deltas),
            'max_absolute_delta': max(deltas),
            'pearson': pearson(first, second), 'spearman': pearson(midranks(first), midranks(second)),
            'semantic_decision_agreement': sum((a >= .5) == (b >= .5) for a, b in zip(first, second))}


def order_agreement(first, second):
    if len(first) != len(set(first)) or set(first) != set(second) or len(first) != len(second):
        raise ValueError('Require two permutations of the same candidates')
    rank_b = {v: i for i, v in enumerate(second)}
    inversions = sum(rank_b[first[i]] > rank_b[first[j]]
                     for i in range(len(first)) for j in range(i+1, len(first)))
    total = len(first)*(len(first)-1)//2
    return {'same_order': first == second, 'same_top1': first[0] == second[0],
            'same_positions': sum(a == b for a, b in zip(first, second)),
            'top3_overlap': len(set(first[:3]) & set(second[:3])),
            'top5_overlap': len(set(first[:5]) & set(second[:5])),
            'inverted_pairs': inversions, 'kendall_order_tau': 1 - 2 * inversions / total}


def prepare(output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory')
    reference = scorer.OUTPUT
    original_protocol = json.loads((reference / 'protocol.json').read_text())
    cases = json.loads((reference / 'cases.json').read_text())
    pairs = json.loads((reference / 'pairs.json').read_text())
    assert len(pairs) == 80 and scorer.base.digest(pairs) == original_protocol['pairs_sha256']
    assert scorer.base.digest(cases) == original_protocol['cases_sha256']
    for case in cases:
        assert all(a['timestamp'][:10] < original_protocol['cutoff'] for a in case['history'] + case['candidates'])
        assert all(a['last_buy_date'] < original_protocol['cutoff'] for a in case['history'])
    by_case = {c['case_id']: c for c in cases}
    swapped = []
    for pair in pairs:
        case = by_case[pair['case_id']]
        assert pair['prompt'] == scorer.build_prompt(case['history'][pair['history_index']], case['candidates'][pair['candidate_index']])
        prompt = swapped_prompt(pair['prompt'])
        swapped.append(dict(pair, prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
    variants = {'original': pairs, 'swapped': swapped}
    output.mkdir(parents=True, exist_ok=True)
    sources = (Path(__file__), Path(scorer.__file__), Path(scorer.base.__file__))
    protocol = {'model': scorer.MODEL, 'reference_directory': str(reference),
                'cases_sha256': original_protocol['cases_sha256'],
                'original_pairs_sha256': original_protocol['pairs_sha256'],
                'variants_sha256': scorer.base.digest(variants),
                'cutoff': original_protocol['cutoff'], 'future_data_read': False,
                'unique_pairs': 80, 'repetitions': REPETITIONS,
                'configs': CONFIGS, 'scorer_settings': scorer.SETTINGS,
                'semantic_label': {'original': 'A', 'swapped': 'B'},
                'single_change': {'from': ORIGINAL_SUFFIX, 'to': SWAPPED_SUFFIX},
                'analysis': 'probability and semantic logit margin; max across history; exact ties retain CBF order',
                'shared_sha256': scorer.base.shared_hashes(),
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    scorer.base.write_json(output / 'cases.json', cases)
    scorer.base.write_json(output / 'variants.json', variants)
    scorer.base.write_json(output / 'protocol.json', protocol)
    scorer.base.write_json(output / 'model.json', json.loads((reference / 'model.json').read_text()))
    print('Prepared original and label-swapped prompts for the exact same 80 historical pairs.', flush=True)


def summarize(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    variants = json.loads((output / 'variants.json').read_text())
    cases = json.loads((output / 'cases.json').read_text())
    runs = [json.loads(line) for line in (output / 'runs.jsonl').read_text().splitlines()]
    assert len(runs) == 12 and scorer.base.digest(variants) == protocol['variants_sha256']
    pairs = variants['original']
    configs, scores, ranks, repeat_deltas, canonical_runs = {}, {}, {}, {}, {}
    canonical_rows = {}
    for variant, bs in CONFIGS:
        key = f'{variant}_batch_{bs}'
        selected = [r for r in runs if r['variant'] == variant and r['batch_size'] == bs]
        assert sorted(r['repetition'] for r in selected) == [1, 2, 3]
        canonical = next(r for r in selected if r['repetition'] == 1)
        raw = canonical['values']
        assert len(raw) == 80
        mapped = semantic_values(raw, variant)
        probabilities = [v[2] for v in mapped]
        margins = [semantic_margin(v, variant) for v in raw]
        raw_margins = [v[0]-v[1] for v in raw]
        identical = [p for pair, p in zip(pairs, probabilities) if pair['identical_isin']]
        configs[key] = {
            'semantic_related_probability': scorer.distribution(probabilities),
            'raw_logit_A_minus_B': scorer.distribution(raw_margins),
            'semantic_logit_related_minus_unrelated': scorer.distribution(margins),
            'A_selected': sum(v > 0 for v in raw_margins), 'B_selected': sum(v < 0 for v in raw_margins),
            'raw_logit_ties': sum(v == 0 for v in raw_margins),
            'related_selected': sum(v > 0 for v in margins),
            'A_plus_B_vocab_mass': scorer.distribution([v[4] for v in raw]),
            'identical_isin': {'pairs': len(identical), 'related_selected': sum(p > .5 for p in identical),
                               'mean_probability_related': statistics.mean(identical)},
            'median_compute_seconds_80_pairs': statistics.median(r['compute_seconds'] for r in selected),
            'median_end_to_end_seconds_80_pairs': statistics.median(r['end_to_end_seconds'] for r in selected),
        }
        scores[key] = probabilities
        ranks[key] = {c['case_id']: scorer.aggregate_max(pairs, mapped, c['case_id']) for c in cases}
        repeat_deltas[key] = max(abs(a[2]-b[2]) for r in selected[1:] for a,b in zip(mapped, semantic_values(r['values'], variant)))
        canonical_runs[key] = raw
        canonical_rows[key] = [
            {'pair_index': p['pair_index'], 'case_id': p['case_id'],
             'candidate_index': p['candidate_index'], 'history_index': p['history_index'],
             'logit_A': v[0], 'logit_B': v[1], 'raw_logit_A_minus_B': v[0]-v[1],
             'probability_related': m[2], 'semantic_logit_related_minus_unrelated': semantic_margin(v, variant),
             'unconstrained_A_plus_B_mass': v[4]}
            for p,v,m in zip(pairs, raw, mapped)]
    label_agreement, batch_agreement, label_ranks, batch_ranks = {}, {}, {}, {}
    for bs in (1, 8):
        a, b = f'original_batch_{bs}', f'swapped_batch_{bs}'
        agreement = score_agreement(scores[a], scores[b])
        agreement['semantic_margin_pearson'] = pearson(
            [semantic_margin(v, 'original') for v in canonical_runs[a]],
            [semantic_margin(v, 'swapped') for v in canonical_runs[b]])
        agreement['A_selected_in_both'] = sum(x[0] > x[1] and y[0] > y[1]
                                             for x,y in zip(canonical_runs[a],canonical_runs[b]))
        label_agreement[str(bs)] = agreement
        label_ranks[str(bs)] = {c['case_id']: order_agreement(ranks[a][c['case_id']]['ranked_candidate_indices'],
                                                           ranks[b][c['case_id']]['ranked_candidate_indices']) for c in cases}
    for variant in ('original', 'swapped'):
        a, b = f'{variant}_batch_1', f'{variant}_batch_8'
        batch_agreement[variant] = score_agreement(scores[a], scores[b])
        batch_ranks[variant] = {c['case_id']: order_agreement(ranks[a][c['case_id']]['ranked_candidate_indices'],
                                                           ranks[b][c['case_id']]['ranked_candidate_indices']) for c in cases}
    previous = [json.loads(line) for line in (scorer.OUTPUT / 'runs.jsonl').read_text().splitlines()]
    previous = next(r for r in previous if r['batch_size'] == 1 and r['repetition'] == 1)
    summary = {
        'model': scorer.MODEL, 'revision': json.loads((output / 'model.json').read_text())['revision'],
        'unique_pairs': 80, 'measured_passes': 12, 'configs': configs,
        'label_swap_score_agreement': label_agreement, 'batch_score_agreement': batch_agreement,
        'label_swap_rank_agreement': label_ranks, 'batch_rank_agreement': batch_ranks,
        'same_config_repetition_max_abs_probability_delta': repeat_deltas,
        'original_probability_max_abs_delta_vs_previous_pilot': max(abs(a[2]-b[2]) for a,b in zip(canonical_runs['original_batch_1'],previous['values'])),
        'shared_unchanged': scorer.base.shared_hashes() == protocol['shared_sha256'],
        'limits': 'Fixed 80-pair diagnostic only. A is listed first in both prompts, so letter preference and first-option preference are confounded. Top-10 membership is fixed; metrics compare order. No future labels or nDCG, no prompt tuning, no calibrated relevance ground truth.',
    }
    assert summary['shared_unchanged']
    scorer.base.write_json(output / 'canonical_scores.json', canonical_rows)
    scorer.base.write_json(output / 'rankings.json', ranks)
    scorer.base.write_json(output / 'summary.json', summary)
    lines = ['Qwen3-0.6B: fixed-pair A/B label-swap reliability control',
             '80 identical pairs; original and swapped label meanings; MPS FP16; batches 1 and 8; three measured passes each.']
    for key, c in configs.items():
        p, d = c['semantic_related_probability'], c['raw_logit_A_minus_B']
        lines.append(f"{key}: P(related) mean={p['mean']:.6f}, median={p['median']:.6f}, range={p['min']:.6f}..{p['max']:.6f}; mean raw A-B={d['mean']:.6f}; A selected={c['A_selected']}/80; related selected={c['related_selected']}/80; median compute={c['median_compute_seconds_80_pairs']:.3f}s")
    for bs, agreement in label_agreement.items():
        lines.append(f"Label swap, batch {bs}: semantic decisions agree {agreement['semantic_decision_agreement']}/80; A selected in both {agreement['A_selected_in_both']}/80; mean abs P(related) difference={agreement['mean_absolute_delta']:.6f}; probability Pearson={agreement['pearson']}; Spearman={agreement['spearman']}; semantic-margin Pearson={agreement['semantic_margin_pearson']}")
    for case in cases:
        key = case['case_id']
        lines.append(f'{key}:')
        for variant, bs in CONFIGS:
            name = f'{variant}_batch_{bs}'
            lines.append(f"  {name}: " + ' > '.join(ranks[name][key]['ranked_candidate_aliases']))
        lines.append(f"  label-swap rank comparison batch 1: {label_ranks['1'][key]}")
        for variant in ('original', 'swapped'):
            lines.append(f"  {variant} batch-1 vs batch-8 order: {batch_ranks[variant][key]}")
    lines += [f'Batch score agreement: {batch_agreement}', f'Same-config repeat deltas: {repeat_deltas}',
              'C01..C10 are original CBF candidate positions. No customer IDs are published.', summary['limits']]
    (output / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def run(output):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    protocol = json.loads((output / 'protocol.json').read_text())
    variants = json.loads((output / 'variants.json').read_text())
    cases = json.loads((output / 'cases.json').read_text())
    assert scorer.base.digest(variants) == protocol['variants_sha256']
    assert scorer.base.digest(cases) == protocol['cases_sha256']
    assert scorer.base.shared_hashes() == protocol['shared_sha256']
    for p in (Path(__file__), Path(scorer.__file__), Path(scorer.base.__file__)):
        assert hashlib.sha256(p.read_bytes()).hexdigest() == protocol['source_sha256'][p.name]
    if (output / 'runs.jsonl').exists():
        raise FileExistsError('Measured run log exists; use a fresh output directory')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; CPU fallback is forbidden')
    torch.set_num_threads(scorer.SETTINGS['torch_threads'])
    torch.manual_seed(42)
    info = json.loads((output / 'model.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(info['snapshot'], local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    chats, audits = {}, {}
    for variant, pairs in variants.items():
        chats[variant], audits[variant] = scorer.audit_tokenization(tokenizer, pairs)
    assert audits['original']['label_token_ids'] == audits['swapped']['label_token_ids']
    scorer.base.write_json(output / 'tokenization.json', audits)
    model = AutoModelForCausalLM.from_pretrained(info['snapshot'], local_files_only=True,
                                                trust_remote_code=False, dtype=torch.float16,
                                                attn_implementation='sdpa').to('mps').eval()
    assert all(p.device.type == 'mps' for p in model.parameters())
    assert all(p.dtype == torch.float16 for p in model.parameters() if p.is_floating_point())
    scorer.base.write_json(output / 'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'mps_available': True, 'cpu_fallback': False, 'settings': scorer.SETTINGS,
        'versions': {p: importlib.metadata.version(p) for p in ('torch','transformers','tokenizers','psutil')},
    })
    print('MPS FP16 ready. Both label definitions verified on all 80 unchanged pairs.', flush=True)
    warmups = []
    for variant, bs in CONFIGS:
        indices = sorted(range(80), key=lambda i: -audits[variant]['input_token_lengths'][i])[:bs]
        _, timing = scorer.score_batch(torch, model, tokenizer, [chats[variant][i] for i in indices], audits[variant]['label_token_ids'])
        warmups.append(dict(variant=variant, batch_size=bs, **timing))
    scorer.base.write_json(output / 'warmup.json', warmups)
    with (output / 'runs.jsonl').open('x') as log:
        for repetition in range(1, REPETITIONS+1):
            for variant, bs in (CONFIGS if repetition % 2 else list(reversed(CONFIGS))):
                values, batches = [], []
                started = time.perf_counter()
                for case in cases:
                    ids = [i for i,p in enumerate(variants[variant]) if p['case_id'] == case['case_id']]
                    for start in range(0,len(ids),bs):
                        batch_ids = ids[start:start+bs]
                        rows, timing = scorer.score_batch(torch, model, tokenizer,
                                                         [chats[variant][i] for i in batch_ids],
                                                         audits[variant]['label_token_ids'])
                        values.extend(rows)
                        batches.append(dict(case_id=case['case_id'], **timing))
                assert len(values) == 80
                record = {'variant': variant, 'batch_size': bs, 'repetition': repetition,
                          'values': values, 'compute_seconds': sum(b['compute_seconds'] for b in batches),
                          'end_to_end_seconds': time.perf_counter()-started, 'batches': batches}
                log.write(json.dumps(record)+'\n')
                log.flush()
                print(json.dumps({k:record[k] for k in ('variant','batch_size','repetition','compute_seconds')}), flush=True)
    summarize(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare','run','summarize'))
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output_dir)
    elif args.stage == 'run':
        run(args.output_dir)
    else:
        summarize(args.output_dir)
