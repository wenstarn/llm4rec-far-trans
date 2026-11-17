"""Verify and summarize the saved pilot without loading a model or exposing histories."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

if __package__:
    from . import benchmark_mps as benchmark
else:
    import benchmark_mps as benchmark


def diagnose(record):
    raw = record['raw_response'].strip()
    fence = re.fullmatch(r'```(?:json)?\s*([\s\S]*?)\s*```', raw, flags=re.IGNORECASE)
    result = {'case_id': record['case_id'], 'markdown_fence': bool(fence),
              'strict_contract_valid': record['schema_valid'],
              'valid_after_fence_removal': False, 'expected_shape': [10, record['history_count']]}
    try:
        json.loads(raw)
        result['raw_json_syntax_valid'] = True
    except ValueError:
        result['raw_json_syntax_valid'] = False
    candidate = fence.group(1) if fence else raw
    try:
        obj = json.loads(candidate)
        matrix = obj.get('scores') if isinstance(obj, dict) else None
        if isinstance(matrix, list):
            result['observed_rows'] = len(matrix)
            result['observed_row_lengths'] = [len(r) if isinstance(r, list) else None for r in matrix]
        matrix = benchmark.validate_response(candidate, record['history_count'])
        result['valid_after_fence_removal'] = True
        result['score_histogram'] = dict(sorted(Counter(s for row in matrix for s in row).items()))
        result['constant_matrix'] = len(result['score_histogram']) == 1
    except (ValueError, TypeError) as exc:
        result['diagnostic_error'] = str(exc)
    return result


def main(output):
    protocol = json.loads((output / 'protocol.json').read_text())
    cases = json.loads((output / 'cases.json').read_text())
    assert benchmark.digest(cases) == protocol['cases_sha256']
    assert benchmark.shared_hashes() == protocol['shared_sha256']
    for case in cases:
        for asset in case['payload']['history'] + case['payload']['candidates']:
            assert asset['timestamp'][:10] < protocol['cutoff']
        assert all(a['last_buy_date'] < protocol['cutoff'] for a in case['payload']['history'])
    summaries, rows, diagnostics = {}, {}, {}
    for tag in ('0.6B', '1.7B'):
        summary = json.loads((output / f'summary_{tag}.json').read_text())
        records = [json.loads(line) for line in (output / f'requests_{tag}.jsonl').read_text().splitlines()]
        assert len(records) == 6 and records[0]['phase'] == 'warmup'
        measured = records[1:]
        assert all(r['phase'] == 'measured' for r in measured)
        assert [r['case_id'] for r in measured] == protocol['measured_case_order']
        assert [r['prompt_sha256'] for r in measured] == [c['prompt_sha256'] for c in cases]
        assert summary['settings'] == protocol['settings'] == benchmark.SETTINGS
        assert summary['benchmark_sha256'] == hashlib.sha256(Path(benchmark.__file__).read_bytes()).hexdigest()
        assert abs(summary['mean_end_to_end_seconds'] - statistics.mean(r['end_to_end_seconds'] for r in measured)) < 1e-10
        assert summary['schema_valid_calls'] == sum(r['schema_valid'] for r in measured)
        summaries[tag], rows[tag] = summary, measured
        diagnostics[tag] = [diagnose(r) for r in measured]
    assert [r['input_ids_sha256'] for r in rows['0.6B']] == [r['input_ids_sha256'] for r in rows['1.7B']]
    configs = [json.loads((output / f'generation_config_{t}.json').read_text()) for t in ('0.6B', '1.7B')]
    assert configs[0] == configs[1]
    paired_valid = [i for i in range(5) if all(diagnostics[t][i]['valid_after_fence_removal'] for t in diagnostics)]
    comparison = {'checks': 'identical prompts, input token IDs and generation configs; 1 warmup + 5 measured calls each; past-only payloads; unchanged teacher code',
                  'models': summaries, 'response_diagnostics': diagnostics,
                  'same_shape_diagnostic_cases': [cases[i]['case_id'] for i in paired_valid],
                  'same_shape_mean_generation_seconds': {t: statistics.mean(rows[t][i]['generation_seconds'] for i in paired_valid) for t in rows},
                  'preliminary_attempt': 'One 0.6B warmup and one interrupted request with model-default sampling were excluded and preserved in discarded_sampling_attempt/.',
                  'semantic_quality': 'No pair-score ground truth or recommendation experiment. Shape/range validity is not semantic accuracy.'}
    benchmark.write_json(output / 'comparison.json', comparison)
    lines = ['Qwen3 / MPS / FAR-Trans pilot', '', comparison['checks'], '',
             'Model | Mean request s | Median s | Output tok/s incl prefill | MPS driver observed peak GiB | Process lifetime RSS peak GiB | Strict valid | Valid after fence removal | Projected 500 min']
    for tag, s in summaries.items():
        repaired = sum(d['valid_after_fence_removal'] for d in diagnostics[tag])
        lines.append(f"{tag} | {s['mean_end_to_end_seconds']:.3f} | {s['median_end_to_end_seconds']:.3f} | {s['weighted_output_tokens_per_second_including_prefill']:.2f} | {s['measured_sampled_peak_memory']['mps_driver_bytes']/2**30:.3f} | {s['process_lifetime_max_rss_bytes']/2**30:.3f} | {s['schema_valid_calls']}/5 | {repaired}/5 | {s['projected_500_requests_seconds']/60:.2f}")
        lines.append(f"Revision: {s['model']}@{s['revision']}; load {s['load_seconds']:.3f}s; warmup {s['warmup_seconds']:.3f}s")
    lines.append('')
    for tag, items in diagnostics.items():
        lines.append(f"{tag}: Markdown-fenced {sum(d['markdown_fence'] for d in items)}/5; raw JSON syntax valid {sum(d['raw_json_syntax_valid'] for d in items)}/5; output tokens {[r['output_tokens'] for r in rows[tag]]}")
        for d in items:
            if not d['valid_after_fence_removal']:
                lines.append(f"  {d['case_id']}: expected {d['expected_shape']}, observed {d.get('observed_rows')} rows with lengths {d.get('observed_row_lengths')}; {d.get('diagnostic_error')}")
            elif d.get('constant_matrix'):
                lines.append(f"  {d['case_id']}: all scores equal {next(iter(d['score_histogram']))}")
    lines += ['', 'Fence removal is a diagnostic, not a retry or change to the strict benchmark. Constant matrices do not distinguish candidates.',
              'Malformed responses can have very different output lengths: similar mean request latency is not equal decoding speed.',
              'Memory sampled every 20ms; RSS and MPS overlap and must not be added. Process lifetime RSS includes loading/warmup.',
              'Projection is 500 x mean latency with the observed (often malformed) outputs; it excludes downloads, retries and format fixes. No full 500-request experiment was run.',
              comparison['preliminary_attempt'], comparison['semantic_quality']]
    (output / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=benchmark.DEFAULT_OUTPUT)
    main(parser.parse_args().output_dir)
