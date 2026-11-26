"""Run the instructor's LLM-Rec Encoder on Apple Silicon using its original prompt."""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
from time import perf_counter

import run_experiments as experiment
from run_experiments import Evaluation, HERE, evaluate, hw2, protocol_for, teacher
from runtime import LocalGenerator, LoggedGenerator

MODEL = 'Qwen/Qwen3-1.7B'
REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
MAX_NEW_TOKENS = 160


class MPSGenerator(LocalGenerator):
    """Use Ivan's local MPS loading pattern and preserve shared greedy decoding."""

    def __init__(self, batch_size):
        import torch
        import transformers

        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable')
        torch.set_num_threads(4)
        self.torch = torch
        self.device = 'mps'
        opts = dict(revision=REVISION, local_files_only=True, trust_remote_code=False)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, padding_side='left', **opts)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.bfloat16, **opts).to(self.device).eval()
        torch.mps.synchronize()
        if any(p.device.type != 'mps' or p.dtype != torch.bfloat16 for p in self.model.parameters()):
            raise RuntimeError('Expected model parameters on MPS in BF16')
        self.metadata = {'model': MODEL, 'requested_revision': REVISION,
                         'resolved_revision': self.model.config._commit_hash, 'device': self.device,
                         'dtype': str(torch.bfloat16), 'torch_version': torch.__version__,
                         'transformers_version': transformers.__version__,
                         'tokenizer_class': type(self.tokenizer).__name__, 'do_sample': False,
                         'thinking': False, 'context_limit': 8192, 'padding_side': 'left',
                         'attn_implementation': self.model.config._attn_implementation,
                         'batch_size': batch_size, 'max_new_tokens': MAX_NEW_TOKENS,
                         'torch_threads': 4, 'use_cache': self.model.generation_config.use_cache,
                         'eos_token_id': self.model.generation_config.eos_token_id,
                         'pad_token_id': self.tokenizer.eos_token_id}
        self.batch_seconds = []
        self.completed_prompts = 0

    def generate_batch(self, prompts, limit=MAX_NEW_TOKENS):
        formatted = [self.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True,
            enable_thinking=False) for prompt in prompts]
        inputs = self.tokenizer(formatted, padding=True, return_tensors='pt')
        width = inputs['input_ids'].shape[1]
        lengths = inputs['attention_mask'].sum(dim=1).tolist()
        if width + limit > self.metadata['context_limit']:
            raise ValueError('Prompt exceeds the fixed context budget')
        inputs = inputs.to(self.device)
        self.torch.mps.synchronize()
        started = perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False,
                                          pad_token_id=self.tokenizer.eos_token_id)
        self.torch.mps.synchronize()
        elapsed = perf_counter() - started
        self.batch_seconds.append(elapsed)
        self.completed_prompts += len(prompts)
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        results = []
        for tokens, size in zip(outputs[:, width:].cpu().tolist(), lengths, strict=True):
            consumed = next((i + 1 for i, token in enumerate(tokens) if token in eos), len(tokens))
            results.append({'text': self.tokenizer.decode(tokens[:consumed], skip_special_tokens=True),
                            'input_tokens': size, 'output_tokens': consumed,
                            'seconds': elapsed / len(prompts), 'batch_wall_seconds': elapsed,
                            'batch_prompts': len(prompts)})
        return results


class BatchedLoggedGenerator(LoggedGenerator):
    """Keep the shared response log and cache schema while batching cache misses."""

    def __init__(self, backend, output_dir, prompts):
        super().__init__(backend, output_dir)
        self.batch_size = backend.metadata['batch_size']
        self.pending = iter(dict.fromkeys(prompt for prompt in prompts if not self.cache_path(prompt).exists()))
        self.ready = {}

    def cache_path(self, prompt):
        descriptor = {'prompt': prompt, 'settings': self.backend.metadata, 'max_new_tokens': MAX_NEW_TOKENS}
        key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        return self.cache / (key + '.json')

    def generate(self, prompt, max_new_tokens=MAX_NEW_TOKENS):
        if max_new_tokens != MAX_NEW_TOKENS:
            raise ValueError('Reference generation must use the documented 160-token limit')
        target = self.cache_path(prompt)
        if not target.exists() and prompt not in self.ready:
            batch = list(islice(self.pending, self.batch_size))
            if not batch or batch[0] != prompt:
                raise ValueError('Prompt order differs from the shared encoder')
            self.ready.update(zip(batch, self.backend.generate_batch(batch), strict=True))
            print(f'Generated {self.backend.completed_prompts} uncached prompts '
                  f'({self.backend.batch_seconds[-1]:.2f}s for {len(batch)})', flush=True)
        hit = target.exists()
        if hit:
            result = json.loads(target.read_text())
            self.hits += 1
        else:
            result = self.ready.pop(prompt)
            temp = target.with_suffix('.tmp')
            temp.write_text(json.dumps(result, ensure_ascii=False) + '\n', encoding='utf-8')
            temp.replace(target)
            self.calls += 1
        record = {'prompt': prompt, 'settings': self.backend.metadata, 'max_new_tokens': max_new_tokens,
                  'response': result, 'cache_hit': hit, 'cache_key': target.stem}
        with self.log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        return result


def smoke(args, documents, backend, load_seconds):
    experiment.fresh_dir(args.output_dir)
    prompts = [teacher.paraphrase_prompt(text) for text in islice(documents.values(), 3)]
    serial = [LocalGenerator.generate(backend, prompt, MAX_NEW_TOKENS) for prompt in prompts]
    batched = backend.generate_batch(prompts)
    keys = ('text', 'input_tokens', 'output_tokens')
    agree = [all(a[key] == b[key] for key in keys) for a, b in zip(serial, batched, strict=True)]
    if not all(result['output_tokens'] and result['text'].strip() for result in serial + batched):
        raise RuntimeError('Expected decoded paraphrases from real MPS generation')
    hw2.write_json(args.output_dir / 'forward_pass.json', {'settings': backend.metadata,
                   'model_load_seconds': load_seconds, 'matches_serial': agree,
                   'serial': serial, 'batched': batched})
    print(f'Real MPS forward-pass check, serial versus batched agreement: {agree}', flush=True)


def run(args, data, documents, backend, timings):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists():
        raise ValueError('Choose a new directory for another completed run')
    official = hw2.read_json(args.controls_dir / 'regression100/protocol.json')
    protocol = protocol_for(official, data, data.users)
    if protocol != hw2.read_json(args.controls_dir / 'development2000/protocol.json'):
        raise ValueError('Reference protocol differs from the control run')

    started = perf_counter()
    prompts = {i: teacher.paraphrase_prompt(text) for i, text in documents.items()}
    with (out / 'encoder_prompts.jsonl').open('w', encoding='utf-8') as stream:
        for i, prompt in prompts.items():
            stream.write(json.dumps({'asset_id': i, 'prompt': prompt}, ensure_ascii=False) + '\n')
    generator = BatchedLoggedGenerator(backend, out, prompts.values())
    timings['prompt_export_and_cache_index_seconds'] = perf_counter() - started
    started = perf_counter()
    generated, failures = teacher.generate_views(documents, generator, on_invalid='fallback')
    timings['inference_and_generation_io_seconds'] = perf_counter() - started
    timings['mps_generate_seconds'] = sum(backend.batch_seconds)
    started = perf_counter()
    original = teacher.tfidf(documents)
    combined = teacher.combine_views(original, teacher.tfidf(generated))
    timings['tfidf_and_view_concatenation_seconds'] = perf_counter() - started
    started = perf_counter()
    eligible = sorted(documents)
    contexts = {uid: {'eligible_asset_ids': eligible} for uid in data.users}
    result = teacher.predict(contexts, data.history, original, combined, data.popularity)
    timings['both_reference_recommendation_arms_seconds'] = perf_counter() - started
    predictions = experiment.transpose(result, data.users, ('original_text_cbf', 'llmrec_encoder'))
    controls_file = args.controls_dir / 'development2000/predictions.csv'
    with controls_file.open(newline='') as stream:
        expected = [(r['user_id'], r['rank'], r['ISIN']) for r in csv.DictReader(stream)
                    if r['method'] == 'original_text_cbf']
    actual = [(uid, str(rank), isin) for uid, ranking in predictions['original_text_cbf'].items()
              for rank, isin in enumerate(ranking, 1)]
    if actual != expected:
        raise ValueError('Instructor TF-IDF control differs from the independent ranking run')
    _, metrics = evaluate(out, data, protocol, predictions)
    independent = hw2.read_json(args.controls_dir / 'development2000/metrics.json')['original_text_cbf']
    if metrics['original_text_cbf'] != independent:
        raise ValueError('Instructor TF-IDF metrics differ from the independent run')
    hw2.write_json(out / 'run.json', dict(generator.metadata, timings=timings, assets=len(documents),
                   batches=len(backend.batch_seconds), unique_prompts=len(set(prompts.values())),
                   invalid_outputs=len(failures), fallbacks=sum(f['fallback'] for f in failures),
                   valid_asset_paraphrases=len(documents) - len(failures), invalid_policy='original source text',
                   failures=failures, original_control_rankings_match=True,
                   response_seconds_scope='Batch GPU wall time divided by the number of prompts in that batch'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('smoke', 'evaluate'))
    parser.add_argument('--data-dir', type=Path, default=hw2.ROOT / 'data/FAR-Trans')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--controls-dir', type=Path, default=HERE / 'outputs/controls')
    parser.add_argument('--batch-size', type=int, default=24)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('Expected a positive batch size')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    data = Evaluation.load(args.data_dir)
    started = perf_counter()
    documents = {i: teacher.source_text(data.assets[i]) for i in sorted(data.assets)}
    timings = {'data_load_hashes_selection_popularity_seconds': data.preparation_seconds,
               'documents_seconds': perf_counter() - started}
    started = perf_counter()
    backend = MPSGenerator(args.batch_size)
    timings['model_load_seconds'] = perf_counter() - started
    if args.stage == 'smoke':
        smoke(args, documents, backend, timings['model_load_seconds'])
    else:
        run(args, data, documents, backend, timings)


if __name__ == '__main__':
    main()
