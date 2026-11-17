"""MaxSim and MeanSim over the same past-only assets and cached Qwen embeddings.

Reuse the pilot's text representation, MPS encoder and cosine implementation.
The persistent cache is intended for one writer at a time.
"""
from __future__ import annotations

from datetime import date
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from experiments import benchmark_embeddings as pilot
import torch

sys.path.insert(0, str(pilot.base.ROOT / 'shared/far-trans-llm'))
import ranker as teacher_ranker

MODEL = pilot.MODEL
REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'
METHODS = ('MaxSim', 'MeanSim')
DEFAULT_CACHE = pilot.base.DEFAULT_OUTPUT.parent / 'embedding_cache'


def cache_metadata():
    keys = ('device', 'dtype', 'attn_implementation', 'batch_size', 'torch_threads',
            'seed', 'padding_side', 'pooling', 'normalization_dtype',
            'embedding_dimensions', 'context_limit', 'instruction', 'chat_template', 'use_cache')
    return {'schema': 1, 'model': MODEL, 'revision': REVISION,
            'features': list(pilot.FIELDS.items()),
            'settings': {k: pilot.SETTINGS[k] for k in keys},
            'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers')},
            'encoder_source_sha256': hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest()}


def prepare_case(context, assets, events):
    """Use exactly the supplied 50 CBF candidates and <=5 most recent distinct Buys."""
    cutoff = context['cutoff']
    date.fromisoformat(cutoff)
    ids = teacher_ranker.candidate_ids(context)
    if len(ids) != 50:
        raise ValueError('Require exactly 50 CBF candidates')
    recent = pilot.base.last_distinct_buys(
        (event for event in events if event[1] in assets), cutoff, limit=5)
    if not recent:
        raise ValueError('Require a nonempty pre-cutoff Buy history')

    def record(isin):
        timestamp = assets.get(isin, {}).get('timestamp', '')[:10]
        if not timestamp or date.fromisoformat(timestamp) >= date.fromisoformat(cutoff):
            raise ValueError('Require known asset metadata strictly before cutoff')
        return teacher_ranker.asset_record(isin, assets, cutoff)

    bought = [dict(record(isin), last_buy_date=day) for isin, day in recent]
    return {'cutoff': cutoff, 'history': bought,
            'candidates': [record(isin) for isin in ids]}


class EmbeddingCache:
    """Cache by exact feature text within a pinned model/configuration namespace."""

    def __init__(self, directory=DEFAULT_CACHE, metadata=None):
        self.metadata = cache_metadata() if metadata is None else metadata
        key = pilot.base.digest(self.metadata)
        self.path = Path(directory) / f'{key}.pt'
        self.vectors = {}
        if self.path.exists():
            saved = torch.load(self.path, map_location='cpu', weights_only=True)
            if saved['metadata'] != self.metadata:
                raise ValueError('Embedding cache metadata mismatch')
            self.vectors = saved['vectors']
            for text, vector in self.vectors.items():
                if not isinstance(text, str):
                    raise ValueError('Invalid cache text key')
                self.validate(vector.unsqueeze(0))

    def validate(self, vectors):
        dimensions = self.metadata['settings']['embedding_dimensions']
        if (vectors.ndim != 2 or vectors.shape[1] != dimensions
                or vectors.dtype != torch.float32 or vectors.device.type != 'cpu'
                or not torch.isfinite(vectors).all().item()
                or not torch.allclose(vectors.norm(dim=1), torch.ones(len(vectors)), atol=1e-6, rtol=0)):
            raise ValueError('Cache requires finite, unit, CPU FP32 embedding vectors')

    def ensure(self, texts, encode):
        unique = list(dict.fromkeys(texts))
        missing = [text for text in unique if text not in self.vectors]
        if missing:
            vectors = encode(missing).detach().cpu()
            self.validate(vectors)
            if len(vectors) != len(missing):
                raise ValueError('Encoder returned the wrong number of embeddings')
            self.vectors.update((text, vector.clone()) for text, vector in zip(missing, vectors))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.tmp')
            try:
                torch.save({'metadata': self.metadata, 'vectors': self.vectors}, temporary)
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
        return {'unique_texts': len(unique), 'cache_hits': len(unique) - len(missing),
                'cache_misses': len(missing), 'cache_entries': len(self.vectors)}

    def get(self, texts):
        return torch.stack([self.vectors[text] for text in texts])


class QwenEmbedder:
    """Load only the pinned local snapshot, lazily on the first cache miss."""

    def __init__(self):
        self.model = self.tokenizer = None
        self.metadata = {'model': MODEL, 'revision': REVISION, 'model_loaded': False,
                         'load_seconds': 0., 'encoding_calls': []}

    def encode(self, texts):
        require_mps()
        if self.model is None:
            from huggingface_hub import snapshot_download
            from transformers import AutoModel, AutoTokenizer
            started = time.perf_counter()
            snapshot = snapshot_download(MODEL, revision=REVISION, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True,
                                                           trust_remote_code=False, padding_side='left')
            self.model = AutoModel.from_pretrained(snapshot, local_files_only=True,
                                                   trust_remote_code=False, dtype=torch.float16,
                                                   attn_implementation='sdpa').to('mps').eval()
            torch.mps.synchronize()
            if (self.model.config.hidden_size != 1024 or
                    any(p.device.type != 'mps' or p.dtype != torch.float16 for p in self.model.parameters())):
                raise RuntimeError('Expected a 1024-dimensional Qwen model on MPS FP16')
            self.metadata.update(model_loaded=True, snapshot=snapshot,
                                 load_seconds=time.perf_counter() - started)
        vectors, timing = pilot.encode(torch, self.model, self.tokenizer, texts)
        self.metadata['encoding_calls'].append(timing)
        return vectors.cpu()


def require_mps():
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS is unavailable; CPU fallback is disabled')
    torch.set_num_threads(pilot.SETTINGS['torch_threads'])
    torch.manual_seed(pilot.SETTINGS['seed'])


def rank_scores(candidate_ids, scores):
    if (len(candidate_ids) != 50 or len(set(candidate_ids)) != 50
            or any(not isinstance(i, str) for i in candidate_ids)
            or len(scores) != 50 or not all(torch.isfinite(torch.tensor(scores)).tolist())):
        raise ValueError('Expected 50 distinct candidate IDs and finite scores')
    order = sorted(range(50), key=lambda i: (-round(scores[i], 6), i))
    return [candidate_ids[i] for i in order[:10]]


def rerank_cases(cases, cache, encoder, *, device='mps'):
    """Return both methods together; scoring never receives truth or future events.

    CPU is accepted only for synthetic unit tests. The CLI always requires MPS.
    """
    if device == 'mps':
        require_mps()
    texts = []
    for case in cases:
        if not 1 <= len(case['history']) <= 5:
            raise ValueError('Require 1–5 distinct historical assets')
        if len({a['ISIN'] for a in case['history']}) != len(case['history']):
            raise ValueError('Duplicate historical asset')
        ids = [a['ISIN'] for a in case['candidates']]
        rank_scores(ids, [0.] * len(ids))
        for asset in case['history'] + case['candidates']:
            if not asset.get('timestamp') or asset['timestamp'][:10] >= case['cutoff']:
                raise ValueError('Metadata is missing or not before cutoff')
        if any(not a.get('last_buy_date') or a['last_buy_date'] >= case['cutoff'] for a in case['history']):
            raise ValueError('Buy date is missing or not before cutoff')
        texts.extend(pilot.asset_text(a) for a in case['history'] + case['candidates'])
    if not cases:
        raise ValueError('No cases to rerank')
    started = time.perf_counter()
    cache_stats = cache.ensure(texts, encoder.encode)
    cache_seconds = time.perf_counter() - started
    results = []
    for case in cases:
        started = time.perf_counter()
        history = cache.get([pilot.asset_text(a) for a in case['history']]).to(device)
        candidates = cache.get([pilot.asset_text(a) for a in case['candidates']]).to(device)
        with torch.inference_mode():
            if history.mean(dim=0).norm().item() <= 1e-12:
                raise ValueError('History centroid has zero norm')
            pairs, maximum, mean = pilot.similarity_scores(history, candidates)
        if device == 'mps':
            torch.mps.synchronize()
        if (not torch.isfinite(pairs).all().item() or pairs.abs().max().item() > 1 + 1e-6
                or not torch.isfinite(mean).all().item() or mean.abs().max().item() > 1 + 1e-6):
            raise ValueError('Invalid cosine scores')
        ids = [a['ISIN'] for a in case['candidates']]
        scores = dict(zip(METHODS, (maximum.cpu().tolist(), mean.cpu().tolist())))
        results.append({'history_count': len(case['history']), 'candidate_ids': ids,
                        'scores': scores,
                        'rankings': {name: rank_scores(ids, values) for name, values in scores.items()},
                        'scoring_end_to_end_seconds': time.perf_counter() - started})
    return results, {'cache': cache_stats, 'cache_seconds': cache_seconds,
                     'scoring_end_to_end_seconds': sum(r['scoring_end_to_end_seconds'] for r in results)}
