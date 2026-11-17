# Session 2: MaxSim and MeanSim on FAR-Trans

Embedding-based reranking with **Qwen/Qwen3-Embedding-0.6B**, pinned to revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. Teacher code in `shared/` is used unchanged.

## Method

Both methods rerank the same 50 CBF candidates using up to five most recent
distinct purchased assets. Asset descriptions contain name, category, subcategory,
sector, industry and market; identifiers and dates are excluded from the text.
Last-token pooling produces 1024-dimensional, L2-normalized embeddings.

- **MaxSim:** maximum cosine similarity between a candidate and individual history assets.
- **MeanSim:** cosine similarity between a candidate and the normalized mean of history embeddings.

Evaluation settings: 100 warm users, cutoff **2021-06-01**, a **30-day** evaluation
horizon and Top-10. History and metadata precede the cutoff; future purchases
are used only for evaluation. Repeat recommendations are allowed. Score ties
after rounding to six decimals retain CBF order. Both methods share an embedding cache.

## Requirements and usage

Python **3.12** and an Apple Silicon Mac with PyTorch MPS are required.
Dependencies are pinned in `requirements-mps.txt`: torch, transformers, tokenizers,
huggingface-hub, numpy and psutil. Inference uses **MPS, FP16, SDPA, batch 8**;
normalization and cosine arithmetic use FP32. CPU fallback is disabled.
The pinned model must already exist in the local Hugging Face cache.

Using the existing virtual environment:

```bash
cd Ivan/session2
export PYTHONDONTWRITEBYTECODE=1

# Unit tests: synthetic inputs, no model inference.
.venv/bin/python -m unittest discover -s tests -t . -p 'test_*.py' -v

# Three users; no evaluation against future purchases.
.venv/bin/python run_embeddings.py smoke --output-dir outputs/smoke_new

# Full 100-user evaluation using the teacher's metrics.
.venv/bin/python run_embeddings.py evaluate --output-dir outputs/evaluate_new
```

Use a fresh output directory for each run. Optional `--data-dir` and `--cache-dir`
arguments override the default locations. Main code is in `embedding_reranker.py`
and `run_embeddings.py`; tests are in `tests/`, pilot scripts in `experiments/`.

## Results

- `results/table_results.csv`: aggregate metrics for four baselines, MaxSim and MeanSim; included in Git.
- `results/final_results/`: unchanged local archive of the completed experiment, including predictions, per-user metrics, protocols, checks and logs; ignored by Git.
- `outputs/`: original runs, intermediate experiments, model and embedding caches; ignored by Git.
- `REPORT_FACTS.txt`: local notes for the separate report; ignored by Git.
