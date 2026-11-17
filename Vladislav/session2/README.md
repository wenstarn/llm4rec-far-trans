# Homework 2 · Neklyudov Vladislav

VibeCoding Team 7 · FAR-Trans frozen-encoder comparison

The research question is whether frozen Qwen3 metadata embeddings improve nDCG@10 over field-specific one-hot metadata for users with one to three distinct past buys. Both arms use category, subcategory, sector and industry, with `[MISSING]` in either arm for blank or unavailable values. Each retained pre-cutoff Buy ISIN contributes once. Both arms normalize each asset vector, average the user's history vectors, and rank the full starter-eligible universe by cosine to the centroid. Ties use the starter's pre-cutoff distinct-buyer popularity, then ascending ISIN. Repeat recommendations are allowed. The official starter determines which transactions and assets are eligible.

Qwen input is four labeled lines in the order above, with values quoted as JSON strings. Each unique description is encoded once with Ivan's frozen `Qwen/Qwen3-Embedding-0.6B` inference code at revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`. It uses MPS FP16, SDPA, batches of eight, last nonpadding token pooling, and FP32 output normalization. Both arms use the same FP64 scorer. The four-field cache has its own directory and binds the template, checkpoint, settings, and inference code. Keep one writer per cache directory.

## Run from the repository root

One-time setup for Apple Silicon, with Python 3.12, uv and MPS. The last line stages the pinned public model ahead of the data run. Keep `HF_HOME` consistent between staging and inference if you set it.

```bash
uv venv --python 3.12 Vladislav/session2/.venv
uv pip install --python Vladislav/session2/.venv/bin/python -r Ivan/session2/requirements-mps.txt
Vladislav/session2/.venv/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3-Embedding-0.6B", revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"])'
```

Synthetic tests use NumPy, temporary CSVs, and a stand-in encoder.

```bash
Vladislav/session2/.venv/bin/python -B -m unittest discover -s Vladislav/session2 -v
```

Set the extracted dataset path to the directory containing the four CSVs. `smoke` validates both rankings on up to three users and skips nDCG. Go directly to `evaluate` if time is tight. Use an empty output directory for each run.

```bash
DATA_DIR="/absolute/path/to/FAR-Trans"
Vladislav/session2/.venv/bin/python -B Vladislav/session2/run_embeddings.py smoke --data-dir "$DATA_DIR" --output-dir Vladislav/session2/outputs/smoke
Vladislav/session2/.venv/bin/python -B Vladislav/session2/run_embeddings.py evaluate --data-dir "$DATA_DIR" --output-dir Vladislav/session2/outputs/evaluate
```

## Evaluation outputs

Before encoding, `evaluate` runs the unchanged official starter on those CSVs. It checks the ordered 100-user list, cutoff, exclusive end date, eligible assets, repeat policy and the four CSV SHA-256 hashes against the starter's protocol. The two loading paths must also yield identical baseline rankings and metrics. All four aggregate baseline rows must match Ivan's CSV at full precision. Invalid rankings or parity mismatches stop the run.

- `predictions.csv`, `per_user.json`, `metrics.json` and `run_manifest.json` use the shared adapter format. Metrics include both nDCG@10 means and truth counts for all six methods.
- `protocol.json` records the verified starter protocol, configuration, source hashes, cache and timing details.
- `short_history.json` records the descriptive paired Qwen-minus-one-hot result, both arms' means, user counts and win/loss/tie counts for histories of one to three distinct buys.
- `manual_case.json` records a real pseudonymous case, preferring a short-history user with future truth. It includes the eligible truth, the Top-10 lists and each hit's position so the student can calculate nDCG independently.

Keep user-level outputs, the cache and the native session export local. Use the native JSONL for the individual submission and preserve the full session through the data run. The report and personal verification are separate from these automated checks.
