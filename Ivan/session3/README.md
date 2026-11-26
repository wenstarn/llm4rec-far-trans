# Session 3 — FAR-Trans reranking

The experiment compares four starter baselines, MaxSim, MeanSim and LLMRank on the same 500 warm users. The cutoff is 2021-06-01 with a 30-day evaluation window. The rerankers share the same ordered Top-50 CBF pools. MeanSim changes only the aggregation of history embeddings in the MaxSim ablation. The binary nDCG@10 table reports the mean over all users and over users with nonempty targets.

## Requirements and commands

Use Python 3.12 and the pins in `requirements.txt`, which reuse the HW2 environment. Inference requires Apple Silicon with PyTorch MPS and both model snapshots in the local Hugging Face cache. Tests and analysis of completed results run without inference.

Run these commands from the repository root with the HW2 virtual environment.

```bash
session3_python=Ivan/session2/.venv/bin/python
"$session3_python" -B -m unittest discover -s Ivan/session3/tests -t Ivan/session3 -v
"$session3_python" -B Ivan/session3/analyze.py --run-dir Ivan/session3/results/final_results --data-dir data

export HF_HOME="$PWD/Ivan/session2/outputs/hf_cache"
export HF_HUB_OFFLINE=1
"$session3_python" -B Ivan/session3/run.py smoke --data-dir data --output-dir Ivan/session3/outputs/smoke_new
"$session3_python" -B Ivan/session3/run.py evaluate --data-dir data --output-dir Ivan/session3/outputs/full500_new
```

Resume an interrupted evaluation by adding `--resume` to the same command. Evaluation publishes metrics after all 500 users have a valid result or a CBF fallback.

## Results

- `results/table_results.csv` — aggregate metrics for all seven methods.
- `results/failure_analysis.json` — aggregate comparison, candidate coverage, repetition, LLMRank fallback and timing.
- `results/final_results/` — local copy of the completed run, including per-user results and caches.
- `outputs/` — local run outputs, anonymized examples and manual-check files. `REPORT_FACTS.txt` contains local report notes.

Git includes the aggregate results and source files. The full archive, local outputs and report notes are ignored.
