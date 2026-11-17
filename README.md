# VibeCoding Team 7

Members: Ivan, Vladislav, Alexsandr.

## Research hypotheses

Based on [Vladislav's Session 1 report](Vladislav/session1/session1_neklyudov_vladislav_report.pdf):

- **Vladislav:** Frozen Qwen3 embeddings improve nDCG@10 over one-hot metadata for users with 1–3 distinct previously purchased assets, keeping the cosine ranker fixed.
- **Ivan:** LLM reranking based on individual past purchases improves nDCG@10 over the original candidate ranking, potentially preserving interests lost through embedding averaging.
- **Alexsandr:** One-hot metadata can remain competitive with LLM embeddings due to coarse asset categories and the concentration of purchases in popular assets.

## Project structure

- `shared/` — instructor's starter, LLM baselines and documentation from [RecSys-LLMs](https://github.com/dryjins/RecSys-LLMs). [Version and license](shared/UPSTREAM.md).
- `data/` — local [FAR-Trans dataset](https://researchdata.gla.ac.uk/1658/), excluded from Git.
- `Ivan/`, `Vladislav/`, `Alexsandr/` — individual session folders (`session1/`, `session2/`, etc.).
