# Curated research artifacts

This directory preserves compact results from the local uncertainty-v2
checkout without committing the bulk experiment tree.

## Included

- `evaluation_summaries/`: 650 completed runs. Each run retains its
  `evaluate.txt` and, when present, the matching `config.yaml`. The two legacy
  runs named `1` and `2` did not have configuration files.
- `api_results/batch/`: three non-duplicate OpenAI batch-result snapshots from
  2024-11-14 (`193332`, `194018`, and `194431`). Duplicate snapshots were
  omitted.
- `api_results/chatgpt_eval/`: the latest complete response tables for the
  2024-10-31 and 2024-11-01 runs, plus the available 2024-11-05 response table.

All included files were scanned for high-confidence API keys, private keys,
GitHub tokens, and AWS access-key identifiers before publication.

## Kept local and excluded from Git

- The approximately 16 GB raw `output/` tree, including `results_base.json`,
  plots duplicated per run, and the roughly two-million-file IGVLM result
  cache.
- Downloadable model checkpoints and model caches, including the local SeViLA
  and Flipped-VQA weights.
- Datasets, Python/build caches, logs, temporary notebooks, and scratch files.
- Duplicate API snapshots (`20241114_193300`, `20241114_194021`, and
  `20241114_194429`) whose retained counterparts contain the same files or a
  strict superset.

The excluded files were not deleted from the source checkout.
