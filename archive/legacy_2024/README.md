# 2024 Legacy SubQA Snapshot

This directory preserves the standalone `LBA_SubQA` checkout's ignored legacy artifacts before that checkout was retired.

- `legacy_generate_and_api_examples.py` contains the March 2024 prototype and two embedded API response examples used for offline debug/reference.
- `legacy_run.sh` records the historical invocation and its site-specific `/data1` output path.

No API credential is included. If adapting the prototype for execution, provide `OPENAI_API_KEY` through the environment and update the dataset/output paths for the current machine. The files are archival references, not maintained entrypoints.
