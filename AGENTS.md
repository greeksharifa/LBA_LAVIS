# Repository Guidelines

## Project Structure & Module Organization
This is a LAVIS-based vision-language training and evaluation repository. Core code lives in `lavis/`: datasets in `lavis/datasets/`, models in `lavis/models/`, runners in `lavis/runners/`, and tasks in `lavis/tasks/`. Tests are in `tests/`. Experiment configs are in root YAML files and `configs/`, including SQ-InstructBLIP configs in `configs/sq_instructblip/`. Utility scripts live at the root and in `scripts/`. Keep generated outputs in `lavis/ywjang_output_*`, `output/`, or `archived/`, not in source directories.

## Build, Test, and Development Commands
Use `uv` for the main Python environment. Recreate the locked Python 3.10 environment including test dependencies with `uv sync --locked --python 3.10 --group dev`. Run main-project Python commands and tests with `uv run`.

- `uv run --python 3.10 python -m pytest tests --ignore=tests/models -q`: run repository regression tests without upstream model integration tests.
- `uv run --python 3.10 python -m pytest tests/test_vqa_introspect_fraction.py -q`: run a focused test.
- `tests/models/` contains upstream model integration tests that may download large weights and run inference. Run them separately from tmux `lavis:0 (run)` after the required model assets are prepared.
- `uv run --python 3.10 python train.py --cfg-path <config.yaml>`: launch single-process training.
- `DRY_RUN=1 scripts/run_sq_instructblip_fraction_experiments.sh`: print planned fraction runs.

## Coding Style & Naming Conventions
Use 4-space Python indentation and keep edits localized to the relevant dataset, model, runner, config, or script. Follow existing naming: classes use `PascalCase`, config keys use `snake_case`, and tests use `test_*.py`. Avoid broad style-only rewrites. For multi-GPU or multi-worker code, gate routine logs to one main process or rank, especially for vLLM or distributed training.

## Testing Guidelines
Tests use `pytest`. Add focused tests for dataset sampling, config plumbing, runners, and generation scripts when those areas change. Prefer deterministic fixtures and fixed seeds. The default regression command excludes `tests/models/`; run that upstream integration suite separately from the `lavis:0 (run)` tmux window after preparing its model weights.

## Commit & Pull Request Guidelines
Recent commits are short; keep messages concise but behavior-focused. Pull requests should include a summary, validation commands, linked issue or experiment context when relevant, and required model/data paths.

## Security & Configuration Tips
Do not commit private datasets, checkpoints, tokens, or large generated outputs. Keep machine-specific paths in configs or environment overrides. Document required model/data paths when experiments depend on them.

## Agent-Specific Instructions
Use Korean by default unless content clearly favors English. Be direct, concise, and technical; prefer doing work over proposing it. Preserve user changes, minimize diff churn, and ask before destructive actions. In reviews, list bugs, regressions, build breaks, and missing validation first. For library/API docs, setup, config, migration, or codegen, consult Context7 by default; for OpenAI API or Codex questions, prefer official OpenAI developer docs.


## Current Research Runbook

- Start with `docs/sq-instructblip-runbook.md` for canonical environment, paths, tmux, and provenance rules; use `scripts/README.md` for command-level detail. The SQ-InstructBLIP section in `README.md` is the human entry point; generic LAVIS setup/benchmark text below it is upstream reference.
- Run all research, training, and evaluation commands inside tmux session `lavis`, window `0` named `run`. Queue sequential experiment commands there and keep persistent logs under `/home/ywjang/.codex`.
- Use Python 3.10 for the main LAVIS project (`pyproject.toml` requires `>=3.8,<3.11`). Qwen2.5-VL generation uses a separate Python `>=3.11` environment; the current local `.venv-qwen` was provisioned with 3.11.4.
- Review `DRY_RUN=1` output before starting a scripted sweep. Do not send automatic Slack messages.
- Keep experiment results and model weights outside source commits. Record source HEAD and dirty state, config snapshot/hash, dataset/split/seed, model revision, checkpoint pair/index, and result paths. Do not invent an archive URL or remote object ID before one exists.
