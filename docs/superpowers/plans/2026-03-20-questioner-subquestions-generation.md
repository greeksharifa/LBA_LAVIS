# Questioner Sub-Questions Generation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate 3 chained sub-questions per validation example for VQA-Introspect and AOK-VQA using the trained Questioner checkpoint, then save one JSON file per dataset.

**Architecture:** Reuse the existing InstructBLIP Questioner loading path and prompt template from the repo, but run Questioner-only distributed inference with lightweight dataset adapters and a final merge step. Keep model input processing identical to existing evaluation by using the same `blip_image_eval` and `blip_question` processors and the same generation parameters.

**Tech Stack:** Python, PyTorch distributed, existing LAVIS config/model loading, repo prompt templates, JSON export.

---

### Task 1: Add pure helper logic

**Files:**
- Create: `subqa/questioner_subquestions.py`
- Test: `tests/test_questioner_subquestions.py`

- [ ] Add prompt-chaining helpers that mirror the repo's `Questioner_MultipleSubQ` behavior for sequential generation.
- [ ] Add dataset-normalization helpers for VQA-Introspect and AOK-VQA val annotations.
- [ ] Add output-record and merge helpers for the required JSON schema.
- [ ] Add unit tests for prompt chaining, dataset normalization, and output formatting.

### Task 2: Add distributed generation entrypoint

**Files:**
- Create: `generate_questioner_subquestions.py`
- Modify: `subqa/__init__.py`

- [ ] Reuse the existing Questioner model-loading path and prompt file.
- [ ] Initialize distributed inference across up to 4 GPUs with per-rank dataset shards.
- [ ] Run 3-step chained generation for each batch and save rank-local shard JSON.
- [ ] Merge shards on rank 0 into one final JSON per dataset.

### Task 3: Verify and run

**Files:**
- Modify: none required unless verification exposes bugs

- [ ] Run focused tests for the new helper module.
- [ ] Run a small smoke job on both datasets to verify schema and prompt chaining.
- [ ] Run full validation generation and record final absolute output paths.
