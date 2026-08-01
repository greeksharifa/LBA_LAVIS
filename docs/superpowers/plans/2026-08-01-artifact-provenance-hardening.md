# Artifact Provenance Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fail closed on stale lineage, unverifiable annotation selection, generation-ID ABA reuse, and incompatible runtime provenance.

**Architecture:** Extend the existing manifest protocol with persistent per-stage generation history and runtime config fields. Reuse the central stage DAG in the evaluator and reconstruct selected qids from canonical resolved annotation JSON before accepting refined records.

**Tech Stack:** Python, JSON manifests, NumPy selection semantics, `unittest`.

---

### Task 1: Permanent generation identity

**Files:**
- Modify: `util/artifacts.py`
- Test: `tests/test_artifacts.py`

- [ ] Add a failing test that reuses a stage generation after a different generation replaced it.
- [ ] Add failing tests for malformed explicit history (wrong container, wrong/extraneous keys, invalid/duplicate IDs, and missing visible membership).
- [ ] Add a failing legacy-manifest test proving history seeds every visible generation ID before invalidation.
- [ ] Add a normal-path test proving an automatic UUID is recorded identically in active status/history and can complete.
- [ ] Run focused tests and confirm expected ABA failures.
- [ ] Add per-stage generation history to new manifests and initialize missing legacy history under the manifest lock.
- [ ] Reject reused IDs and preserve IDs of stages cleared by invalidation.
- [ ] Run focused tests to green.

### Task 2: Evaluator lineage and annotation selection

**Files:**
- Modify: `evaluation/c2r.py`
- Test: `tests/test_c2r_evaluation.py`

- [ ] Expand fixtures to create real annotation JSON and complete exact stage lineage, including MMMU `question_id` extraction.
- [ ] Add failing tests for missing/extra/mismatched parent lineage on completed stages.
- [ ] Add failing tests for incomplete full-data qids, wrong positive-limit qids/count/order, unsafe/noncanonical or duplicate resolved paths, path-list cardinality, malformed annotation containers/records, missing/empty/duplicate source qids, and invalid `num_data`.
- [ ] Run focused tests and confirm failures are caused by missing validation.
- [ ] Validate all completed stage lineage against the central DAG and active parent generations.
- [ ] Flatten canonical resolved annotation files in path/record order, require MMMU `question_id` and convert it to string with no production fallback, validate global uniqueness, and apply one full-list uniform selection exactly.
- [ ] Require manifest and record qids to equal the reconstructed selection in order.
- [ ] Run focused tests to green.

### Task 3: Runtime provenance compatibility

**Files:**
- Modify: `util/artifacts.py`
- Modify: `evaluation/c2r.py`
- Test: `tests/test_artifacts.py`
- Test: `tests/test_c2r_evaluation.py`

- [ ] Add failing tests for four missing runtime fields and cross-run mismatches before scoring.
- [ ] Add failing tests for invalid TP/eager/swap/limit types and ranges plus canonical core-config normalization.
- [ ] Run focused tests and confirm expected failures.
- [ ] Record normalized runtime values in `core_run_config`.
- [ ] Require and type-check the runtime fields in C2R and add them to pair compatibility.
- [ ] Run focused tests to green.

### Task 4: Verification and review

**Files:**
- Modify only files required by review findings.

- [ ] Run `/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m compileall -q` for changed Python files.
- [ ] Run `/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest discover -s tests -v` with no GPU command or GPU artifact mutation.
- [ ] Run `git diff --check`, `git status --short`, and inspect the scoped diff.
- [ ] Request independent code review and fix critical/important findings with TDD.
- [ ] Repeat full CPU verification and commit the final implementation.
