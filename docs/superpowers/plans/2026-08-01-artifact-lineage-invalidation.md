# Artifact Lineage Invalidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent regenerated upstream stages or mismatched sampling policies from making stale C2R artifacts appear valid.

**Architecture:** The manifest owns a dependency DAG and records parent generation IDs on every stage start. Start atomically validates parents and invalidates descendants; completion atomically validates generation/lineage, promotes canonical JSON through a callback under the manifest lock, and marks complete. C2R run pairing compares the configured `num_data` selection policy.

**Tech Stack:** Python 3.10, `unittest`, JSON manifests, `fcntl` file locking, OmegaConf fixtures.

---

### Task 1: Manifest DAG and lineage guards

**Files:**
- Modify: `tests/test_artifacts.py`
- Modify: `util/artifacts.py`

- [ ] **Step 1: Write failing lifecycle tests**

Add tests covering all exact invalidation edges, base independence, rejection of an old invalidated generation, rejection of an outdated parent snapshot, strict rejection of missing/extra/empty parent mappings, non-empty completion generation IDs, prevention of an old artifact-writer callback after a newer generation completes, and a raising artifact writer that leaves the manifest running/incomplete.

Update the existing completion helper/tests to start stages and pass their exact generation IDs. Change the concurrent completion case from dependent `subq`/`suba` to independent `subq`/`base`, preserving its lock-update purpose without contradicting the DAG.

- [ ] **Step 2: Run tests and verify RED**

Run: `/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest -v tests.test_artifacts`

Expected: failures because starts do not invalidate descendants or validate parent generations, and completion still permits missing/stale generations and runs artifact writes outside its guard.

- [ ] **Step 3: Implement minimal manifest protocol**

In `util/artifacts.py`, define:

```python
STAGE_DEPENDENCIES = {
    "subq": (),
    "suba": ("subq",),
    "base": (),
    "refined": ("subq", "suba", "base"),
}
STAGE_DEPENDENTS = {
    "subq": ("suba", "refined"),
    "suba": ("refined",),
    "base": ("refined",),
    "refined": (),
}
```

Extend `mark_stage_started(..., parent_generations=None)` to validate exact parents under `_manifest_lock`, store `parent_generations`, set the active generation, and clear descendant generation IDs while marking them invalidated. Dependent `suba` and `refined` starts must receive all and only the keys in `STAGE_DEPENDENCIES`, with non-empty generation IDs matching currently completed parents; reject `None`, missing/extra keys, empty IDs, or mismatches. Only parentless `subq` and `base` may accept `None` or `{}`. Extend `mark_stage_complete(..., generation_id, artifact_writer=None)` to require a non-empty ID, revalidate active generation and stored parents under the same lock, invoke `artifact_writer` only after validation, then persist completion. If the writer raises, do not persist completion. Require completed dependencies to have a non-empty generation ID.

- [ ] **Step 4: Run artifact tests and verify GREEN**

Run the command from Step 2; expected all artifact tests pass.

### Task 2: Dependency snapshots and guarded pipeline promotion

**Files:**
- Modify: `tests/test_dataset_dependencies.py`
- Modify: `tests/test_pipeline.py`
- Modify: `dataset/base_dataset.py`
- Modify: `pipeline.py`

- [ ] **Step 1: Write failing integration tests**

Assert that `BaseDataset` exposes exact dependency generation IDs, that a completion guard failure leaves the prior canonical output unchanged, that a refined second-file writer failure leaves the manifest running so evaluation rejects the partial promotion, and that restarting/crashing `subq` after a complete run leaves stale refined files present but makes `evaluation.c2r.load_run` reject the run.

Add a snapshot-bearing fake dataset helper for pipeline tests. For standalone dependent-stage tests, prepare a manifest with completed parent stages and return their exact generation mapping. For `run_multi_stage`, have the fake loader read the completed parent generation IDs produced by prior stages. Replace every plain-list fake loader used for `suba` or `refined`; parentless `subq`/`base` loaders remain unchanged.

- [ ] **Step 2: Run integration tests and verify RED**

Run: `/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest -v tests.test_dataset_dependencies tests.test_pipeline`

Expected: failures because dependency snapshots are not exposed, canonical files are written before the completion guard, and upstream starts leave refined complete.

- [ ] **Step 3: Implement minimal integration**

Have `BaseDataset._load_stage_dependencies` save `{stage: generation_id}` from each validated manifest. For dependent modes, require the dataset loader to expose the exact mapping and pass it into `mark_stage_started`; parentless modes pass `{}`. Build an `artifact_writer` closure in `run_stage` that writes the stage output and, for `refined`, `refined_samples.json`; pass it to `mark_stage_complete` rather than writing canonical files first.

- [ ] **Step 4: Run integration tests and verify GREEN**

Run the command from Step 2; expected all dependency and pipeline tests pass.

### Task 3: C2R selection-policy compatibility

**Files:**
- Modify: `tests/test_c2r_evaluation.py`
- Modify: `evaluation/c2r.py`
- Modify: `README.md`
- Modify: `tests/test_readme_commands.py`

- [ ] **Step 1: Write failing policy tests**

Parameterize the C2R manifest fixture with `num_data`. Add one test that accepts the standard `-1/-1` full policy and one that rejects `1/-1` before invoking the scorer. Tighten the README contract to require an explicit identical-`num_data` explanation.

- [ ] **Step 2: Run policy tests and verify RED**

Run: `/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest -v tests.test_c2r_evaluation tests.test_readme_commands`

Expected: the `1/-1` pair is incorrectly evaluated and the README lacks the exact policy wording.

- [ ] **Step 3: Implement minimal compatibility check**

Add `num_data` to `_COMPATIBILITY_FIELDS`, remove its duplicate from `_REQUIRED_CONFIG_FIELDS`, and clarify README wording that both role runs must use identical configured `num_data` (`-1/-1` for the standard full pair).

- [ ] **Step 4: Run policy tests and verify GREEN**

Run the command from Step 2; expected all C2R and README tests pass.

### Task 4: Verification and commit

**Files:**
- Verify all changed Python files and repository tests.

- [ ] **Step 1: Compile changed Python files**

Run: `/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m compileall -q util/artifacts.py pipeline.py dataset/base_dataset.py evaluation/c2r.py tests/test_artifacts.py tests/test_dataset_dependencies.py tests/test_pipeline.py tests/test_c2r_evaluation.py tests/test_readme_commands.py`

- [ ] **Step 2: Run the full CPU test suite**

Run: `/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest discover -s tests -v`

- [ ] **Step 3: Review the diff**

Run: `git diff --check` and inspect only scoped files. Confirm no GPU commands ran and no GPU outputs changed.

- [ ] **Step 4: Commit**

Stage only the scoped code, tests, README, spec, and plan; commit with `fix: invalidate stale stage artifacts`.
