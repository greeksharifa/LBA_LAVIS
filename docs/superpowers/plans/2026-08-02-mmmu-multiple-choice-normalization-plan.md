# MMMU Multiple-Choice Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse conservative, explicit MMMU multiple-choice response formats before exact scoring, then recompute Qwen2.5/Qwen3 dev and validation results from preserved raw artifacts.

**Architecture:** Keep raw generations and confidence selection unchanged. Add one deterministic parser at `dataset/mmmu_eval.py`, which is already shared by the MMMU adapter and C2R evaluator, and validate it at both boundaries. Re-run CPU evaluation against existing completed manifests; use GPU generation only if artifact validation shows a generation-side defect.

**Tech Stack:** Python 3.10, `re`, `unittest`, existing `evaluation.c2r` report pipeline, JSON artifacts.

---

## File map

- Modify `dataset/mmmu_eval.py`: own all MMMU response parsing and scoring.
- Modify `tests/test_mmmu_open.py`: parser behavior and MMMU adapter regression coverage.
- Modify `evaluation/c2r.py`: expose reproducible in-sample evaluation and paired transition counts.
- Create `scripts/evaluate_c2r_in_sample.py`: write a validation-tuned diagnostic report without private imports.
- Modify `tests/test_c2r_evaluation.py`: prove C2R uses the shared parser and validate both report modes.
- Modify `README.md`: document normalization and fresh corrected measurements.
- Recompute ignored JSON reports under existing Qwen2.5/Qwen3 output roots; do not mutate generation artifacts or manifests.

### Task 1: Add failing parser and integration tests

**Files:**
- Modify: `tests/test_mmmu_open.py:228`
- Modify: `tests/test_c2r_evaluation.py`

- [x] **Step 1: Add focused accepted-format tests**

Add tests through the real `MMMU.get_score()` boundary for:

```python
for prediction in (
    "A",
    "(A).",
    "A. $6",
    "A) explanation",
    "(A) explanation",
    "**A.** $6",
    "The answer is A.",
    "answer: option A",
    "Final answer: option A",
    "final answer is (A)",
    "final answer is a",
    r"\boxed{A}",
):
    self.assertEqual(1, dataset.get_score(prediction, "A", "multiple-choice"))
```

- [x] **Step 2: Add precedence and rejection tests**

Assert the last explicit conclusion wins, while ambiguous/non-answer prose is
not guessed:

```python
prediction = "The answer is B. After checking, final answer: C."
self.assertEqual(1, dataset.get_score(prediction, "C", "multiple-choice"))
self.assertEqual(0, dataset.get_score(prediction, "B", "multiple-choice"))

for prediction in (
    "A because it is correct",
    "A result was observed",
    "",
    "No conclusion",
    None,
):
    self.assertEqual(0, dataset.get_score(prediction, "A", "multiple-choice"))
```

- [x] **Step 3: Add a C2R boundary test**

Create a normal multiple-choice `refined_samples` record with `base_answer` set
to `A. option text` and the selected refined answer set to `Final answer: A`.
Call `prepare_samples()` without a custom scorer and assert both
`base_correct` and `refined_correct` are true.

- [x] **Step 4: Run RED tests**

Run:

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest \
  tests.test_mmmu_open tests.test_c2r_evaluation
```

Expected: the new formatted-answer assertions fail because the current scorer
requires a whole-string single letter; existing open-answer tests remain green.

### Task 2: Implement the conservative parser

**Files:**
- Modify: `dataset/mmmu_eval.py:90-107`

- [x] **Step 1: Add dedicated compiled patterns**

Create separate patterns for whole answers, explicit answer markers, boxed
answers, and leading delimited labels. Keep matching case-insensitive and
require a non-letter boundary after every captured option letter.

- [x] **Step 2: Implement the parser without the gold answer**

Implement this control flow:

```python
def parse_multiple_choice_response(response: str):
    if not isinstance(response, str) or not response.strip():
        return None
    whole = _WHOLE_CHOICE.fullmatch(response)
    if whole:
        return whole.group(1).lower()
    explicit = [
        (match.start(), next(group for group in match.groups() if group))
        for pattern in (_EXPLICIT_CHOICE, _BOXED_CHOICE)
        for match in pattern.finditer(response)
    ]
    if explicit:
        return max(explicit, key=lambda item: item[0])[1].lower()
    leading = _LEADING_CHOICE.match(response)
    if leading:
        return next(group for group in leading.groups() if group).lower()
    return None
```

Do not add option-text matching, random fallback, candidate-list dependencies,
or gold-dependent extraction.

- [x] **Step 3: Route multiple-choice evaluation through the parser**

Parse only the prediction. Normalize the gold with the existing exact-letter
normalizer, and return false when prediction parsing returns `None`. This avoids
malformed `None == None` matches. Leave open-answer code byte-for-byte
unchanged.

- [x] **Step 4: Run GREEN focused tests**

Run the Task 1 command. Expected: all focused tests pass.

- [x] **Step 5: Run the full suite**

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expected: zero failures.

- [x] **Step 6: Commit the parser and tests**

```bash
git add dataset/mmmu_eval.py tests/test_mmmu_open.py tests/test_c2r_evaluation.py
git commit -m "fix: normalize MMMU choice responses"
```

### Task 3: Make both evaluation modes reproducible

**Files:**
- Modify: `evaluation/c2r.py:642-758`
- Create: `scripts/evaluate_c2r_in_sample.py`
- Modify: `tests/test_c2r_evaluation.py`

- [x] **Step 1: Add failing paired-transition assertions**

Extend report tests with samples containing one base-wrong/gated-correct and one
base-correct/gated-wrong transition. Require every split report to contain:

```python
"paired_transitions": {
    "wrong_to_correct": 1,
    "correct_to_wrong": 1,
}
```

- [x] **Step 2: Add a failing in-sample report test**

Specify a public `evaluate_run_in_sample()` API. Assert that it searches all
231 threshold pairs, reports every max-accuracy threshold, applies the existing
deterministic tie break, labels its method
`fixed_validation_grid_search_in_sample`, and keeps the same provenance,
bootstrap, and transition fields as pair evaluation.

- [x] **Step 3: Add a failing CLI test**

Run `scripts/evaluate_c2r_in_sample.py --run <fixture>` and require atomic
`c2r_evaluation_validation_tuned.json` output with no temporary file left.

- [x] **Step 4: Run tests to verify RED**

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest tests.test_c2r_evaluation
```

Expected: failures for the missing public API, transition fields, and CLI.

- [x] **Step 5: Implement transition counts and the public report builder**

Add `paired_transitions` in `_split_report()`. Add
`evaluate_run_in_sample(run_directory, *, scorer=evaluate_answer,
bootstrap_seed=42, bootstrap_count=10000)` using `load_run()`,
`prepare_samples()`, `search_thresholds()`, and `_split_report()`. Build the
grid metadata in the public function rather than the CLI so direct callers and
the CLI produce an identical schema.

- [x] **Step 6: Implement the thin CLI**

Parse `--run` and optional `--output`, call `evaluate_run_in_sample()`, and use
`write_report_atomic()`. Do not import private names from `evaluation.c2r`.

- [x] **Step 7: Run GREEN tests and commit**

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest tests.test_c2r_evaluation
git add evaluation/c2r.py scripts/evaluate_c2r_in_sample.py tests/test_c2r_evaluation.py
git commit -m "feat: add reproducible in-sample C2R evaluation"
```

### Task 4: Re-evaluate preserved model outputs

**Files:**
- Rewrite ignored reports only:
  - `output/hierarchical-full-repair-tp1/.../c2r_evaluation.json`
  - `output/hierarchical-full-repair-tp1/.../c2r_evaluation_validation_tuned.json`
  - `output/hierarchical-full-qwen3-tp1/.../c2r_evaluation.json`
  - `output/hierarchical-full-qwen3-tp1/.../c2r_evaluation_validation_tuned.json`

- [x] **Step 1: Snapshot generation artifact hashes and manifests**

Hash both models' dev/validation `run_manifest.json`, `base_outputs.json`,
`refined_outputs.json`, and `refined_samples.json`. These hashes are the
non-mutation acceptance baseline.

- [x] **Step 2: Recompute dev-selected reports in parallel on CPU**

Run two `scripts/evaluate_c2r.py` processes concurrently, one per model, with
the existing Qwen2.5 and Qwen3 dev/validation run directories.

- [x] **Step 3: Recompute validation-selected diagnostic reports in parallel**

Run two `scripts/evaluate_c2r_in_sample.py` processes concurrently to rewrite
each `c2r_evaluation_validation_tuned.json` with the explicit
`fixed_validation_grid_search_in_sample` provenance label.

- [x] **Step 4: Verify metrics and transitions**

Report direct, raw hierarchy, dev-selected validation, validation-selected
validation, thresholds, switch counts, paired confidence intervals, and
wrong-to-correct/correct-to-wrong transitions for both models.

Require Qwen3 validation direct accuracy to be at least 50% and no more than
55%, rather than the strict-parser `300/900 = 33.33%`. The preliminary expected
count under the approved parser is `472/900 = 52.44%`; investigate rather than
accept completion if the fresh result falls outside that range.

- [x] **Step 5: Prove generation artifacts were not changed**

Recompute Task 4 Step 1 hashes and require exact equality. Validate both
manifests, qid ordering, stage lineage, tree counts, and report sample counts.

### Task 5: Refresh README and close verification

**Files:**
- Modify: `README.md:118-140`
- Test: `tests/test_readme_commands.py`

- [x] **Step 1: Document the scoring boundary**

Explain that MMMU multiple-choice generations are deterministically parsed from
explicit answer formats before exact option-letter comparison; no random or
ground-truth-based fallback is used.

- [x] **Step 2: Add fresh branch-local results**

Add the corrected Qwen2.5/Qwen3 validation comparison. Keep the historical
table clearly labeled, and label validation-tuned figures as in-sample rather
than leakage-free generalization results.

- [x] **Step 3: Run README and evaluator tests**

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest \
  tests.test_readme_commands tests.test_mmmu_open tests.test_c2r_evaluation
```

- [x] **Step 4: Run closing verification**

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest discover -s tests -p 'test_*.py'
git diff --check
git status --short
```

- [x] **Step 5: Request independent code review and address blockers**

Review for parser false positives, open-answer regression, C2R scorer drift,
artifact mutation, and misleading validation-tuned reporting.

- [x] **Step 6: Commit documentation and final fixes**

```bash
git add README.md tests/test_readme_commands.py
git commit -m "docs: report normalized MMMU results"
```

## Implementation outcome

Completed on 2026-08-02. The implementation added gold-independent MMMU
multiple-choice normalization, reproducible dev-selected and validation
in-sample evaluators, paired transition counts, and branch-local Qwen2.5/Qwen3
results. Final verification covered 216 unit tests, exact report recomputation,
four-run manifest/tree/projection validation, and SHA-256 equality for all 16
source generation and manifest artifacts before and after evaluation.

The only observed hierarchy degradation was one allowed partial descendant
expansion in the 900-question Qwen2.5 validation run; all qids and all four
depth-1 projections were retained. Validation-tuned numbers remain diagnostic
because threshold selection and reporting use the same split.
