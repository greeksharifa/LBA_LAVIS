# Questioner Sub-Questions Generation: Implementation Record

This document records the implementation planned on 2026-03-20. The active usage guide is [scripts/README.md](../../../scripts/README.md).

**Goal:** Generate three chained sub-questions for VQA-Introspect and AOK-VQA validation examples using the trained Questioner checkpoint and save one JSON file per dataset.

**Implementation present in the repository:**

- `subqa/questioner_subquestions.py` contains prompt chaining, annotation normalization, output-record, and shard-merge helpers.
- `tests/test_questioner_subquestions.py` contains unit tests for those helpers. This plan does not record a test execution result.
- `generate_questioner_subquestions.py` loads the Questioner, shards records across distributed ranks, writes rank-local results, and merges the dataset output on rank 0.
- The entrypoint uses the repository's InstructBLIP processors and prompt configuration. Dataset choices, local path overrides, and invocation examples are in the runbook.

**Verification evidence and limits:**

- No GPU smoke run or full validation generation, with final output paths, is recorded in this plan.
- GPU generation remains unverified by this historical record. Do not infer successful end-to-end generation from the source and unit-test files alone.
- The code and tests may have changed since this plan was written; use the current source and runbook as the operational reference.
