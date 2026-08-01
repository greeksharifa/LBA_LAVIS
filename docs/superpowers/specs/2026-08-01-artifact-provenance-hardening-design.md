# Artifact Provenance Hardening Design

## Goal

Close the remaining evaluator and generation-lineage gaps without changing the
normal UUID-based pipeline or requiring GPU validation.

## Manifest lineage

The manifest will retain a top-level per-stage generation history. Every stage
start records its generation ID before replacing the active status. Existing
manifests without history remain writable: the first start initializes history
from every currently visible stage generation. A generation ID already present
in that stage's history is rejected permanently, including after invalidation.
Malformed explicit history fails closed.

Completed stages must carry an exact `parent_generations` mapping matching
`STAGE_DEPENDENCIES`. Parentless stages require `{}`. Each dependent entry must
equal the current completed generation of that parent. C2R evaluation validates
this rule for every completed stage, so legacy completed artifacts without
lineage are rejected rather than silently trusted.

## Dataset selection provenance

C2R evaluation reads annotation JSON only from canonical absolute paths in
`annotation_paths_resolved`; configured paths are retained as labels and must
have a one-to-one correspondence with resolved paths. Each annotation file must
contain an ordered list of objects with unique non-empty `qid` values.

For `num_data=-1`, manifest qids must exactly equal all annotation qids in file
and record order. For positive `num_data`, evaluation reproduces the pipeline's
uniform `numpy.linspace` index selection when the limit is smaller than the
annotation count, otherwise it selects the full list. Other values fail closed.

## Runtime provenance

`core_run_config` records `tensor_parallel_size`, `enforce_eager`, `swap_space`,
and normalized `limit_mm_per_prompt`. C2R requires valid values and compares all
four fields across dev and validation manifests before scoring.

## Validation

Tests first demonstrate missing/mismatched lineage acceptance, annotation
selection acceptance, generation ABA reuse, and runtime provenance omission.
Minimal implementation follows, then focused and full CPU-only test suites,
independent code review, and a clean committed worktree.
