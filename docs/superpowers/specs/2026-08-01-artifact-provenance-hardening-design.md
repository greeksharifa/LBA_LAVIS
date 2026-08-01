# Artifact Provenance Hardening Design

## Goal

Close the remaining evaluator and generation-lineage gaps without changing the
normal UUID-based pipeline or requiring GPU validation.

## Manifest lineage

The manifest will retain a top-level `generation_history` object with exactly
all `STAGES` keys, each containing unique non-empty string IDs. Every visible
stage generation must be a member of its stage history. Every stage start
validates these invariants and records its new ID before replacing the active
status. Existing manifests without history remain writable: under the manifest
lock, the first start atomically seeds history from every currently visible
stage generation, validates it, and records the new ID. A generation ID already
present in that stage's history is rejected permanently, including after
invalidation. Malformed explicit history or missing visible membership fails
closed.

Completed stages must carry an exact `parent_generations` mapping matching
`STAGE_DEPENDENCIES`. Parentless stages require `{}`. Each dependent entry must
equal the current completed generation of that parent. C2R evaluation validates
this rule for every completed stage, so legacy completed artifacts without
lineage are rejected rather than silently trusted.

## Dataset selection provenance

C2R evaluation reads annotation JSON only from canonical absolute paths in
`annotation_paths_resolved`; configured paths are retained as labels and must
have a one-to-one correspondence with resolved paths. Resolved values must be
canonical absolute paths. Files are flattened in configured path-list order and
then JSON record order, and qids must be globally unique. Qid extraction mirrors
the dataset adapter: MMMU requires `question_id` and converts it to `str`.
Production evaluation does not accept a `qid` fallback. Missing or empty
identifiers fail closed, and test fixtures use the real MMMU source schema.

For `num_data=-1`, manifest qids must exactly equal all annotation qids in file
and record order. For positive `num_data`, evaluation reproduces the pipeline's
uniform `numpy.linspace` index selection when the limit is smaller than the
annotation count, otherwise it selects the full list. In both modes manifest
qids must equal the selected qids exactly in count and order. Other values fail
closed.

## Runtime provenance

`core_run_config` records `tensor_parallel_size`, `enforce_eager`, `swap_space`,
and normalized `limit_mm_per_prompt` using the same conversions as engine kwargs:
positive non-bool integer TP, strict boolean eager mode, finite non-negative
numeric swap converted to float, and either `None` or a string-keyed mapping of
non-negative non-bool integer modality limits. C2R rejects other representations
and compares all four canonical values exactly across dev and validation before
scoring.

## Validation

Tests first demonstrate missing/mismatched lineage acceptance, annotation
selection acceptance, generation ABA reuse, and runtime provenance omission.
The former one-record `num_data=-1` acceptance fixture is explicitly converted
to a rejection when its backing annotation contains additional qids.
Minimal implementation follows, then focused and full CPU-only test suites,
independent code review, and a clean committed worktree.
