# Artifact Lineage Invalidation Design

## Problem

Starting an upstream stage currently changes only that stage to `running`.
Previously completed downstream stages remain marked complete, so dependency
loaders and C2R evaluation can accept artifacts derived from an older upstream
generation. Separately, C2R run-pair compatibility does not compare the
configured `num_data` selection policy, allowing a smoke dev run to be paired
with a full validation run.

## Design

Use explicit downstream invalidation in the manifest. The dependency DAG is:

- `subq` invalidates `suba` and `refined`.
- `suba` invalidates `refined`.
- `base` invalidates `refined` and remains independent of `subq`/`suba`.
- `refined` invalidates nothing.

`mark_stage_started` will update the active stage and all of its downstream
dependents under the existing manifest file lock. Each downstream status will
become incomplete with state `invalidated`, and its previous `generation_id`
will be removed. Therefore an interrupted or concurrent old downstream worker
cannot call `mark_stage_complete` successfully with its former generation ID.
Stale files may remain for recovery/debugging, but manifest validation rejects
them until their stage is regenerated and completed.

C2R compatibility will treat `num_data` as a run-pair selection-policy field.
Both role runs must have the same configured value. This permits the standard
MMMU full pair (`-1` for dev/150 and `-1` for validation/900) while rejecting a
smoke/full pair (`1` versus `-1`).

## Failure Safety

Invalidation happens before generation. If an upstream run fails or is
interrupted, downstream stages remain invalidated. Generation-guarded
completion rejects a worker whose stage was invalidated after it started.
Starting `base` does not disturb `subq` or `suba`, preserving independent
execution.

## Tests

- Artifact tests assert each DAG edge and base independence.
- A generation-race test starts a downstream stage, invalidates it through an
  upstream start, and verifies completion with the old generation ID fails.
- A pipeline test verifies an upstream rerun makes an existing refined result
  unavailable to the evaluator even if stale files remain.
- Evaluation tests accept `-1/-1` and reject `1/-1` before scoring.
