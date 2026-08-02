# MMMU Multiple-Choice Normalization Design

## Goal

Restore benchmark-appropriate multiple-choice response parsing without changing
raw model generations, confidence values, open-ended scoring, or hierarchy
selection. The fix must make branch-local Qwen3 results comparable to the
historical C2R workflow while avoiding answer guessing.

## Root cause

`dataset/mmmu_eval.py` currently compares the complete generated response with
the gold option letter. This rejects ordinary model outputs such as `A. $6`.
The strict behavior was introduced with the MMMU open-answer scorer and differs
from both the earlier repository cleanser and the official MMMU workflow, which
parse a generated multiple-choice response into an option letter before exact
evaluation.

## Parsing contract

Add a deterministic `parse_multiple_choice_response(response)` boundary in
`dataset/mmmu_eval.py`. It returns one case-normalized option letter or `None`.
It must not receive the gold answer.

Apply these rules in order:

1. Accept a response consisting only of one letter with optional parentheses,
   Markdown emphasis, whitespace, and terminal punctuation, such as `A`,
   `(A).`, or `**A.**`.
2. Find explicit conclusions such as `answer is A`, `answer: option A`,
   `final answer is (C)`, `**Answer: D**`,
   `Final Answer: **B. option text**`, `\\boxed{C}`, and
   `\\boxed{C. option text}`. Markdown headings, emphasis, and blockquote
   wrappers do not change the selected leading label. If multiple conclusion
   events occur, select the last one by source position.
3. Accept a leading, delimited option label followed by option text or an
   explanation, such as `A. $6`, `A) explanation`, `(A) explanation`, or
   `A: option text`.
4. Otherwise return `None`, making the response incorrect.

The parser is case-insensitive. It may return letters `A` through `Z`; the
separate gold comparison determines correctness.

The parser models recognized conclusions as ordered valid, invalid, or ignored
events. A later malformed strong marker such as `Final answer: unknown`
invalidates an earlier answer. Bounded explanatory reuse such as
`The answer is based on the calculation` is ignored rather than treated as a
new conclusion. Coordinated alternatives remain invalid through Markdown and
blockquote wrappers, including `**A** or **B**` and `(A), (B)`. A recognized
answer marker may use `is:` punctuation and may be followed by a bounded
explanation introducer such as `because`, `since`, or `Explanation:`; the same
bare unmarked text remains invalid.

The parser must reject ambiguous prose such as `A because it is correct`, a
letter occurring incidentally inside a sentence, responses with no explicit or
leading answer format, empty values, malformed values, numeric boxed payloads,
and multiple coordinated labels. It must not match option text, use the gold
answer to extract a candidate, or randomly select a fallback.

`evaluate_multiple_choice()` parses the prediction, normalizes the gold as an
exact option letter, and compares the two. MMMU open-ended evaluation is
unchanged. Both `MMMU.get_score()` and `evaluation.c2r` continue to share this
single scorer boundary.

## Artifact and evaluation behavior

Raw `base_outputs.json`, `refined_outputs.json`, `refined_samples.json`, token
confidence, manifests, and generation lineage remain unchanged. The evaluator
recomputes correctness and threshold selection from those raw records, so the
existing Qwen2.5 and Qwen3 dev/validation generations are re-evaluated rather
than regenerated.

Re-evaluate both models in parallel on CPU:

- dev-selected thresholds applied unchanged to validation;
- validation-selected thresholds reported separately as an explicitly
  in-sample diagnostic;
- paired bootstrap intervals and transition counts recomputed with the new
  scorer.

GPU generation is not part of this fix because no generation input, prompt,
model adapter, or sampling setting changes. If later evidence requires fresh
generation, use four independent TP=1 runs: GPU 5/6 for Qwen2.5 dev/validation
and GPU 7/8 for Qwen3 dev/validation. Do not use TP=4 or manually merge qid
shards because those paths are slower or violate the current artifact contract.

## Tests

Use TDD to add focused tests before production changes.

- Accept whole, parenthesized, punctuated, Markdown-emphasized answers.
- Accept leading delimited labels with option text.
- Accept explicit answer/final-answer/boxed forms and choose the last explicit
  conclusion when earlier reasoning conflicts.
- Accept Markdown-emphasized and labeled boxed singleton conclusions while
  rejecting coordinated alternatives across emphasis and blockquote wrappers.
- Reject `A because ...`, incidental letters, missing conclusions, and empty
  responses.
- Preserve all existing open-ended normalization cases.
- Prove `MMMU.get_score()` and `evaluation.c2r.prepare_samples()` receive the
  same normalized correctness.
- Run the focused MMMU/C2R suites, then full unit-test discovery.

## Documentation and acceptance

README must describe the response-normalization boundary and distinguish fresh
branch-local results from the historical table. Acceptance requires:

- focused red/green regression evidence;
- full CPU suite passing;
- fresh Qwen2.5 and Qwen3 evaluation reports from current artifacts;
- Qwen3 direct accuracy returning to the historical range rather than the
  strict-parser 33.33%;
- no raw generation or manifest mutation during re-evaluation.
