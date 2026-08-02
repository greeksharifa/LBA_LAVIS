# LBA_2025

## Installation and verified runtime

The server profile verified for these commands is:

- Python 3.10.15 at `/home/ywjang/miniconda3/envs/qwen2vl/bin/python`
- torch 2.6.0+cu124 (PyTorch 2.6.0 with the CUDA 12.4 runtime)
- vLLM 0.8.2
- transformers 4.55.2

Install the repository requirements with the verified interpreter:

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m pip install -r requirements.txt
```

The absolute interpreter, cache, and GPU-wrapper paths below are
environment-specific. Adapt them if the checkout runs on another server. The
runtime setup in `model/vllm_config.py` installs the V0 engine
(`VLLM_USE_V1=0`) and spawn workers (`VLLM_WORKER_MULTIPROC_METHOD=spawn`) as
process defaults via `setdefault`, so explicit user settings still win. The
commands repeat those defaults inside the wrapper's `-- env` section for
reproducibility, while `model.enforce_eager=true` explicitly keeps eager
execution enabled.

The verified Qwen2.5-VL MMMU profile uses greedy generation with
`sampling_n=1`, so it sets CPU KV `swap_space=0`. This avoids the default
16 GiB-per-GPU swap reservation (64 GiB at TP=4) and prevents RAM exhaustion.

## Available models and benchmarks

- Models: `qwen2.5-vl-7b`, `qwen3-vl-8b`
- Benchmarks: MMLU, MMMU, DramaQA

## Execution modes

The four single-stage modes are `subq`, `suba`, `base`, and `refined`. Run them
in that dependency order when executing stages separately; all commands must
use the same run-defining options so their artifact manifests remain
compatible. These minimal examples use physical GPU 6 and a separate output
root:

```bash
/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=subq model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=4 runner.M=2 runner.K=4 runner.output_dir=output/single-stage
/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=suba model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=4 runner.M=2 runner.K=4 runner.output_dir=output/single-stage
/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=base model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=4 runner.M=2 runner.K=4 runner.output_dir=output/single-stage
/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=refined model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=4 runner.M=2 runner.K=4 runner.output_dir=output/single-stage
```

Set `runner.mode=multi_stage` to execute those four stages in order while
reusing one loaded model.

### One-item smoke run

This exact hierarchical Sub-QA smoke profile uses physical GPU 6, tensor
parallelism 1, and one question from MMMU's `val` split. It expands four
depth-1 questions and three depth-2 questions below each one, then projects
only the four selected depth-1 QA pairs into the existing refined stage. Its
`output/smoke-hierarchical` root is intentional: a one-qid manifest must never
collide with a full dev run in `output`.

The smoke and full commands assume the model is already cached in `HF_HOME`.
For the first model download, remove `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, then restore them after the cache is populated.

```bash
/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=multi_stage model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=4 runner.M=2 runner.K=4 runner.subqa_depth=2 'runner.branching_by_depth=[4,3]' runner.suba_M=2 runner.suba_K=3 runner.suba_confidence_type=token_min_prob runner.condition_on_direct_suba=true runner.output_dir=output/smoke-hierarchical
```

### Full MMMU dev and validation runs

In this repository, MMMU `dataset.split=val` selects `dev.json` (150
questions), and `dataset.split=test` selects `validation.json` (900
questions). `dataset.num_data=-1` selects the entire configured split. The two
full commands use physical GPUs 5, 6, 7, and 8 with tensor parallelism 4.

```bash
/home/ywjang/.codex/bin/run_gpu.sh 5,6,7,8 -- env HF_HOME=/home/ywjang/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=multi_stage model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=4 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=-1 runner.N=4 runner.M=2 runner.K=4 runner.output_dir=output
/home/ywjang/.codex/bin/run_gpu.sh 5,6,7,8 -- env HF_HOME=/home/ywjang/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=multi_stage model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=4 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=test dataset.num_data=-1 runner.N=4 runner.M=2 runner.K=4 runner.output_dir=output
```

## Run namespace and artifact guards

The depth-1 run namespace is
`<output-root>/<dataset>/<model>/<split>/N=<N>_M=<M>_K=<K>`. Dataset split and
the N/M/K signature therefore separate the full run directories. Hierarchical
Sub-QA adds one canonical child directory, `D=<depth>_H=<hash>`, whose hash
covers branching, internal M/K/confidence, direct conditioning, limits,
repair/batch settings, schema version, and fallback policy. Existing depth-1
artifacts keep their old path; an older N=5 artifact remains addressable only
when `runner.N=5` is explicitly supplied. A different `num_data` value is
checked by provenance but does not create another path, so partial and full
runs must use different output roots, as the smoke command does.

Each namespace contains `run_manifest.json`, which records the exact run
configuration including the normalized hierarchy contract, annotation paths,
selected qids, per-stage completion state, and each stage's `generation_id`.
`subq_outputs.json` stores the schema-v2 tree and `suba_outputs.json` stores
direct/candidate/selected answers by node while retaining depth-1 flat
projections. The final stage also writes
`refined_samples.json`. Producers reject incompatible manifests and require
completed dependencies before consuming their outputs.

The hierarchy contract is documented in the
[design](docs/superpowers/specs/2026-08-01-hierarchical-subqa-design.md), with
[implementation steps](docs/superpowers/plans/2026-08-01-hierarchical-subqa-plan.md)
and [completed execution evidence](docs/exec-plans/completed/2026-08-01-hierarchical-subqa.md).

Evaluation applies additional generation and provenance guards: the refined
stage must be completed, its `generation_id` must be stable while files are
read and must match every refined sample, and ordered qids and splits must
match the manifest exactly. A dev/validation pair must also agree on the model,
dataset, N/M/K, confidence configuration, and identical configured `num_data`
selection policy (`-1` on both full splits); overlapping qids or resolved
annotation paths are rejected.

## MMMU multiple-choice answer normalization

Before comparing a multiple-choice prediction with the gold label, MMMU uses a
deterministic, conservative parser. It accepts a whole-letter response,
parenthesized choices, a leading-delimited choice such as `A. explanation`,
explicit final-answer forms, Markdown-emphasized explicit conclusions such as
`**Answer: D**`, and both `\boxed{...}` and labeled boxed payloads such as
`\boxed{C. option text}`. Only the leading label in such a payload is used.
When a response contains multiple recognized conclusions, the last conclusion
controls the score; a later malformed explicit conclusion invalidates an
earlier one. Ambiguous outputs are rejected. Parsing never uses option text,
the candidate list, ground truth, or a random fallback.

## Leakage-free C2R evaluation

After both full runs complete, evaluate the exact run directories with:

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python scripts/evaluate_c2r.py --dev-run output/MMMU/qwen2.5-vl-7b/val/N=4_M=2_K=4 --validation-run output/MMMU/qwen2.5-vl-7b/test/N=4_M=2_K=4
```

The evaluator selects the C2R thresholds only on the dev role (`val`, the 150
question MMMU dev file). It then fixes those thresholds and applies them
unchanged to the validation role (`test`, the 900 question MMMU validation
file). Do not tune again on validation.

For a same-split diagnostic, the public in-sample evaluator searches and
applies thresholds on one run:

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python scripts/evaluate_c2r_in_sample.py --run output/MMMU/qwen2.5-vl-7b/test/N=4_M=2_K=4
```

This command tunes on the same validation split that it reports, so its result
is diagnostic only. The dev-tuned evaluation above is the leakage-free primary
result.

## Fresh branch-local hierarchical TP=1 MMMU validation results

The baseline is the backbone model's direct answer to the main question; it is
not the Flat pipeline. Raw hierarchy is the ungated refined answer from the
hierarchical run. `Dev-tuned gated` fixes thresholds selected on the separate
150-question dev split and is the leakage-free primary result. `Validation
in-sample` selects thresholds on the same validation split and is diagnostic
only. Deltas and paired transitions are relative to the direct baseline.

| Model | Direct baseline | Raw hierarchy | Dev-tuned gated (primary) | Validation in-sample (diagnostic) |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 454/900 (50.44%) | 448/900 (49.78%) | 449/900 (49.89%); tau1=0.7, tau2=-0.1; delta=-0.56 pp; W-to-C/C-to-W=46/51; 95% CI [-2.67, +1.67] pp | 461/900 (51.22%); tau1=0.7, tau2=0.2; delta=+0.78 pp; W-to-C/C-to-W=30/23; 95% CI [-0.78, +2.44] pp |
| Qwen3-VL-8B | 472/900 (52.44%) | 505/900 (56.11%) | 495/900 (55.00%); tau1=0.8, tau2=0.1; delta=+2.56 pp; W-to-C/C-to-W=25/2; 95% CI [+1.56, +3.67] pp | 505/900 (56.11%); tau1=1.0, tau2=-0.1; delta=+3.67 pp; W-to-C/C-to-W=36/3; 95% CI [+2.33, +5.00] pp |

This reevaluation only wrote evaluation reports: raw generation artifacts and
manifests remained byte-for-byte unchanged. The only recorded structural
caveat is that one Qwen2.5 validation parent expansion was partial; all 900
qids and all depth-1 projections were retained.

## Historical results

| Model | Mode | MMMU |
|---|---:|---:|
| Qwen3-VL-8B | base | 50.44 |
| Qwen3-VL-8B | refined | 53.44 |

These numbers came from the historical `/home/ywjang/C2R` workflow. They are
historical only and must not be presented as fresh performance from this
branch; run the commands above to obtain branch-local measurements.
