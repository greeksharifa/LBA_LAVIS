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
/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=subq model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=5 runner.M=2 runner.K=8 runner.output_dir=output/single-stage
/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=suba model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=5 runner.M=2 runner.K=8 runner.output_dir=output/single-stage
/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=base model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=5 runner.M=2 runner.K=8 runner.output_dir=output/single-stage
/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=refined model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=5 runner.M=2 runner.K=8 runner.output_dir=output/single-stage
```

Set `runner.mode=multi_stage` to execute those four stages in order while
reusing one loaded model.

### One-item smoke run

This exact smoke profile uses physical GPU 6, tensor parallelism 1, and one
question from MMMU's `val` split. Its `output/smoke` root is intentional: a
one-qid manifest must never collide with a full dev run in `output`.

The smoke and full commands assume the model is already cached in `HF_HOME`.
For the first model download, remove `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`, then restore them after the cache is populated.

```bash
/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=multi_stage model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=5 runner.M=2 runner.K=8 runner.output_dir=output/smoke
```

### Full MMMU dev and validation runs

In this repository, MMMU `dataset.split=val` selects `dev.json` (150
questions), and `dataset.split=test` selects `validation.json` (900
questions). `dataset.num_data=-1` selects the entire configured split. The two
full commands use physical GPUs 5, 6, 7, and 8 with tensor parallelism 4.

```bash
/home/ywjang/.codex/bin/run_gpu.sh 5,6,7,8 -- env HF_HOME=/home/ywjang/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=multi_stage model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=4 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=val dataset.num_data=-1 runner.N=5 runner.M=2 runner.K=8 runner.output_dir=output
/home/ywjang/.codex/bin/run_gpu.sh 5,6,7,8 -- env HF_HOME=/home/ywjang/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V1=0 VLLM_WORKER_MULTIPROC_METHOD=spawn /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=multi_stage model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=4 model.enforce_eager=true dataset.dataset_name=MMMU dataset.split=test dataset.num_data=-1 runner.N=5 runner.M=2 runner.K=8 runner.output_dir=output
```

## Run namespace and artifact guards

The run namespace is
`<output-root>/<dataset>/<model>/<split>/N=<N>_M=<M>_K=<K>`. Dataset split and
the N/M/K signature therefore separate the full run directories. A different
`num_data` value is checked by provenance but does not create another path, so
partial and full runs must use different output roots, as the smoke command
does.

Each namespace contains `run_manifest.json`, which records the exact run
configuration, annotation paths, selected qids, per-stage completion state,
and each stage's `generation_id`. The final stage also writes
`refined_samples.json`. Producers reject incompatible manifests and require
completed dependencies before consuming their outputs.

Evaluation applies additional generation and provenance guards: the refined
stage must be completed, its `generation_id` must be stable while files are
read and must match every refined sample, and ordered qids and splits must
match the manifest exactly. A dev/validation pair must also agree on the model,
dataset, N/M/K, confidence configuration, and sample-count policy; overlapping
qids or resolved annotation paths are rejected.

## Leakage-free C2R evaluation

After both full runs complete, evaluate the exact run directories with:

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python scripts/evaluate_c2r.py --dev-run output/MMMU/qwen2.5-vl-7b/val/N=5_M=2_K=8 --validation-run output/MMMU/qwen2.5-vl-7b/test/N=5_M=2_K=8
```

The evaluator selects the C2R thresholds only on the dev role (`val`, the 150
question MMMU dev file). It then fixes those thresholds and applies them
unchanged to the validation role (`test`, the 900 question MMMU validation
file). Do not tune again on validation.

## Historical results

| Model | Mode | MMMU |
|---|---:|---:|
| Qwen3-VL-8B | base | 50.44 |
| Qwen3-VL-8B | refined | 53.44 |

These numbers came from the historical `/home/ywjang/C2R` workflow. They are
historical only and must not be presented as fresh performance from this
branch; run the commands above to obtain branch-local measurements.
