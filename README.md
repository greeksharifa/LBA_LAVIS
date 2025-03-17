# LBA_2025


---

## Installation

```bash
uv init
uv venv
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

---
## Available Models and Benchmarks

- Models:
  - qwen2.5-vl-7b
  - qwen3-vl-8b
- Benchmarks:
  - MMLU
  - MMMU

---
## Run

### sub-QA generation

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 uv run main.py --options runner.mode="subq" model.model_name="qwen2.5-vl-7b" dataset.dataset_name="MMLU"
CUDA_VISIBLE_DEVICES=1,2,3,4 uv run main.py --options runner.mode="suba" model.model_name="qwen2.5-vl-7b" dataset.dataset_name="MMLU" dataset.num_data=10
```

### Inference

```bash
```

### Visualize

```bash
```

---

## Results

| Model    | MMLU | MMLU-Pro | StrategyQA | MMMU | EgoSchema |
|----------|------|----------|------------|------|-----------|
|          |      |          |            |      |           |

