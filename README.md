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
  - DramaQA

---
## Run

### sub-QA generation

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 uv run main.py --options runner.mode="subq" model.model_name="qwen2.5-vl-7b" dataset.dataset_name="MMLU"
CUDA_VISIBLE_DEVICES=1,2,3,4 uv run main.py --options runner.mode="suba" model.model_name="qwen2.5-vl-7b" dataset.dataset_name="MMLU" dataset.num_data=10
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 uv run main.py --options runner.mode="base" model.model_name="qwen2.5-vl-7b" dataset.dataset_name="MMLU"
CUDA_VISIBLE_DEVICES=1,2,3,4 uv run main.py --options runner.mode="refined" model.model_name="qwen2.5-vl-7b" dataset.dataset_name="MMLU"
```

### Visualize

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 uv run main.py --options runner.mode="refined" model.model_name="qwen2.5-vl-7b" dataset.dataset_name="MMLU" runner.visualize_only=True
```

---

## Results

| Model        |  mode   | MMMU  |
|--------------|---------|-------|
| Qwen3-VL-8B  |  base   | 50.44 |
| Qwen3-VL-8B  | refined | 53.44 |

