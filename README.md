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

## Run

### sub-QA generation

```bash
uv run main.py --options runner.mode="subqa" model.model_name="qwen2.5-vl-7b" dataset.dataset_name="MMLU"
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

