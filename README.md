# LBA Method

## Installation

## Inference

```bash
conda activate LBA_uncertainty_v2

CUDA_VISIBLE_DEVICES=0 python main.py --options datasets.num_data=100 runner.match1ok=True
CUDA_VISIBLE_DEVICES=1 python main.py 
CUDA_VISIBLE_DEVICES=2 python main.py --options runner.match1ok=True
CUDA_VISIBLE_DEVICES=3 python main.py --options datasets.dataset_name="AOKVQA"
CUDA_VISIBLE_DEVICES=4 python main.py --options datasets.dataset_name="AOKVQA" runner.match1ok=True
```