# LBA Method

## Installation

**docker**

```bash
# LBA
docker run -it --gpus all --name LBA_v2 --volume /home/ywjang/LBA_LAVIS_uncertainty_v2:/workspace --volume /data1:/data1 --volume /data2:/data2 pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
# BIVI
docker run --gpus all --name ywjang --shm-size 64G -i -t -p 22 -p 6006 -p 8888 -p 8889 -v /data:/data -v /home/ywjang/LBA_LAVIS_uncertainty_v2:/workspace nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 /bin/bash
```

## Inference

```bash
conda activate LBA_uncertainty_v2

CUDA_VISIBLE_DEVICES=0 python main.py --options datasets.num_data=100 runner.match1ok=True
CUDA_VISIBLE_DEVICES=1 python main.py 
CUDA_VISIBLE_DEVICES=2 python main.py --options runner.match1ok=True
CUDA_VISIBLE_DEVICES=3 python main.py --options datasets.dataset_name="AOKVQA"
CUDA_VISIBLE_DEVICES=4 python main.py --options datasets.dataset_name="AOKVQA" runner.match1ok=True
```