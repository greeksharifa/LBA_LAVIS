# LBA Method

## Installation

- follow install videollava: https://github.com/PKU-YuanGroup/Video-LLaVA?tab=readme-ov-file#%EF%B8%8F-requirements-and-installation
- or Salesforce-lavis

```bash
pip install OmegaConf tqdm webdataset
```

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

CUDA_VISIBLE_DEVICES=0 python main.py --options datasets.num_data=100 
CUDA_VISIBLE_DEVICES=1 python main.py 
CUDA_VISIBLE_DEVICES=2 python main.py 
CUDA_VISIBLE_DEVICES=3 python main.py --options datasets.dataset_name="AOKVQA"
CUDA_VISIBLE_DEVICES=4 python main.py --options datasets.dataset_name="AOKVQA"
CUDA_VISIBLE_DEVICES=4 python main.py --options datasets.dataset_name="DramaQA" datasets.num_data=10 runner.recomposer_name="LanguageBind/Video-LLaVA-7B-hf" datasets.vis_root="/data1/AnotherMissOh/AnotherMissOh_videos/total/" runner.batch_size=1

# sevila
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --options datasets.dataset_name="DramaQA" datasets.num_data=10 runner.recomposer_name="sevila" runner.decomposer_name="xl" runner.answerer_name="Salesforce/blip2-flan-t5-xl" runner.batch_size=1 datasets.n_frms=32
CUDA_VISIBLE_DEVICES=3,4,5 python main.py --options datasets.dataset_name="DramaQA" runner.recomposer_name="sevila" runner.decomposer_name="xxl" runner.answerer_name="Salesforce/blip2-flan-t5-xxl" runner.batch_size=1 datasets.n_frms=32
CUDA_VISIBLE_DEVICES=2 python main.py --options datasets.dataset_name="DramaQA" datasets.num_data=10 runner.recomposer_name="sevila" datasets.vis_root="/data1/AnotherMissOh/AnotherMissOh_videos/total/" runner.batch_size=1 datasets.n_frms=32
```