# LBA Method

## Installation

- follow install videollava: https://github.com/PKU-YuanGroup/Video-LLaVA?tab=readme-ov-file#%EF%B8%8F-requirements-and-installation
- or Salesforce-lavis
- or SeViLA

```bash
cd SeViLA && pip install -e .
conda create -n LBA python=3.10
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install transformers==4.44.2 sentencepiece protobuf
pip install accelerate numpy pandas pyav opencv-python==4.7.0.72 
pip install matplotlib seaborn OmegaConf nltk tqdm webdataset decord
# pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless 
```

### Submodule
```bash
# repository clone
git clone https://github.com/greeksharifa/LBA_2024.git

# 서브모듈 정보를 기반으로 로컬 환경설정 파일을 만들어준다.
git submodule init

# 서브모듈의 리모트 저장소에서 데이터를 가져오고 Checkout을 한다.
git submodule update

# 서브모듈이 외부에서 업데이트가 되었을 때 현재 사용하려는 메인 깃에도 반영
git submodule update 

# 서브모듈 명령어 한 번에 실행하기
git submodule foreach 'git pull'
```

**docker**

```bash
# LBA
docker run -it --gpus all --name LBA_v2 --volume /home/ywjang/LBA_LAVIS_uncertainty_v2:/workspace --volume /data1:/data1 --volume /data2:/data2 pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
# BIVI
docker run --gpus all --name ywjang --shm-size 64G -i -t -p 22 -p 6006 -p 8888 -p 8889 -v /data:/data -v /home/ywjang/LBA_LAVIS_uncertainty_v2:/workspace nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 /bin/bash
```

## SeViLA

```python
# SeViLA/lavis/models/sevila_models/sevila.py line no 678. 
# after pred_ans = torch.argmax(pred_logits_qa, dim=-1).cpu().tolist()
# before out['output_text'] = pred_ans
return pred_ans, torch.exp(pred_logits_qa[:, 0]).cpu().tolist()
# or 
out['confidence'] = torch.exp(pred_logits_qa[:, 0]).cpu().tolist()
```


## Inference

```bash
# image baseline
CUDA_VISIBLE_DEVICES=5 python main_multi_subqa.py --verbose --options datasets.dataset_name="AOKVQA" runner.batch_size=32 runner.recomposer_name="Salesforce/blip2-flan-t5-xl" datasets.num_data=-1 runner.select_high_confidence=False runner.threshold_lba=False runner.vision_supple=False runner.num_sub_qa_generate=1


# video baseline
CUDA_VISIBLE_DEVICES=4 python main_multi_subqa.py --verbose --options datasets.dataset_name="VLEP" runner.batch_size=12 runner.recomposer_name="Salesforce/blip2-flan-t5-xl" datasets.num_data=-1 runner.select_high_confidence=False runner.threshold_lba=False runner.vision_supple=False runner.num_sub_qa_generate=1

# sevila
CUDA_VISIBLE_DEVICES=2 python main_multi_subqa.py --verbose --options datasets.dataset_name="NExTQA" runner.batch_size=6 runner.recomposer_name="sevila" datasets.num_data=-1 runner.select_high_confidence=True runner.train_recomposer_examplar=True runner.vision_supple=False runner.num_sub_qa_generate=1 datasets.n_frms=32 

# video_llava
CUDA_VISIBLE_DEVICES=4 python main_multi_subqa.py --verbose --options datasets.dataset_name="NExTQA" runner.batch_size=8 runner.recomposer_name="LanguageBind/Video-LLaVA-7B-hf" datasets.num_data=-1 runner.select_high_confidence=True runner.vision_supple=True use_pre_generated_sub_q=False runner.num_sub_qa_generate=1 datasets.n_frms=4


# video description
CUDA_VISIBLE_DEVICES=1 python main_multi_subqa.py --verbose --options datasets.dataset_name="" runner.batch_size=16 runner.recomposer_name="Salesforce/blip2-flan-t5-xl" datasets.num_data=-1 runner.select_high_confidence=True datasets.n_frms=8 runner.sub_mode="description" model.cache_dir="/data/LLMs/" datasets.root_dir="/data/video_datasets"


# instructblip
CUDA_VISIBLE_DEVICES=1 python main_multi_subqa.py --verbose --options datasets.dataset_name="DramaQA" runner.batch_size=6 runner.recomposer_name="Salesforce/instructblip-flan-t5-xl" runner.decomposer_name="Salesforce/blip2-flan-t5-xl" datasets.num_data=-1 runner.select_high_confidence=True runner.vision_supple=True runner.num_sub_qa_generate=1 datasets.n_frms=4

# use_pre_generated_sub_q
CUDA_VISIBLE_DEVICES=1 python main_multi_subqa.py --verbose --options datasets.dataset_name="DramaQA" runner.batch_size=12 runner.recomposer_name="Salesforce/blip2-flan-t5-xl" datasets.num_data=-1 runner.select_high_confidence=True runner.vision_supple=True runner.use_pre_generated_sub_q=True runner.num_sub_qa_generate=3 datasets.n_frms=4

# instructblip & use_pre_generated_sub_q
CUDA_VISIBLE_DEVICES=2 python main_multi_subqa.py --verbose --options datasets.dataset_name="DramaQA" runner.batch_size=6 runner.recomposer_name="Salesforce/instructblip-flan-t5-xl" datasets.num_data=-1 runner.select_high_confidence=True runner.vision_supple=True runner.use_pre_generated_sub_q=True runner.num_sub_qa_generate=3 datasets.n_frms=4


# visualize
python main_multi_subqa.py --options runner.visualize=True runner.output_dir="output/"
python main_multi_subqa.py --options runner.visualize=True runner.sub_mode="subqa" datasets.root_dir="/data1/" runner.select_high_confidence=True runner.max_conf_gap=False runner.output_dir="output/20240820_185932"
```



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