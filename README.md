<p align="center">
    <br>
    <img src="docs/_static/logo_final.png" width="400"/>
    <br>
<p>

<div align="center">
  <a href="https://github.com/salesforce/LAVIS/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/salesforce/LAVIS.svg" /></a>
  <a href="https://opensource.salesforce.com/LAVIS/index.html">
  <img alt="docs" src="https://github.com/salesforce/LAVIS/actions/workflows/docs.yaml/badge.svg"/>
  <a href="https://opensource.org/licenses/BSD-3-Clause">
  <img alt="license" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg"/>
  </a> 
  <a href="https://pepy.tech/project/salesforce-lavis">
  <img alt="Downloads" src="https://pepy.tech/badge/salesforce-lavis">
  </a>
</div>

<div align="center">
<a href="https://opensource.salesforce.com/LAVIS//latest/benchmark.html">Benchmark</a>,
<a href="https://arxiv.org/abs/2209.09019">Technical Report</a>,
<a href="https://opensource.salesforce.com/LAVIS//latest/index.html">Documentation</a>,
<a href="https://github.com/salesforce/LAVIS/tree/main/examples">Jupyter Notebook Examples</a>,
<a href="https://blog.salesforceairesearch.com/lavis-language-vision-library/">Blog</a>
</div>

# LAVIS - A Library for Language-Vision Intelligence

## DramaQA sub QA 데이터셋 생성 관련
### 개선 가능한 부분
- 성능 평가에서 `SentenceTransformer` (ST)의 embedding 비교를 통한 모델 answer 결정 대신 각 답안에 대한 Perplexity를 평가하여 그 중 가장 낮은 답안을 모델의 answer로 정한다.
  - 기존 ST의 embedding을 이용하는 방식
    - ST 모델로 인한 추가적인 연산이 필요
    - ST 모델의 성능에 의존한다는 한계
    - 모델의 answer가 주어진 답안 중 어떠한 것과도 관련이 없다면 (모델의 answer가 틀리지 않더라도) 이 방식으로 평가하는 것이 의미가 없음
  - Perplexity는 모델이 생각하는 각 답안의 확률과 직접적인 연관이 있기 때문에 더 직접적인 정확도 평가가 가능함
  - 각 답안을 `text_output`으로 준 각 loss와 동치하므로 구현이 간단하다
- 단일 이미지 입력이 아닌 이미지 시퀀스 (비디오) 입력
  - Causal relationship과 관련된 질문을 답할 수 있게 함
  - 카메라 전환, occlusion 등으로 단일 프레임에는 포착되지 않는 정보를 retrieve 할 수 있음
- PEFT (Parameter-Efficient Fine-Tuning)을 통한 fine tuning
  - 적은 학습 시간에 높은 일반화 성능을 내도록 fine tuning 가능
  - 일정 수준의 오버피팅 방지
  - 적용 예시: Prefix tuning, LoRA 등

### 기존에 지향하던 연구 방향
- 비디오를 입력으로 각종 모델 (BLIP, SAM 등)을 통하여 Scene Graph를 생성하여 해당 Scene Graph를 기반으로 sub QA를 생성하며 질의응답에 대한 추론을 진행하는 방법론
- 구상하였던 Scene Graph 생성 방식과 거의 동일한 논문 존재
  - https://arxiv.org/abs/2310.01356
- 시간 및 결과 성능 부족으로 진행 중단

### 주요 클래스
- `DramaQASQDataset`
    - `lavis/datasets/datasets/video_vqa_datasets.py`
    - 모델에 전달될 입력, 타겟 출력 반환
    - 반환값
        - `image`: 입력 이미지, 현재는 비디오 클립의 전체 프레임 중 랜덤한 프레임 하나를 가져옴
        - `text_input`: 입력 prompt
            - 구체적인 형식은 ./prompts.json 및 해당 코드 참고
        - `text_output`: 타겟 출력
        - `text_input`, `text_output`은 `prompt_type`이 `“questioner”`, `“answerer”`, 혹은 `“reasoner”`인지에 따라 형식이 결정되며 `prompt_type`은 랜덤하게 설정됨
            - `“questioner”`: 입력 이미지와 main question, (선택적으로) sub QA 예시를 `text_input`으로써 모델이 sub-question을 생성하도록 `text_output`이 결정됨
            - `“answerer”`: 입력 이미지와 main question, (선택적으로) sub QA 예시, 답변해야 할 sub-question을 `text_input`으로써 모델이 sub-answer를 생성하도록 `text_output`이 결정됨
            - `“reasoner”`: 입력 이미지와 main question, (1개 이상의) sub QA를 `text_input`으로써 모델이 main answer을 생성하도록 `text_output`이 결정됨
- `DramaQASQTask`
    - `lavis/tasks/vqa.py`
    - `SentenceTransformer`로 모델의 최종 출력과 각 답안의 embedding을 cosine similarity로 비교하여 정확도를 평가

### 기타 수정 사항
  - `Blip2VicunaInstructSQ`
    - `lavis/models/blip2_models/blip2_vicuna_instruct_SQ.py`
    - Inference 시 sub-question 출력의 다양성을 위해 다음의 설정을 사용
      - `sub_question_kwargs["use_nucleus_sampling"] = True`
      - `sub_question_kwargs["temperature"] = 2.0`
      - `"questioner"` 단계에서만 해당 변경된 세팅을 사용하며 나머지 단계에서는 기존 세팅을 사용
        - `"answerer"`, `"reasoner"`의 sub-answer와 main answer는 높은 확률을 가진 답변을 주는 것을 강조하기 위해
## InstructBLIP Fine-Tuning을 위한 수정사항(VQA-Introspect)

- https://opensource.salesforce.com/LAVIS/latest/tutorial.datasets.html 참고
- 참고: qar(Questioner-Answerer-Reasoner)은 이전 이름으로 추후 (SQ_InstructBLIP로) 변경 예정
- 아래 내용은 InstructBLIP-vicuna7b를 수정하고 새로운 dataset, task를 정의한 경우에 대한 내용
- 새로운 task 추가(vqa_introspect_captioning): `lavis/tasks/captioning.py` or `lavis/tasks/vqa.py`
  - `__init__.py`에도 등록(위 사이트 참조)
- 새로운 데이터셋 추가(vqa_introspect_qar_caption)
  - sub_qa를 불러오는 방식에 따라 2가지로 구분함
  - `lavis/datasets/datasets/vqa_introspect_datasets.py`: role(Q, A, R)에 따라 sub_qa를 불러오는 방식이 다름
    - 원래는 prompt 적용을 모델 안에서 하는 것이 맞으나 임시로 이렇게 작성
  - `lavis/datasets/datasets/vqa_introspect_datasets_test.py`: 최종 모델인 SQ_InstructBLIP을 테스트하기 위한 데이터셋
  - dataset builder(`lavis/datasets/builders/vqa_builder.py, __init__.py`)에도 등록 완료
  - dataset config yaml 파일로 새로 생성하여 등록(`lavis/configs/datasets/vqa_introspect/defaults.yaml`)
- configs(yaml) 파일 수정: `./instructBLIP_FT_vicuna7b.yaml`
  - baseline model test를 위한 yaml 파일 수정: `./instructBLIP_FT_vicuna7b_test.yaml`
  - SQ_InstructBLIP 모델 test를 위한 yaml 파일 수정: `./instructBLIP_FT_vicuna7b_qar.yaml`
  - yaml 파일의 datasets나 task만 변경해도 되지만 빠른 테스트를 위하 파일을 분리한 것. 합쳐도 상관 없음
  - 주의할 점:
    - yaml 파일의 `datasets:` 아래의 데이터셋 이름은 dataset builder의 register 이름과 일치(class 이름 아님)
    - dataset은 여러 개 지정 가능
    - `run: task:`는 `tasks/<blabla>.py`의 원하는 task와 register 이름과 일치
- 모델 수정(`blip2_vicuna_instruct.py, blip2_vicuna_instruct_qar.py`)
  - `lavis/models/blip2_models/blip2_vicuna_instruct.py` 수정:
    - model role(Questioner, Answerer, or Reasoner) 추가
    - sq_prompts 추가: role 별로 적용하는 prompts가 다를 수 있음(추가 예정)
    - log를 위한 cnt 추가
  - `lavis/models/blip2_models/blip2_vicuna_instruct_qar.py` 수정:
    - Questioner-Answerer-Reasoner framework를 generate() 함수에서 구현
    - module weight 통합 시 deprecate 시킬 예정
- 학습: `train.py / train.sh`, baseline test: `test.py / test.sh`, SQ_Instruct test: `test_qar.py / test_qar.sh`
- `./prompts.json`에 현재 SQ_InstructBLIP에 적용할 prompt 저장
- 데이터 root 폴더는 `lavis/configs/default.yaml`에서 지정
- KT의 보안에 의해 huggingface 등에서 모델 다운로드가 안 되므로 모델을 local에 다운로드하여 사용하기 위한 코드 변경:
  - `lavis/models/eva_vit.py`
  - `lavis/models/blip2_models/blip2.py`
  - `lavis/models/blip2_models/blip2_vicuna_instruct.py`
  - `lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml`
  - 추가로 기존 데이터셋(ex. A-OKVQA) 등을 사용할 때에도 그에 맞는 `configs/datasets/이름`에 들어가서 경로 수정해주어야 함
- 기타
  - `init_distributed_mode(cfg.run_cfg)` 바로 다음에 `setup_logger()`을 위치시켜야 `logging.info()`가 제대로 작동
  - evaluate(test) 할 때는 `runner.train()` 대신 `runner.evaluate(skip_reload=True)`을 사용
  - yaml 파일의 `output_dir:`은 `lavis/` 폴더 아래에 저장됨

ywjang 수정:

- output: `/home/ywjang/LBA/LBA_LAVIS/lavis/ywjang_output/`


## What's New: 🎉 
  * [Model Release] July 2023, released implementation of **BLIP-Diffusion** <br>
  [Paper](https://arxiv.org/abs/2305.06500), [Project Page](https://github.com/salesforce/LAVIS/tree/main/projects/blip-diffusion), [Website](https://dxli94.github.io/BLIP-Diffusion-website/)
  > A text-to-image generation model that trains 20x than DreamBooth. Also facilitates zero-shot subject-driven generation and editing.
  * [Model Release] May 2023, released implementation of **InstructBLIP** <br>
  [Paper](https://arxiv.org/abs/2305.06500), [Project Page](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip)    
  > A new vision-language instruction-tuning framework using BLIP-2 models, achieving state-of-the-art zero-shot generalization performance on a wide range of vision-language tasks.
  * [Model Release] Jan 2023, released implementation of **BLIP-2** <br>
  [Paper](https://arxiv.org/abs/2301.12597), [Project Page](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb)
  > A generic and efficient pre-training strategy that easily harvests development of pretrained vision models and large language models (LLMs) for vision-language pretraining. BLIP-2 beats Flamingo on zero-shot VQAv2 (**65.0** vs **56.3**), establishing new state-of-the-art on zero-shot captioning (on NoCaps **121.6** CIDEr score vs previous best **113.2**). In addition, equipped with powerful LLMs (e.g. OPT, FlanT5), BLIP-2 also unlocks the new **zero-shot instructed vision-to-language generation** capabilities for various interesting applications!
  * Jan 2023, LAVIS is now available on [PyPI](https://pypi.org/project/salesforce-lavis/) for installation!
  * [Model Release] Dec 2022, released implementation of **Img2LLM-VQA** (**CVPR 2023**, _"From Images to Textual Prompts: Zero-shot VQA with Frozen Large Language Models"_, by Jiaxian Guo et al) <br>
  [Paper](https://arxiv.org/pdf/2212.10846.pdf), [Project Page](https://github.com/salesforce/LAVIS/tree/main/projects/img2llm-vqa), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/img2llm-vqa/img2llm_vqa.ipynb)
  > A plug-and-play module that enables off-the-shelf use of Large Language Models (LLMs) for visual question answering (VQA). Img2LLM-VQA surpasses Flamingo on zero-shot VQA on VQAv2 (61.9 vs 56.3), while in contrast requiring no end-to-end training! 
  * [Model Release] Oct 2022, released implementation of **PNP-VQA** (**EMNLP Findings 2022**, _"Plug-and-Play VQA: Zero-shot VQA by Conjoining Large Pretrained Models with Zero Training"_, by Anthony T.M.H. et al), <br> 
  [Paper](https://arxiv.org/abs/2210.08773), [Project Page](https://github.com/salesforce/LAVIS/tree/main/projects/pnp-vqa), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/projects/pnp-vqa/pnp_vqa.ipynb))
  >  A modular zero-shot VQA framework that requires no PLMs training, achieving SoTA zero-shot VQA performance. 
    
## Table of Contents
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
    - [Model Zoo](#model-zoo)
    - [Image Captioning](#image-captioning)
    - [Visual question answering (VQA)](#visual-question-answering-vqa)
    - [Unified Feature Extraction Interface](#unified-feature-extraction-interface)
    - [Load Datasets](#load-datasets)
  - [Jupyter Notebook Examples](#jupyter-notebook-examples)
  - [Resources and Tools](#resources-and-tools)
  - [Documentations](#documentations)
  - [Ethical and Responsible Use](#ethical-and-responsible-use)
  - [Technical Report and Citing LAVIS](#technical-report-and-citing-lavis)
  - [License](#license)

## Introduction
LAVIS is a Python deep learning library for LAnguage-and-VISion intelligence research and applications. This library aims to provide engineers and researchers with a one-stop solution to rapidly develop models for their specific multimodal scenarios, and benchmark them across standard and customized datasets.
It features a unified interface design to access
- **10+** tasks
(retrieval, captioning, visual question answering, multimodal classification etc.);
- **20+** datasets (COCO, Flickr, Nocaps, Conceptual
Commons, SBU, etc.);
- **30+** pretrained weights of state-of-the-art foundation language-vision models and their task-specific adaptations, including [ALBEF](https://arxiv.org/pdf/2107.07651.pdf),
[BLIP](https://arxiv.org/pdf/2201.12086.pdf), [ALPRO](https://arxiv.org/pdf/2112.09583.pdf), [CLIP](https://arxiv.org/pdf/2103.00020.pdf).
<p align="center">
    <br>
    <img src="assets/demo-6.png"/>
    <br>
<p>

Key features of LAVIS include:

- **Unified and Modular Interface**: facilitating to easily leverage and repurpose existing modules (datasets, models, preprocessors), also to add new modules.

- **Easy Off-the-shelf Inference and Feature Extraction**: readily available pre-trained models let you take advantage of state-of-the-art multimodal understanding and generation capabilities on your own data.

- **Reproducible Model Zoo and Training Recipes**: easily replicate and extend state-of-the-art models on existing and new tasks.

- **Dataset Zoo and Automatic Downloading Tools**: it can be a hassle to prepare the many language-vision datasets. LAVIS provides automatic downloading scripts to help prepare a large variety of datasets and their annotations.


The following table shows the supported tasks, datasets and models in our library. This is a continuing effort and we are working on further growing the list.

|                  Tasks                   |     Supported Models     |             Supported Datasets             |
| :--------------------------------------: | :----------------------: | :----------------------------------------: |
|         Image-text Pre-training          |       ALBEF, BLIP        | COCO, VisualGenome, SBU ConceptualCaptions |
|           Image-text Retrieval           |    ALBEF, BLIP, CLIP     |              COCO, Flickr30k               |
|           Text-image Retrieval           |    ALBEF, BLIP, CLIP     |              COCO, Flickr30k               |
|        Visual Question Answering         |       ALBEF, BLIP        |           VQAv2, OKVQA, A-OKVQA            |
|             Image Captioning             |           BLIP           |                COCO, NoCaps                |
|           Image Classification           |           CLIP           |                  ImageNet                  |
| Natural Language Visual Reasoning (NLVR) |       ALBEF, BLIP        |                   NLVR2                    |
|          Visual Entailment (VE)          |          ALBEF           |                  SNLI-VE                   |
|             Visual Dialogue              |           BLIP           |                  VisDial                   |
|           Video-text Retrieval           |       BLIP, ALPRO        |               MSRVTT, DiDeMo               |
|           Text-video Retrieval           |       BLIP, ALPRO        |               MSRVTT, DiDeMo               |
|    Video Question Answering (VideoQA)    |       BLIP, ALPRO        |                MSRVTT, MSVD                |
|              Video Dialogue              |         VGD-GPT          |                    AVSD                    |
|      Multimodal Feature Extraction       | ALBEF, CLIP, BLIP, ALPRO |                 customized                 |
|         Text-to-image Generation         |      [COMING SOON]       |                                            |

## Installation

1. (Optional) Creating conda environment

```bash
conda create -n lavis python=3.8
conda activate lavis
```

2. install from [PyPI](https://pypi.org/project/salesforce-lavis/)
```bash
pip install salesforce-lavis
```
    
3. Or, for development, you may build from source

```bash
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```

## Getting Started
### Model Zoo
Model zoo summarizes supported models in LAVIS, to view:
```python
from lavis.models import model_zoo
print(model_zoo)
# ==================================================
# Architectures                  Types
# ==================================================
# albef_classification           ve
# albef_feature_extractor        base
# albef_nlvr                     nlvr
# albef_pretrain                 base
# albef_retrieval                coco, flickr
# albef_vqa                      vqav2
# alpro_qa                       msrvtt, msvd
# alpro_retrieval                msrvtt, didemo
# blip_caption                   base_coco, large_coco
# blip_classification            base
# blip_feature_extractor         base
# blip_nlvr                      nlvr
# blip_pretrain                  base
# blip_retrieval                 coco, flickr
# blip_vqa                       vqav2, okvqa, aokvqa
# clip_feature_extractor         ViT-B-32, ViT-B-16, ViT-L-14, ViT-L-14-336, RN50
# clip                           ViT-B-32, ViT-B-16, ViT-L-14, ViT-L-14-336, RN50
# gpt_dialogue                   base
```

Let’s see how to use models in LAVIS to perform inference on example data. We first load a sample image from local.

```python
import torch
from PIL import Image
# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load sample image
raw_image = Image.open("docs/_static/merlion.png").convert("RGB")
```

This example image shows [Merlion park](https://en.wikipedia.org/wiki/Merlion) ([source](https://theculturetrip.com/asia/singapore/articles/what-exactly-is-singapores-merlion-anyway/)), a landmark in Singapore.


### Image Captioning
In this example, we use the BLIP model to generate a caption for the image. To make inference even easier, we also associate each
pre-trained model with its preprocessors (transforms), accessed via ``load_model_and_preprocess()``.

```python
import torch
from lavis.models import load_model_and_preprocess
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# generate caption
model.generate({"image": image})
# ['a large fountain spewing water into the air']
```

### Visual question answering (VQA)
BLIP model is able to answer free-form questions about images in natural language.
To access the VQA model, simply replace the ``name`` and ``model_type`` arguments
passed to ``load_model_and_preprocess()``.

```python
from lavis.models import load_model_and_preprocess
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
# ask a random question.
question = "Which city is this photo taken?"
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
question = txt_processors["eval"](question)
model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
# ['singapore']
```

### Unified Feature Extraction Interface

LAVIS provides a unified interface to extract features from each architecture. 
To extract features, we load the feature extractor variants of each model.
The multimodal feature can be used for multimodal classification.
The low-dimensional unimodal features can be used to compute cross-modal similarity.


```python
from lavis.models import load_model_and_preprocess
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
caption = "a large fountain spewing water into the air"
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
text_input = txt_processors["eval"](caption)
sample = {"image": image, "text_input": [text_input]}

features_multimodal = model.extract_features(sample)
print(features_multimodal.multimodal_embeds.shape)
# torch.Size([1, 12, 768]), use features_multimodal[:,0,:] for multimodal classification tasks

features_image = model.extract_features(sample, mode="image")
features_text = model.extract_features(sample, mode="text")
print(features_image.image_embeds.shape)
# torch.Size([1, 197, 768])
print(features_text.text_embeds.shape)
# torch.Size([1, 12, 768])

# low-dimensional projected features
print(features_image.image_embeds_proj.shape)
# torch.Size([1, 197, 256])
print(features_text.text_embeds_proj.shape)
# torch.Size([1, 12, 256])
similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
print(similarity)
# tensor([[0.2622]])
```

### Load Datasets
LAVIS inherently supports a wide variety of common language-vision datasets by providing [automatic download tools](https://opensource.salesforce.com/LAVIS//latest/benchmark) to help download and organize these datasets. After downloading, to load the datasets, use the following code:

```python
from lavis.datasets.builders import dataset_zoo
dataset_names = dataset_zoo.get_names()
print(dataset_names)
# ['aok_vqa', 'coco_caption', 'coco_retrieval', 'coco_vqa', 'conceptual_caption_12m',
#  'conceptual_caption_3m', 'didemo_retrieval', 'flickr30k', 'imagenet', 'laion2B_multi',
#  'msrvtt_caption', 'msrvtt_qa', 'msrvtt_retrieval', 'msvd_caption', 'msvd_qa', 'nlvr',
#  'nocaps', 'ok_vqa', 'sbu_caption', 'snli_ve', 'vatex_caption', 'vg_caption', 'vg_vqa']
```
After downloading the images, we can use ``load_dataset()`` to obtain the dataset.
```python
from lavis.datasets.builders import load_dataset
coco_dataset = load_dataset("coco_caption")
print(coco_dataset.keys())
# dict_keys(['train', 'val', 'test'])
print(len(coco_dataset["train"]))
# 566747
print(coco_dataset["train"][0])
# {'image': <PIL.Image.Image image mode=RGB size=640x480>,
#  'text_input': 'A woman wearing a net on her head cutting a cake. ',
#  'image_id': 0}
```

If you already host a local copy of the dataset, you can pass in the ``vis_path`` argument to change the default location to load images.

```python
coco_dataset = load_dataset("coco_caption", vis_path=YOUR_LOCAL_PATH)
```

## Jupyter Notebook Examples
See [examples](https://github.com/salesforce/LAVIS/tree/main/examples) for more inference examples, e.g. captioning, feature extraction, VQA, GradCam, zeros-shot classification.

## Resources and Tools
- **Benchmarks**: see [Benchmark](https://opensource.salesforce.com/LAVIS//latest/benchmark) for instructions to evaluate and train supported models.
- **Dataset Download and Browsing**: see [Dataset Download](https://opensource.salesforce.com/LAVIS//latest/benchmark) for instructions and automatic tools on download common language-vision datasets.
- **GUI Demo**: to run the demo locally, run ```bash run_scripts/run_demo.sh``` and then follow the instruction on the prompts to view in browser. A web demo is coming soon.


## Documentations
For more details and advanced usages, please refer to
[documentation](https://opensource.salesforce.com/LAVIS//latest/index.html#).

## Ethical and Responsible Use
We note that models in LAVIS provide no guarantees on their multimodal abilities; incorrect or biased predictions may be observed. In particular, the datasets and pretrained models utilized in LAVIS may contain socioeconomic biases which could result in misclassification and other unwanted behaviors such as offensive or inappropriate speech. We strongly recommend that users review the pre-trained models and overall system in LAVIS before practical adoption. We plan to improve the library by investigating and mitigating these potential biases and
inappropriate behaviors in the future.


## Technical Report and Citing LAVIS
You can find more details in our [technical report](https://arxiv.org/abs/2209.09019).

If you're using LAVIS in your research or applications, please cite using this BibTeX:
```bibtex
@inproceedings{li-etal-2023-lavis,
    title = "{LAVIS}: A One-stop Library for Language-Vision Intelligence",
    author = "Li, Dongxu  and
      Li, Junnan  and
      Le, Hung  and
      Wang, Guangsen  and
      Savarese, Silvio  and
      Hoi, Steven C.H.",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-demo.3",
    pages = "31--41",
    abstract = "We introduce LAVIS, an open-source deep learning library for LAnguage-VISion research and applications. LAVIS aims to serve as a one-stop comprehensive library that brings recent advancements in the language-vision field accessible for researchers and practitioners, as well as fertilizing future research and development. It features a unified interface to easily access state-of-the-art image-language, video-language models and common datasets. LAVIS supports training, evaluation and benchmarking on a rich variety of tasks, including multimodal classification, retrieval, captioning, visual question answering, dialogue and pre-training. In the meantime, the library is also highly extensible and configurable, facilitating future development and customization. In this technical report, we describe design principles, key components and functionalities of the library, and also present benchmarking results across common language-vision tasks.",
}
}
```

## Contact us
If you have any questions, comments or suggestions, please do not hesitate to contact us at lavis@salesforce.com.

## License
[BSD 3-Clause License](LICENSE.txt)
