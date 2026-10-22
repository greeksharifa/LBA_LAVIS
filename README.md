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

## InstructBLIP Fine-Tuning을 위한 수정사항

- https://opensource.salesforce.com/LAVIS/latest/tutorial.datasets.html 참고
- 참고: qar(Questioner-Answerer-Reasoner)은 이전 이름으로 추후 SQ_InstructBLIP로 변경 예정
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

## BIVI 사본의 실제 실행·보존 범위

이 checkout은 `captioning_base` 기준 `c134045`의 후속 QAR/SQ-InstructBLIP 실험이다.
`lavis/datasets/datasets/vqa_introspect_test_datasets.py`가 실제 test dataset 파일명이며,
위의 `vqa_introspect_datasets_test.py` 설명은 이 이름으로 읽는다.
`backups/`의 작은 Python/YAML/shell 파일은 당시 caption·VQA 구현을 보존한 source이다.

| entrypoint / 설정 | 역할 및 출력 |
| --- | --- |
| `train.py`, `train.sh`, `instructBLIP_FT_vicuna7b.yaml` | 역할별 fine-tuning, `ywjang_output_train/`, seed 42 |
| `test.py`, `test.sh`, `instructBLIP_FT_vicuna7b_test.yaml` | Reasoner checkpoint 검증, `ywjang_output_test/`, val split |
| `test_qar.py`, `test_qar.sh`, `instructBLIP_FT_vicuna7b_qar.yaml` | QAR 평가, `vqa_introspect_test`, seed 48, val split |
| `test_qar_answerer.py`, `test_qar_answerer.sh` | Answerer 실험 분기 |
| `test_vqa.sh`, `instructBLIP_FT_vicuna7b_vqa_test.yaml` | A-OKVQA 설정, seed 42, val split |

현재 `lavis/configs/default.yaml`의 data root는 `/data`, base InstructBLIP은
`/models/instruct_blip_vicuna7b_trimmed.pth`, Vicuna는 `/models/lmsys/vicuna-7b-v1.1`이다.
`prompts.json`이 역할 prompt를 지정한다. QAR YAML의 dummy checkpoint 값은 실제
`test_qar.py` 로딩 경로와 함께 확인해야 하며 그대로 독립 실행 가능한 설정이라고
보장하지 않는다. GQA defaults의 annotation URL은 A-OKVQA URL을 가리키는 기존
실험 변경이므로 일반 GQA 다운로드 recipe로 사용하면 안 된다.

설치는 기존 `requirements.txt` / `setup.py` 기준이다. 의존성 lock, 모델 revision,
최종 논문 행과 run/QID/표본 수·metric의 대응은 이 사본만으로 확정하지 못했다.
`lavis/ywjang_output*/`와 `ywjang_log.txt`는 산출물로 제외하며 삭제하지 않는다.
`port_value_file.txt`의 증가 값은 실행 counter여서 source commit에서 제외한다.
소스 syntax 및 shell syntax만 점검했고 학습·GPU·평가를 실행하지 않았다.
기존 debug 출력과 multi-worker 파일 logger는 이 보존 작업에서 변경하지 않았다.
chmod/CRLF만 다른 대량 파일과 바이너리는 보존 commit에서 제외했다.
