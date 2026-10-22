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

## SQ-InstructBLIP / VQA-Introspect

이 브랜치는 InstructBLIP-vicuna7b 기반 Questioner·Answerer·Reasoner 연구 코드다. Questioner와 Reasoner를 학습하고, 생성한 sub-QA로 최종 답변을 평가한다. `QAR`은 기존 registry와 파일명의 호환 명칭이다.

- [SQ-InstructBLIP runbook](docs/sq-instructblip-runbook.md): 환경·경로·실험 안전 규칙의 짧은 진입점.
- [스크립트 상세 안내](scripts/README.md): fraction 학습, epoch3 비교, QAR 평가, Questioner·Qwen·baseline 생성, 논문용 export와 결과 보관.
- [작업 지침](AGENTS.md): 환경, 검증, 실험 실행 위치와 코드 규칙.
- 학습 설정: [Questioner](configs/sq_instructblip/questioner.yaml), [Reasoner](configs/sq_instructblip/reasoner.yaml).
- 학습/역할별 평가 데이터: `lavis/datasets/datasets/vqa_introspect_datasets.py`, registry `vqa_introspect_qar_caption`, task `vqa_introspect_captioning`.
- 통합 QAR 평가 데이터: `lavis/datasets/datasets/vqa_introspect_test_datasets.py`, registry `vqa_introspect_test`. `test_qar.py`와 `run_test_qar_entrypoint.py`는 `instructBLIP_FT_vicuna7b_vqaintrospect_qar.yaml`을 사용한다.
- 역할별 prompt는 `prompts.json`, baseline prompt는 `baseline_prompts_single_subq.json`, 변환·검증 helper는 `subqa/`에 있다.

### 환경과 검증

프로젝트는 `.python-version`의 Python 3.10을 사용하며 `pyproject.toml`과 `uv.lock`은 Python `>=3.8,<3.11` 범위다. `uv.lock`에는 개발용 pytest도 포함된다. 기본 LAVIS 의존성은 Transformers 4.28 계열과 CUDA 12.1 PyTorch index를 사용한다. Qwen 생성은 별도 Python 3.11 환경에서 실행한다. 상세 절차는 [runbook](docs/sq-instructblip-runbook.md)과 [스크립트 안내](scripts/README.md)를 따른다.

```bash
uv sync --locked --python 3.10 --group dev
uv run --python 3.10 python -m pytest tests --ignore=tests/models -q
DRY_RUN=1 scripts/run_sq_instructblip_fraction_experiments.sh
DRY_RUN=1 scripts/run_sq_instructblip_epoch3_fraction_comparison.sh
```

`tests/models/`는 대형 가중치를 다운로드하거나 모델 추론을 수행하는 upstream integration suite다. 해당 가중치를 준비한 뒤 tmux `lavis:0 (run)`에서 별도로 실행한다.

학습·생성·평가 실험은 tmux 세션 `lavis`, 창 `0`(`run`)에서 실행한다. 순차 실행 큐도 같은 창에 두고 지속 로그는 `${HOME}/.codex/`에 남긴다.

### 모델·데이터·출력

기본 모델 위치는 `/model/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth`와 `/model/lmsys/vicuna-7b-v1.1`이다. fraction 스크립트는 `BASE_CKPT`, `VICUNA_PATH`, `GPUS`, `CUDA_VISIBLE_DEVICES`로 조정한다. 직접 실행은 `--options model.finetuned=... model.llm_model=...`을 사용한다.

`lavis/configs/default.yaml`의 `env.cache_root` 기본값은 `/home/ywjang/data`이며, VQA annotation과 COCO images 기본 경로는 `lavis/configs/datasets/vqa_introspect/defaults.yaml`에 있다. 다른 서버에서는 annotation/image storage를 절대경로로 override한다. 간단한 fraction 스크립트는 데이터 경로 환경변수를 받지 않으므로 직접 `train.py --options`를 사용하거나 설정 파일을 준비한다. epoch3 스크립트는 `ANN_TRAIN`, `ANN_VAL`, `IMAGE_ROOT`를 받는다.

`RunnerBase`는 상대 `run.output_dir`를 `lavis/` 아래에 해석한다. 따라서 기본 fraction 출력은 `lavis/ywjang_output_sq_fraction/<role>_<size>/<job_id>/`, epoch3 비교는 `lavis/ywjang_output_sq_epoch3_fraction/`이다. 절대 출력 경로는 그대로 사용한다. 원시 결과·로그·가중치는 Git에 넣지 않고 run ID, split/sample 수, 코드·설정 revision, archive 위치와 모델 해시를 연결해 별도 보존한다.

아래 일반 LAVIS 소개와 예제는 upstream 참고 자료다. 이 브랜치의 설치·실행은 위 문서와 잠금 환경을 기준으로 한다.

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
