# 실행 스크립트 안내

현재 SQ-InstructBLIP 실험과 sub-QA 생성 entrypoint의 로컬 사용법이다. 일반 LAVIS 설치·benchmark 문서는 upstream 역사 자료로 남겨 둔다. 경로 기본값은 각 서버의 로컬 파일이므로 실행 전에 확인한다.

## 실행 환경과 위치

잠금 환경 설치, 서버별 기본 경로, tmux 실행 규칙과 provenance 기준은 [저장소 runbook](../docs/sq-instructblip-runbook.md)을 따른다. 아래 주 프로젝트 명령은 Python 3.10을 사용하고 Qwen 생성은 별도 Python `>=3.11` 환경을 사용한다. 자동 Slack 메시지는 보내지 않는다.

## Fraction 학습과 epoch-3 비교

Fraction sweep은 1/8, 1/4, 1/2, 전체에 대해 Questioner와 Reasoner를 학습해 총 8개 작업을 실행한다. seed는 42, GPU 기본값은 4다.

```bash
DRY_RUN=1 scripts/run_sq_instructblip_fraction_experiments.sh
scripts/run_sq_instructblip_fraction_experiments.sh
```

환경 override: `GPUS`(4), `BASE_CKPT`(`/model/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth`), `VICUNA_PATH`(`/model/lmsys/vicuna-7b-v1.1`), `UV_PYTHON_VERSION`(3.10), `DRY_RUN`(0). 스크립트는 `sample_fraction`, `sample_seed=42`, `eval_max_samples=null`, role/prompt, checkpoint/model path, world size와 output dir를 `--options`로 지정한다. Runner가 relative output dir 앞에 `lavis/`를 붙이므로 출력은 `lavis/ywjang_output_sq_fraction/<role>_<fraction>/...`이다.

Epoch-3 fraction 스크립트는 네 fraction을 학습하고 각 checkpoint 2를 10% validation subset에서 평가한다. checkpoint 2가 있으면 해당 role/fraction 학습을 건너뛴다. 이 스크립트의 평가는 QAR fraction 결과를 만든다. naive InstructBLIP 기준선과 순서대로 비교하려면 별도 `scripts/run_epoch3_ordered_eval.py`를 사용한다.

```bash
DRY_RUN=1 scripts/run_sq_instructblip_epoch3_fraction_comparison.sh
scripts/run_sq_instructblip_epoch3_fraction_comparison.sh
```

환경 override: `CUDA_VISIBLE_DEVICES`(기본 `0,1`), `GPUS`(2), `OUT_ROOT`(`ywjang_output_sq_epoch3_fraction`), `BASE_CKPT`(`/model/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth`), `ANSWERER_CKPT`(기본 BASE_CKPT), `VICUNA_PATH`(`/model/lmsys/vicuna-7b-v1.1`), `ANN_TRAIN`(`/home/ywjang/data/VQA-Introspect/VQAIntrospect_trainv1.0.json`), `ANN_VAL`(`/home/ywjang/data/VQA-Introspect/VQAIntrospect_valv1.0.json`), `IMAGE_ROOT`(`/data/coco/images`), `EVAL_SAMPLE_FRACTION`(0.1), `EVAL_SAMPLE_SEED`(42), `BATCH_SIZE_TRAIN`(1), `BATCH_SIZE_EVAL`(1), `ACCUM_GRAD_ITERS`(128), `EVAL_DEVICES`(`cuda:0 cuda:1`), `EVAL_DEVICE`(첫 eval device), `UV_PYTHON_VERSION`(3.10), `DRY_RUN`, `SKIP_TRAIN`, `SKIP_EVAL`(마지막 3개 기본 0). Relative `OUT_ROOT`는 `lavis/` 아래이며 요약은 `lavis/<OUT_ROOT>/summary_epoch3_ckpt2.{json,csv}`에 저장된다.

`DRY_RUN=1`은 명령만 출력한다. Epoch-3 스크립트는 `SKIP_TRAIN=1` 또는 `SKIP_EVAL=1`로 단계를 생략할 수 있다. Naive 기준선과 fraction 모델의 순차 평가 entrypoint는 다음과 같다. 기본 versions는 small/medium/large이며 naive 실행 후 각 QAR checkpoint 2를 평가한다.

```bash
uv run --python 3.10 python scripts/run_epoch3_ordered_eval.py --dry-run
```

이 entrypoint의 옵션은 `--out-root`, `--ordered-subdir`, `--cfg-path`, `--base-checkpoint`, `--answerer-checkpoint`, `--llm-model`, `--ann-val`, `--image-root`, `--eval-sample-fraction`, `--eval-sample-seed`, `--device`, `--devices`, `--max-len`, `--min-len`, `--num-beams`, `--python-executable`, `--versions`, `--dry-run`이다. 직접 QAR 평가 CLI는 `--cfg-path`, `--questioner-dir`, `--reasoner-dir`, `--answerer-checkpoint`, `--llm-model`, `--ann-val`, `--image-root`, `--output-dir`, `--eval-sample-fraction`, `--eval-sample-seed`, `--device`, `--devices`, `--num-shards`, `--shard-rank`, `--checkpoint-start`, `--checkpoint-end`, `--summary-prefix`, `--naive-results`, `--max-len`, `--min-len`, `--num-beams`를 받는다. Naive evaluator는 `--model-checkpoint`를 받는 대신 Questioner/Reasoner dirs와 naive-results를 받지 않는다.

Fraction 스크립트는 annotation/image 경로를 덮어쓰지 않는다. 기본 위치는 `lavis/configs/default.yaml`의 `cache_root: /home/ywjang/data`와 `lavis/configs/datasets/vqa_introspect/defaults.yaml`의 VQA-Introspect JSON 및 `coco/images/` 경로다. 경로 변경은 config `build_info` 편집 또는 epoch-3의 `ANN_TRAIN`, `ANN_VAL`, `IMAGE_ROOT`로 반영한다. 학습 fraction은 `vqa_introspect_qar_caption`에 적용되며 생성 CLI가 지원하는 dataset은 도구별 선택지로 한정된다.

## Questioner·baseline·C2R 생성 및 export

아래 flag 정보는 runtime을 import하지 않고 각 스크립트의 argparse source를 읽어 정리했다.

- `generate_questioner_subquestions.py`: VQA-Introspect/A-OKVQA validation Questioner 생성. `--cfg-path`, `--questioner-checkpoint`, `--llm-model`, `--datasets`, `--prompt-path`, annotation, `--coco-root`, dataset별 output dir, `--num-sub-questions`, `--sample-stride`, `--limit`, batch/worker 및 길이·beam 옵션. 기본 dataset 둘 다 실행.
- `generate_baseline_single_subquestion.py`: 같은 validation dataset의 prompt 기준선. `--baseline-prompt-path`, `--datasets`, config/checkpoint/model, annotation/image/output 경로, `--sample-stride`, `--limit`, batch/worker 및 생성 옵션.
- `generate_c2r_baseline_single_subquestion.py`: C2R의 `MMLU`, `MMMU`, `MMLU-Pro`, `StrategyQA`, `EgoSchema`만 지원. `--c2r-root`, `--source-model-name`, `--datasets`, config/checkpoint/model, prompt, `--limit`, batch/worker 및 생성 옵션, `--tmp-dir`, `--dummy-image-path`. 경로 형식은 `subqa/c2r_baseline_single_subq.py`에 정의돼 있다.
- `generate_questioner_subquestions_qwen.py`: Qwen2.5-VL로 같은 C2R 다섯 dataset 생성. `.venv-qwen/bin/python` 사용. `--model-name`(Hub ID 또는 로컬 경로), `--dataset` 또는 `--datasets`, `--c2r-root`, `--qids-root`, `--output-root`, prompt, limit, batch/worker, dtype, attention backend, pixel range, 생성 옵션. 데이터 경로 탐색은 `C2R_DATASET_SPECS`에 정의돼 있다.
- `export_thesis_sub_qas.py`: Questioner 결과와 Answerer로 thesis 형식 sub-QA 생성. `--vqa-introspect-source`, `--aok-vqa-source`, `--datasets`, config/prompt, `--answerer-checkpoint`, `--llm-model`, `--output-root`, batch/worker. checkpoint 미지정 시 예전 절대경로 후보를 찾으므로 명시한다.
- `subqa/vqa_introspect_export.py`: compact sub-QA를 judge-ready JSON으로 변환. 필수 `--generated-subqa-path`, `--output-path`; 선택 `--ann-path`, `--vis-root`, `--split`, `--sample-stride`.

CLI source 정적 확인 예:

```bash
rg -n 'add_argument|DEFAULT_|choices=' generate_questioner_subquestions.py generate_questioner_subquestions_qwen.py generate_baseline_single_subquestion.py generate_c2r_baseline_single_subquestion.py export_thesis_sub_qas.py
```

VQA-Introspect/A-OKVQA 도구는 annotation·COCO image root를 바꿀 수 있다. Qwen/C2R는 C2R root, Qwen 모델, qids root를 지정한다. 제공되지 않는 annotation mapping은 지원하지 않는다. 생성 JSON은 입력 annotation·prompt·모델·checkpoint 정보와 함께 보관한다.

```bash
uv run --python 3.10 python generate_questioner_subquestions.py --datasets vqa_introspect --questioner-checkpoint /path/to/questioner.pth --llm-model /path/to/vicuna --vqa-introspect-ann /data/VQAIntrospect_valv1.0.json --coco-root /data/coco/images
.venv-qwen/bin/python generate_questioner_subquestions_qwen.py --dataset MMMU --model-name /path/to/Qwen2.5-VL-7B-Instruct --c2r-root /data/C2R --qids-root /path/to/qids --output-root /data/generated
```

## 로컬 검증

주 회귀 테스트는 upstream model integration suite를 제외해 실행한다. 이 suite는 대형 model weight 다운로드와 추론을 포함하므로 필요한 모델 파일을 준비한 뒤 tmux `lavis:0 (run)`에서 따로 실행한다.

```bash
uv run --python 3.10 python -m pytest tests --ignore=tests/models -q
uv run --python 3.10 python -m pytest tests/test_vqa_introspect_fraction.py -q
```

## 결과, checkpoint, archive

source/config revision, dataset/model provenance와 archive 위치 기록은 [저장소 runbook](../docs/sq-instructblip-runbook.md)을 따른다. Fraction 출력은 `lavis/ywjang_output_sq_fraction/<role>_<fraction>/...`, epoch-3 비교 출력은 `lavis/ywjang_output_sq_epoch3_fraction/...`이다. Epoch-3 비교는 각 fraction에서 Questioner/Reasoner checkpoint 2를 평가한다. 이전 실험은 `lavis/ywjang_output_sq_epoch*`와 `archived/`에 있을 수 있다. checkpoint는 역할, fraction, run directory, index를 연결해 추적한다.

`scripts/archive_sq_instructblip_artifacts.sh`는 기본 dry-run이며 pre-fraction 출력과 임시 산출물을 timestamp가 붙은 `archived/` 아래로 이동한다. 목록을 먼저 확인하고 실제 이동은 `--execute`로 지정한다. 원시 결과·checkpoint는 source 커밋과 분리한다.
