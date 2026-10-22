# SQ-InstructBLIP Runbook

이 문서는 서버 이동 후 연구 실행을 재개하기 위한 환경과 경로의 짧은 기준이다. 상세 CLI와 지원 override는 [scripts/README.md](../scripts/README.md)를, 저장소 구조와 개발 규칙은 [AGENTS.md](../AGENTS.md)를 본다.

## 환경

주 LAVIS는 `.python-version`의 Python 3.10을 사용한다. `pyproject.toml`과 `uv.lock`의 범위는 `>=3.8,<3.11`이며 lockfile은 저장소에 포함한다. 개발·테스트 환경은 다음과 같이 준비한다.

```bash
uv sync --locked --python 3.10 --group dev
uv run --offline --python 3.10 python -m pytest tests --ignore=tests/models -q
```

Qwen2.5-VL 생성은 별도 Python `>=3.11` 환경을 권장한다. 로컬 `.venv-qwen`은 Python 3.11.4로 운영 중이다. 준비 명령은 다음과 같다.

```bash
uv venv --python 3.11 .venv-qwen
uv pip install --python .venv-qwen/bin/python -r requirements-qwen-vl.txt
```

`requirements-qwen-vl.txt`의 Transformers Git 설치는 revision을 고정하지 않으므로 설치한 commit/version 및 모델 revision을 기록한다. `tests/models/`의 upstream integration tests는 대형 weight 다운로드와 모델 추론이 필요하므로, weight를 준비한 뒤 tmux `lavis:0 (run)`에서 별도로 실행한다.

## 데이터와 모델 경로

기본 checkpoint는 `/model/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth`, Vicuna는 `/model/lmsys/vicuna-7b-v1.1`이다. cache root는 `lavis/configs/default.yaml`; VQA-Introspect annotation과 COCO image 경로는 `lavis/configs/datasets/vqa_introspect/defaults.yaml`을 따른다. 서버별 실제 경로를 확인하고 config 또는 스크립트의 지원 override로 지정한다.

## 실행과 산출물

학습·생성·평가 명령은 tmux `lavis`, window `0` (`run`)에서 실행한다. 스윕은 먼저 `DRY_RUN=1`로 확인하고, 지속 로그는 `/home/ywjang/.codex` 아래에 둔다. 기본 fraction 산출물은 `lavis/ywjang_output_sq_fraction/`, epoch-3 비교는 `lavis/ywjang_output_sq_epoch3_fraction/` 아래에 생성된다. `.gitignore`는 이 경로와 저장소 루트 `ywjang_output*/`, `archived/`, 가중치, `.venv*`를 제외한다.

실행 기록에는 source HEAD와 dirty 상태, config 사본/hash, CLI 및 환경 override, dataset/split/표본 수/seed, 모델 revision, checkpoint 식별자와 결과 경로를 남긴다. 원시 결과와 checkpoint는 source Git에 넣지 않고 별도 보관 위치를 기록한다. 원격 archive URL/object ID는 실제 보관이 확인된 뒤 기록한다.
