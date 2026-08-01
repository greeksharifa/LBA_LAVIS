# Qwen2.5-VL MMMU Multi-stage 구현 계획

> 설계: `docs/superpowers/specs/2026-08-01-multi-stage-mmmu-design.md`
>
> 테스트 명령은 `/home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest discover -s tests -v`를 기준으로 한다.

## Task 1: Stage config, artifact namespace, 의존성 검증

**Files**

- Create: `util/artifacts.py`
- Modify: `util/path.py`
- Modify: `config/configs.py`
- Modify: `dataset/base_dataset.py`
- Create: `tests/test_artifacts.py`
- Create: `tests/test_dataset_dependencies.py`

### TDD 순서

1. `tests/test_artifacts.py`에 다음 실패 테스트를 작성한다.
   - dev/test 및 N/M/K가 다르면 run directory가 다르다.
   - `Config.for_stage(mode)`가 원본 `multi_stage` config를 변경하지 않는 독립 복제본을 반환한다.
   - manifest 핵심 config 또는 exact qid set 불일치는 명시적 오류가 된다.
   - `num_data=1`이면 전체 annotation이 아니라 sampling 이후 선택된 1개 qid로 manifest를 생성·검증한다.
2. `tests/test_dataset_dependencies.py`에 최소 `BaseDataset` subclass fixture를 만들고 다음 실패 테스트를 작성한다.
   - `base`와 `subq`는 기존 stage 파일 없이 sample을 반환한다.
   - `suba`는 subq 파일이 없으면 경로를 포함한 `FileNotFoundError`를 낸다.
   - `refined`는 subq/suba/base 파일과 qid/key를 모두 검증한다.
3. 위 두 테스트 파일을 실행해 production 코드의 현재 실패를 확인한다.
4. `Config.for_stage()`를 구현한다. OmegaConf 전체 tree를 deep copy하고 stage mode만 바꾼다.
5. run directory를 `{output_dir}/{dataset}/{model}/{split}/N={N}_M={M}_K={K}`로 통일한다. stage 파일명은 유지한다.
6. `util/artifacts.py`에 manifest 생성, atomic write, stage 완료 기록, config/qid 검증 함수를 구현한다. qid 검증은 `num_data` sampling 이후의 최종 annotation set에 대해서만 수행한다.
7. `BaseDataset` 의존성을 다음으로 제한하고 필요한 파일을 indexing 전에 검증한다.
   - subq/base: 없음
   - suba: subq
   - refined: subq, suba, base
8. 대상 테스트와 전체 테스트를 실행하고 커밋한다.

## Task 2: MMMU open-ended 정규화 및 공식 호환 채점

**Files**

- Create: `dataset/mmmu_eval.py`
- Modify: `dataset/MMMU.py`
- Modify: `dataset/base_dataset.py`
- Modify: `prompt/prompts.py`
- Create: `tests/test_mmmu_open.py`

### TDD 순서

1. `tests/test_mmmu_open.py`에 다음 실패 테스트를 작성한다.
   - source label `open`이 `open_ended`로 변환된다.
   - open 문항은 빈 option-letter prompt 대신 짧은 답 prompt를 받는다.
   - `1,234.50`, `1234.5`, 결론 문장 속 숫자가 같은 답으로 채점된다.
   - 소문자/공백 정규화 후 문자열 답 포함 관계가 채점된다.
   - multiple-choice는 option letter exact match를 유지한다.
   - 실제 dev/validation annotation에서 canonical open 개수가 각각 9/53이다.
2. 테스트를 실행해 현재 open label/prompt/scorer 실패를 확인한다.
3. `dataset/mmmu_eval.py`에 숫자 추출·정규화, open response parsing, open/multiple-choice 평가 함수를 구현한다.
4. MMMU adapter가 annotation load 시 question type을 canonicalize하도록 수정한다.
5. MMMU가 dataset-specific scorer를 사용하고 공통 base scorer가 답을 소수점에서 잘라내지 않게 경계를 정리한다.
6. 대상 테스트와 전체 테스트를 실행하고 커밋한다.

## Task 3: 안정적인 vLLM 설정과 single-process multi-stage runner

**Files**

- Create: `model/vllm_config.py`
- Modify: `model/models.py`
- Create: `pipeline.py`
- Modify: `main.py`
- Modify: `config/default.yaml`
- Modify: `util/logger.py`
- Create: `tests/test_vllm_config.py`
- Create: `tests/test_pipeline.py`
- Create: `tests/test_logger.py`

### TDD 순서

1. `tests/test_vllm_config.py`에 다음 실패 테스트를 작성한다.
   - 기본 환경은 기존 사용자 값을 덮어쓰지 않으면서 `VLLM_USE_V1=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`을 설정한다.
   - YAML의 max length/sequences, TP, GPU memory, swap, eager가 engine kwargs에 전달된다.
   - visible device count와 TP가 다르면 모델 생성 전에 실패한다.
2. `tests/test_pipeline.py`에 fake model/dataset/generator를 사용해 다음 실패 테스트를 작성한다.
   - stage 순서가 `subq, suba, base, refined`다.
   - model factory는 한 번만 호출된다.
   - 각 stage는 복제된 stage config, stage별 path와 manifest 완료 기록을 사용한다.
   - refined stage가 `refined_samples.json`을 만들고 각 record에 `qid`, `split`, `main_q`, `gt_ans`, `question_type`, `base_answer`, `conf_base`, `refined_answer_list`, `conf_refined`를 포함한다.
   - 단일 mode 실행도 같은 `run_stage` 경계를 사용한다.
3. `tests/test_logger.py`에 INFO 메시지가 console/file handler를 통과하고 handler 중복이 없다는 실패 테스트를 작성한다. `RANK=1` fixture에서는 routine INFO handler가 설치되지 않거나 억제되고 `RANK=0`/unset에서만 한 번 기록되는지도 검증한다.
4. 테스트를 실행해 현재 실패를 확인한다.
5. `model/vllm_config.py`를 vLLM import보다 먼저 호출하고 engine kwargs 생성 및 TP preflight를 분리한다.
6. `config/default.yaml`에 `max_model_len: 16384`, `enforce_eager: true`를 명시한다.
7. `pipeline.py`에 prompt 구성, generate, merge, JSON/manifest 저장을 담당하는 `run_stage`와 한 모델을 공유하는 `run_multi_stage`를 구현한다. refined stage는 Task 4가 독립적으로 읽을 수 있는 위 schema의 `refined_samples.json`도 저장한다.
8. `main.py`를 설정/로깅/모델 생명주기 orchestration으로 단순화하되 기존 네 single-stage CLI를 유지한다.
9. logger 자체 level을 설정하고 `RANK`/`LOCAL_RANK` 기준 main process에서만 routine application logs가 한 번 나오게 한다.
10. 대상 테스트와 전체 테스트를 실행하고 커밋한다.

## Task 4: 누수 없는 dev-to-validation C2R 평가

**Files**

- Create: `evaluation/__init__.py`
- Create: `evaluation/c2r.py`
- Create: `scripts/evaluate_c2r.py`
- Create: `tests/test_c2r_evaluation.py`

### TDD 순서

1. `tests/test_c2r_evaluation.py`에 다음 실패 테스트를 작성한다.
   - token-min-prob와 seq-ppl 후보 선택 방향이 각각 올바르다.
   - gate는 `base>=tau1`이면 base, 아니면 `refined>=base+tau2`일 때만 refined를 선택한다.
   - `seq_ppl` base/refined가 모두 `min(1, 1/max(ppl, 1e-12))`로 `[0,1]` confidence에 변환된 뒤 같은 gate에 들어간다. 0 또는 극소값도 epsilon으로 안전하게 처리한다.
   - 고정 11x21 grid와 동률 규칙(최소 switch, 낮은 tau1, 높은 tau2)이 결정적이다.
   - dev에서 선택한 threshold를 validation에 그대로 적용하며 validation GT로 재탐색하지 않는다.
   - paired bootstrap seed 42/10,000회가 재현 가능하다.
   - split/run manifest 또는 qid 불일치는 평가 전에 실패한다.
2. 테스트를 실행해 모듈 부재로 실패함을 확인한다.
3. pure evaluation 함수와 JSON report serializer를 구현한다.
4. `scripts/evaluate_c2r.py`에 dev/validation run directory를 명시적으로 받는 CLI를 구현한다.
5. report에 dev/validation의 원본 split과 annotation path, 표본 수, `threshold_source`, base/raw-refined/gated accuracy, absolute/relative delta, tau1/tau2, switch count, paired 95% CI, bootstrap seed/count, qids, manifest provenance를 기록한다.
6. 대상 테스트와 전체 테스트를 실행하고 커밋한다.

## Task 5: 실행 문서와 CPU 통합 검증

**Files**

- Modify: `README.md`
- Modify: `requirements.txt`
- Create: `tests/test_readme_commands.py`

### TDD 및 검증 순서

1. README 명령에 금지된 top-level `CUDA_VISIBLE_DEVICES=` 형식이 없고, GPU 5–8 wrapper 및 matching TP가 들어가는 실패 테스트를 작성한다.
2. README에 다음을 문서화한다.
   - single-stage와 `runner.mode=multi_stage`
   - GPU6/TP1 smoke
   - GPU5,6,7,8/TP4 full dev와 validation
   - dev threshold를 validation에 고정 적용하는 평가 CLI
   - artifact namespace와 historical 결과 구분
3. 현재 검증 환경의 torch/vLLM 버전과 V0+spawn 프로파일을 requirements/README에 명시한다. 무관한 패키징 전면 정비는 하지 않는다.
4. 전체 unittest, `compileall`, `git diff --check`를 실행하고 커밋한다.

## Task 6: GPU smoke와 fresh 성능 실행

**Files/Artifacts**

- Generate ignored run artifacts under `output/MMMU/qwen2.5-vl-7b/...`
- Generate evaluation report under the validation run directory

### 검증 순서

1. `nvidia-smi -L`로 GPU ID를 확인한다.
2. 다음 승인 wrapper 형태로 물리 GPU 6, TP=1, `dataset.num_data=1`, `N=5, M=2, K=8` multi-stage smoke를 실행한다.

   `/home/ywjang/.codex/bin/run_gpu.sh 6 -- env HF_HOME=/home/ywjang/.cache/huggingface /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=multi_stage model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=1 dataset.dataset_name=MMMU dataset.split=val dataset.num_data=1 runner.N=5 runner.M=2 runner.K=8`
3. 네 stage 산출물, manifest, 동일 model process 로그, 최종 score를 확인한다.
4. 다음 wrapper 형태로 GPU 5,6,7,8, TP=4 MMMU dev 150(`dataset.split=val`) multi-stage를 실행한다.

   `/home/ywjang/.codex/bin/run_gpu.sh 5,6,7,8 -- env HF_HOME=/home/ywjang/.cache/huggingface /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options runner.mode=multi_stage model.model_name=qwen2.5-vl-7b model.tensor_parallel_size=4 dataset.dataset_name=MMMU dataset.split=val dataset.num_data=-1 runner.N=5 runner.M=2 runner.K=8`

5. 같은 wrapper/GPU/TP에서 `dataset.split=test`로 바꿔 MMMU validation 900 multi-stage를 실행한다.
6. dev/validation run directory를 평가 CLI에 넘겨 고정-threshold report를 만든다.
7. fresh validation의 base와 gated C2R 정확도, paired delta/CI를 historical artifact와 분리해 보고한다.
8. 전체 CPU 검증을 다시 실행하고 최종 diff와 git status를 검토한다.
