# Qwen2.5-VL MMMU Multi-stage 실행 및 평가 설계

## 목표

Qwen2.5-VL-7B로 MMMU를 다음 두 방식으로 재현 가능하게 실행하고 비교한다.

1. `base`: sub-question 산출물 없이 독립 실행
2. `multi_stage`: 한 모델 프로세스에서 `subq -> suba -> base -> refined` 순차 실행

GPU는 실행 시 사용자가 허용한 물리 GPU 5, 6, 7, 8의 부분집합만 사용한다.

## 확인된 원인

- `BaseDataset`이 `base`를 포함한 거의 모든 mode에서 `subq/suba` 파일을 읽는다. 따라서 독립적이어야 할 base 추론이 선행 산출물 없이는 `NoneType` 접근으로 실패한다.
- YAML의 `tensor_parallel_size`, `gpu_memory_utilization`, `swap_space`가 `EngineArgs`에 전달되지 않아 선언한 GPU/메모리 구성이 적용되지 않는다.
- 현재 vLLM 0.8.2 환경은 CUDA 초기화 뒤 fork worker를 만들면 실패하며, V1의 torch.compile 경로도 이 GPU/torch 조합에서 실패한다.
- 기존 평가는 정답이 있는 평가 split에서 C2R 선택 임계값을 다시 탐색한다. 이는 독립 평가가 아니라 사후 최적화다.

## 실행 구조

`main.py`는 설정과 모델 생명주기를 관리하고, 새 실행 모듈은 한 stage를 수행하는 순수한 경계를 제공한다.

- 기존 `runner.mode={subq,suba,base,refined}`는 호환성을 유지한다.
- 새 `runner.mode=multi_stage`는 모델을 한 번만 생성한 뒤 네 stage를 순차 실행한다.
- 각 stage는 기존 파일명(`subq_outputs.json` 등)은 유지하되, 경로는 split과 run signature가 포함된 namespace로 격리한다. 따라서 중간 결과를 점검하거나 기존 단일-stage 실행으로 재개할 수 있다.
- stage 전환 때 dataset을 다시 만들되 모델은 재사용한다. dataset은 직전 stage가 저장한 JSON을 명시적으로 읽는다.
- top-level `multi_stage` config를 stage 코드에 직접 넘기지 않는다. 각 stage는 전체 config를 복제한 뒤 mode만 해당 stage로 바꾼 독립 config를 받으며, output locator도 이 stage config로 계산한다.
- routine 로그는 main process에서만 기록한다. vLLM worker별 애플리케이션 로그를 새로 추가하지 않는다.
- 애플리케이션 logger 자체 level을 설정해 stage, artifact provenance, 평가 요약 INFO 로그가 실제 handler에 한 번만 전달되게 한다.

## Artifact namespace와 provenance

모든 stage 산출물은 다음 run directory 아래에 둔다.

`{output_dir}/{dataset}/{model}/{split}/N={N}_M={M}_K={K}/`

`subq/suba`도 별도 전역 경로가 아니라 같은 run directory에 저장한다. 이전 경로에 있던 파일을 자동 재사용하지 않아 다른 split이나 설정의 결과가 섞이지 않게 한다.

각 run은 `run_manifest.json`에 dataset, annotation split, model name/id, N/M/K, confidence type, stage별 완료 상태와 qid 목록을 기록한다. 후속 stage와 평가는 현재 annotation의 exact qid set 및 핵심 config가 manifest와 일치하는지 확인한다. 불일치하면 재사용하지 않고 명시적으로 실패한다.

## 산출물 의존성

| stage | 필요한 기존 산출물 | 새 산출물 |
|---|---|---|
| `subq` | 없음 | `subq_outputs.json` |
| `suba` | `subq_outputs.json` | `suba_outputs.json` |
| `base` | 없음 | `base_outputs.json` |
| `refined` | `subq`, `suba`, `base` outputs | `refined_outputs.json`, 평가용 samples |

필수 산출물이 없거나 qid가 빠졌으면 해당 경로와 qid를 포함한 명시적 예외를 낸다. 조용히 `None`으로 진행하지 않는다.

## vLLM 실행 안정성

- 지원하는 model config 키를 `EngineArgs`에 명시적으로 전달한다: `max_model_len`, `max_num_seqs`, `tensor_parallel_size`, `gpu_memory_utilization`, `swap_space`, `enforce_eager`.
- 현재 검증된 안정 조합인 `VLLM_USE_V1=0`과 `VLLM_WORKER_MULTIPROC_METHOD=spawn`을 vLLM import 전에 `setdefault`로 설정한다. 사용자가 이미 지정한 값을 덮어쓰지 않는다.
- compile/CUDA graph 변수를 줄이도록 기본 설정에서 `enforce_eager: true`를 사용하되 CLI override를 허용한다. V1은 `VLLM_USE_V1=1`로 명시적으로 선택하고 별도 GPU smoke를 통과해야 지원 대상으로 간주한다.
- GPU 선택은 코드/config가 아니라 승인된 wrapper의 GPU 인자로만 한다. config의 tensor parallel 크기는 wrapper가 노출한 GPU 수와 일치시킨다.
- TP 크기는 자동 변경하지 않는다. 실행자가 CLI에서 지정하며, 모델 생성 전에 `tensor_parallel_size == torch.cuda.device_count()`인지 검사해 불일치하면 실패한다.

## 평가 원칙

성능 보고는 다음 세 값을 분리한다.

- Base accuracy
- Raw refined accuracy: 각 문항에서 confidence가 가장 큰 refined 후보를 항상 선택
- Gated C2R accuracy: base/refined 선택 임계값을 dev 150문항에서 한 번 정하고 validation 900문항에 고정 적용

Gated C2R 선택식은 다음과 같다. 변환된 base confidence가 `tau1` 이상이면 base를 유지한다. 그렇지 않고 best refined confidence가 `base confidence + tau2` 이상이면 refined로 바꾸며, 나머지는 base를 유지한다.

dev 탐색 grid는 `tau1 = 0.0, 0.1, ..., 1.0`, `tau2 = -1.0, -0.9, ..., 1.0`으로 고정한다. 정확도 동률이면 refined로 바뀐 dev 문항 수가 가장 적은 조합, 다시 동률이면 낮은 `tau1`, 높은 `tau2` 순으로 선택한다. validation에는 선택된 한 조합만 적용한다.

동일 split에서 임계값을 탐색한 oracle/post-hoc 값은 진단값으로만 표시하며 성능 향상으로 보고하지 않는다. 결과 JSON에는 split, 표본 수, 임계값, threshold source를 기록한다.

confidence 비교는 `[0, 1]` 범위의 공통 higher-is-better score로 변환한다. `token_min_prob`는 그대로, `seq_ppl`은 `1 / max(seq_ppl, epsilon)`을 사용한다. 후보 선택과 gate는 이 변환을 공유하며 두 방향을 단위 테스트한다.

현재 저장소의 1문항 smoke는 정확도를 주장할 근거가 아니다. 기존 `/home/ywjang/C2R`의 900문항 수치는 provenance가 다른 historical result로만 보고한다. 이번 구현의 성능 결론은 historical artifact와 비교 가능한 `N=5, M=2, K=8, confidence=token_min_prob`로 fresh dev 150문항에서 threshold를 선택하고 fresh validation 900문항에 고정 적용한 뒤에만 확정한다. 동일 900 qid의 paired base/C2R delta에 대해 seed 42, 10,000회 paired bootstrap의 percentile 95% CI를 함께 기록한다.

## MMMU 문항 및 채점 호환성

MMMU annotation의 `question_type=open`은 adapter에서 즉시 내부 canonical 값 `open_ended`로 변환한다. 이 문항은 선택지 letter가 아니라 짧은 단어·구·수치 답을 요구하는 prompt를 사용한다.

open-ended 채점은 MMMU 공식 의미를 따른다. 응답의 결론 구문과 수치를 추출하고, 문자열은 소문자·공백 정규화 후 포함 관계로, 숫자는 쉼표 제거 및 소수 둘째 자리 반올림 후 비교한다. multiple-choice는 정규화된 option letter의 exact match를 유지한다. dev의 open 9문항과 validation의 open 53문항이 각각 올바른 prompt와 scorer를 쓰는지 테스트한다.

## 테스트와 검증

GPU 없이 다음 회귀 테스트를 먼저 작성한다.

- base dataset이 subq/suba 파일 없이 생성되고 sample을 반환한다.
- suba/refined가 필요한 파일 누락과 qid 누락을 명확히 보고한다.
- multi-stage가 한 model instance를 재사용하고 stage 순서 및 산출물 경계를 지킨다.
- model config가 `EngineArgs`로 전달되고 spawn 기본값이 설정된다.
- dev와 validation artifact의 split, qid, run signature가 분리되고 검증된다.
- dev에서 선택한 임계값이 validation에서는 재탐색되지 않는다.
- confidence 종류별 후보 선택 방향과 paired bootstrap CI가 검증된다.
- MMMU `open` label 정규화, open prompt, 문자열·수치 공식 호환 채점이 검증된다.
- logger INFO 메시지가 중복 없이 기록된다.

그 뒤 Qwen2.5-VL/MMMU 1문항 multi-stage smoke를 다음 형태로 실행한다.

`/home/ywjang/.codex/bin/run_gpu.sh 6 -- env ... python main.py --options ... model.tensor_parallel_size=1`

GPU 6은 허용 범위 5–8 안이며, wrapper가 노출한 1개 GPU와 TP=1을 사전 검증한다. smoke 성공 후 fresh dev 150과 validation 900은 GPU 5,6,7,8 및 TP=4로 실행한다. 다른 물리 GPU는 사용하지 않는다.

## 범위 밖

이번 변경은 MMMU/Qwen2.5 경로에 집중한다. 조사 중 발견된 MMLU adapter 시그니처, 깨진 docs CI, 패키징 전면 정비는 별도 작업으로 남긴다.
