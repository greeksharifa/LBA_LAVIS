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
- 각 stage는 기존 파일명과 경로를 그대로 사용한다. 따라서 중간 결과를 점검하거나 기존 단일-stage 실행으로 재개할 수 있다.
- stage 전환 때 dataset을 다시 만들되 모델은 재사용한다. dataset은 직전 stage가 저장한 JSON을 명시적으로 읽는다.
- routine 로그는 main process에서만 기록한다. vLLM worker별 애플리케이션 로그를 새로 추가하지 않는다.

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
- 현재 검증 환경의 compile 문제를 피하도록 기본 설정에서 `enforce_eager: true`를 사용하되 CLI override를 허용한다.
- 모델 생성 전에 `VLLM_WORKER_MULTIPROC_METHOD=spawn`을 `setdefault`로 설정한다. 사용자가 이미 지정한 값을 덮어쓰지 않는다.
- GPU 선택은 코드/config가 아니라 승인된 wrapper의 GPU 인자로만 한다. config의 tensor parallel 크기는 wrapper가 노출한 GPU 수와 일치시킨다.

## 평가 원칙

성능 보고는 다음 세 값을 분리한다.

- Base accuracy
- Raw refined accuracy: 각 문항에서 confidence가 가장 큰 refined 후보를 항상 선택
- Gated C2R accuracy: base/refined 선택 임계값을 dev 150문항에서 한 번 정하고 validation 900문항에 고정 적용

동일 split에서 임계값을 탐색한 oracle/post-hoc 값은 진단값으로만 표시하며 성능 향상으로 보고하지 않는다. 결과 JSON에는 split, 표본 수, 임계값, threshold source를 기록한다.

## 테스트와 검증

GPU 없이 다음 회귀 테스트를 먼저 작성한다.

- base dataset이 subq/suba 파일 없이 생성되고 sample을 반환한다.
- suba/refined가 필요한 파일 누락과 qid 누락을 명확히 보고한다.
- multi-stage가 한 model instance를 재사용하고 stage 순서 및 산출물 경계를 지킨다.
- model config가 `EngineArgs`로 전달되고 spawn 기본값이 설정된다.
- dev에서 선택한 임계값이 validation에서는 재탐색되지 않는다.

그 뒤 허용 GPU 중 최소 개수로 Qwen2.5-VL/MMMU 1문항 multi-stage smoke를 실행한다. 전체 900문항 재실행은 smoke와 기존 artifact 검증이 끝난 뒤 필요성과 예상 시간을 보고하고 수행한다.

## 범위 밖

이번 변경은 MMMU/Qwen2.5 경로에 집중한다. 조사 중 발견된 MMLU adapter 시그니처, 깨진 docs CI, 패키징 전면 정비는 별도 작업으로 남긴다.
