# 모델·벤치마크 독립 Hierarchical Sub-QA 구현 계획

> 설계: `docs/superpowers/specs/2026-08-01-hierarchical-subqa-design.md`
>
> 상태: 2026-08-01 구현·검증 완료
>
> 완료 증거: `docs/exec-plans/completed/2026-08-01-hierarchical-subqa.md`
>
> 구현은 당시 실행 계약의 slice 순서와 `execution-harness`를 따랐다. 각 task는 실패 테스트를 먼저 확인한 뒤 최소 구현으로 green을 만들고 전체 회귀를 확인했다.

## Task 0: 문서 계약 독립 검토

**Files**

- Review: `docs/superpowers/specs/2026-08-01-hierarchical-subqa-design.md`
- Review: `docs/superpowers/plans/2026-08-01-hierarchical-subqa-plan.md`
- Review: completed execution evidence archive

- [x] 설계 문서를 독립 reviewer에게 전달해 scope, protocol, artifact invariant, failure policy의 blocker가 없음을 확인했다.
- [x] 구현 계획/활성 실행 문서를 다른 독립 reviewer에게 전달해 blocker가 없음을 확인했다.
- [x] high-severity 명확화 항목을 설계와 closure 계약에 반영했다.
- [x] permanent doc가 active plan을 역참조하지 않고 active docs가 permanent docs를 참조하는지 확인했다.
- [x] 현재 사용자의 명시적 구현 요청을 승인 기록으로 남겼다. Review가 승인된 scope/policy를 변경하지 않았음을 확인했다.

## Task 1: 기본 설정과 hierarchy config 계약

**Files**

- Modify: `config/default.yaml`
- Modify: `config/configs.py`
- Modify: `README.md`
- Modify: `tests/test_artifacts.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_readme_commands.py`
- Create: `tests/test_hierarchical_subqa.py`

### TDD 순서

- [x] 기본 config가 `N=4, M=2, K=4`, depth 1, null branching, internal `M=2/K=3`, schema v2를 노출하는 실패 테스트를 작성한다.
- [x] depth 1 null branching이 `[N]`으로 canonicalize되고 legacy runner/path를 선택하는 실패 테스트를 작성한다.
- [x] depth>1 missing branching, depth/length mismatch, `branching[0] != N`, bool/0/negative branching을 거부하는 테스트를 작성한다.
- [x] root 제외 node 총합이 max nodes를 넘는 설정, 각 internal level에서 `suba_M > child_count`, `suba_K > C(child_count, suba_M)`인 설정을 거부하는 테스트를 작성한다.
- [x] invalid batch size, repair count, direct-conditioning type, schema version, unsupported confidence metric을 거부하는 테스트를 작성한다.
- [x] config validation이 model factory 호출 전에 실패한다는 pipeline 테스트를 작성한다.
- [x] 테스트를 실행해 현재 hierarchy config 부재와 기본값 차이로 실패함을 확인한다.
- [x] `config/configs.py` 또는 새 schema helper에 단일 canonical normalization/validation 경계를 구현하고 pipeline, path, manifest가 재사용하게 한다.
- [x] `config/default.yaml`과 README의 일반 실행 예시를 `4/2/4`로 바꾸고 `N=5` fixture/문구는 명시적 legacy-access test만 남긴다.
- [x] focused tests를 green으로 만들고 기존 flat path 회귀를 확인한다.

## Task 2: Model-independent generation 결과와 confidence 공통화

**Files**

- Modify: `model/models.py`
- Modify: `model/__init__.py`
- Create: `model/protocol.py`
- Modify: `prompt/postprocess.py`
- Modify: `evaluation/c2r.py`
- Create: `util/confidence.py`
- Modify: `tests/test_c2r_evaluation.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_hierarchical_subqa.py`

### TDD 순서

- [x] frozen `GenerationResult(text, confidence)`가 문자열 text와 metric mapping을 전달하는 테스트를 작성한다.
- [x] `GenerationResult`/registry import가 torch/vLLM을 eager import하지 않는 기존 CPU-only import contract를 유지하는 테스트를 작성한다.
- [x] fake non-vLLM model의 `generate_results()` 결과만으로 hierarchy generation 순서와 metadata 정렬이 유지되는 실패 테스트를 작성한다.
- [x] Qwen raw vLLM fixture가 text, sequence perplexity, token minimum probability를 `GenerationResult`로 변환하는 테스트를 작성한다.
- [x] zero token/logprob edge case와 output count mismatch를 명시적으로 처리하는 테스트를 작성한다.
- [x] `token_min_prob` 최대, `seq_ppl` 최소가 같은 higher-is-better normalization을 사용하고 동률에서 첫 후보를 선택하는 공통 utility 테스트를 작성한다.
- [x] evaluator가 기존 local 구현 대신 공통 confidence utility를 import해 동일 결과를 내는 회귀 테스트를 유지한다.
- [x] 테스트를 실행해 common result/utility 부재와 raw vLLM 결합 때문에 실패함을 확인한다.
- [x] `GenerationResult`를 lightweight `model/protocol.py`에 두고 `C2RFramework.generate_results()`, Qwen adapter 변환을 구현하며 기존 `generate()`는 유지한다. Fake model은 `C2RFramework` 상속을 요구하지 않는다.
- [x] confidence normalization과 first-maximum selection을 `util/confidence.py`로 이동하고 hierarchy/evaluator가 공유하게 한다.
- [x] legacy `format_vllm_outputs()`의 동작과 기존 evaluator denominator/threshold 테스트를 green으로 유지한다.

## Task 3: Qwen text/image/video template 경계

**Files**

- Modify: `model/models.py`
- Modify: `tests/test_vllm_config.py` 또는 `tests/test_hierarchical_subqa.py`

### TDD 순서

- [x] text modality와 `vision=None`이 plain prompt를 반환하고 vision token 및 multimodal fields를 만들지 않는 실패 테스트를 작성한다.
- [x] image list가 image placeholder, payload, ordered UUID를 유지하는 회귀 테스트를 작성한다.
- [x] video payload가 video placeholder, payload, ordered UUID를 유지하는 회귀 테스트를 작성한다.
- [x] 기존 adapter가 사용하는 ndarray 및 `(video, metadata)` wrapper 표현을 임의 정규화하지 않는 regression fixture를 작성한다.
- [x] unknown modality와 image/video의 invalid missing payload가 명확한 오류를 내는 테스트를 작성한다.
- [x] 테스트를 실행해 text의 unbound placeholder/`len(None)` 실패를 확인한다.
- [x] modality별 template branch를 구현하되 hierarchy orchestration에는 Qwen token 또는 class check를 넣지 않는다.
- [x] focused template tests와 model import CPU tests를 green으로 만든다.

## Task 4: Canonical tree schema와 deterministic selection

**Files**

- Create: `subqa/__init__.py`
- Create: `subqa/schema.py`
- Create: `subqa/selection.py`
- Modify: `tests/test_hierarchical_subqa.py`

### TDD 순서

- [x] depth 2 branching `[4,3]`이 stable IDs `0..3`, `0.0..3.2`와 preorder nodes를 만드는 실패 테스트를 작성한다.
- [x] depth 3 fixture가 같은 recursive builder/validator로 parent, depth, sibling, child order를 만족하는 테스트를 작성한다.
- [x] duplicate ID, missing parent, inconsistent depth/sibling, bad child order, extra depth, invalid status/failure pairing을 거부하는 테스트를 작성한다.
- [x] depth-1 projection이 ID 순서로 정확한 `subq_list`를 만들고 incomplete depth-1을 거부하는 테스트를 작성한다.
- [x] `N=3, M=2, K=3` evidence subset이 `(0,1), (0,2), (1,2)`이고 `K`가 작으면 prefix만 쓰는 테스트를 작성한다.
- [x] partial child list는 available sibling order를 보존하며 child 수가 `M` 미만이면 candidate subset을 만들지 않는 테스트를 작성한다.
- [x] raw confidence selection이 normalized first maximum이며 invalid candidates를 제외하는 테스트를 작성한다.
- [x] 실패 테스트를 확인한 뒤 pure schema/selection helpers를 구현한다.
- [x] focused tests를 green으로 만들고 implementation에 random sampling이 없음을 review한다.

## Task 5: Hierarchical prompt와 question parsing/repair

**Files**

- Create: `subqa/prompts.py`
- Create: `subqa/parsing.py`
- Modify: `tests/test_hierarchical_subqa.py`

### TDD 순서

- [x] root/descendant decomposition prompt가 각각 main question, target parent, ancestor path, modality, exact child count를 포함하는 테스트를 작성한다.
- [x] leaf direct prompt가 main question, ancestor path, target question, modality를 포함하는 테스트를 작성한다.
- [x] internal refined prompt가 target question과 ordered child-QA를 포함하고, flag가 true일 때만 direct draft를 포함하는 테스트를 작성한다.
- [x] multiple-choice와 open-ended sample 모두 hierarchy prompt에 `gt_ans`와 `candidate_list` 값이 나타나지 않는 테스트를 작성한다.
- [x] parser가 numbering/bullet을 제거하고 whitespace를 정규화하며 casefold+terminal punctuation 기준 중복을 첫 출현 순서로 제거하는 테스트를 작성한다.
- [x] 부족 질문은 repair prompt에 기존 valid 질문을 포함하고 repair limit을 지키며 generic padding을 하지 않는 테스트를 작성한다.
- [x] direct/candidate answer의 blank-only output을 invalid로 처리하는 테스트를 작성한다.
- [x] 실패 테스트를 확인한 뒤 hierarchy-only prompt/parser를 구현한다. 기존 base/refined prompt의 choices 동작은 바꾸지 않는다.
- [x] focused prompt/parser tests를 green으로 만든다.

## Task 6: Top-down SubQ orchestration

**Files**

- Create: `subqa/orchestrator.py`
- Modify: `pipeline.py`
- Modify: `tests/test_hierarchical_subqa.py`
- Modify: `tests/test_pipeline.py`

### TDD 순서

- [x] fake model과 qid 2개로 depth frontier가 qid/preorder 순서를 보존하고 최대 batch size로 chunk되는 실패 테스트를 작성한다.
- [x] chunk 경계의 prompt metadata가 output text/confidence와 정확히 대응하는 테스트를 작성한다.
- [x] root parse가 최초 부족 후 repair로 정확히 N개가 되는 성공과 repair 후에도 부족해 stage incomplete가 되는 실패를 작성한다.
- [x] descendant partial/failed expansion이 sibling과 다른 qid를 제거하지 않고 status/confidence/failure reason을 기록하는 테스트를 작성한다.
- [x] output count mismatch, model exception, malformed result가 artifact publish와 stage completion을 막는 테스트를 작성한다.
- [x] canonical artifact는 모든 qid가 끝난 뒤 한 번만 atomic writer 안에서 publish되는 테스트를 작성한다.
- [x] `RANK=1`에서 routine frontier/chunk log가 없고 main rank에서 한 번만 나오는 테스트를 작성한다.
- [x] 실패 테스트를 확인한 뒤 depth-generic top-down orchestration을 구현한다.
- [x] `subqa_depth>1`의 subq만 새 path로 dispatch하고 depth 1 regression test를 green으로 유지한다.

## Task 7: Bottom-up SubA orchestration

**Files**

- Modify: `subqa/orchestrator.py`
- Modify: `subqa/schema.py`
- Modify: `subqa/selection.py`
- Modify: `tests/test_hierarchical_subqa.py`

### TDD 순서

- [x] 모든 node direct answer가 먼저 생성되고 leaf selected source가 `leaf_direct`인 테스트를 작성한다.
- [x] depth 2 `[4,3]`에서 각 depth-1 node가 lexicographic child pair로 최대 3개 candidates를 가지며 support node ID가 정확한 테스트를 작성한다.
- [x] depth 3 fixture가 deepest internal level부터 처리되어 parent candidate가 immediate child selected answer만 사용하는 테스트를 작성한다.
- [x] `token_min_prob` 최대와 `seq_ppl` 최소, 동률 first candidate 선택을 각각 검증한다.
- [x] `condition_on_direct_suba=false`에서 candidate prompt만 direct draft를 빼고 direct fallback material은 유지하는 테스트를 작성한다.
- [x] 일부 descendant answer/candidate 실패 시 성공한 child/candidate를 사용하고, child 부족/모든 candidate 실패 시 `direct_fallback`하는 테스트를 작성한다.
- [x] depth-1 direct가 최초 실패 후 repair 성공하는 case와 repair 후 실패해 stage incomplete가 되는 case를 작성한다.
- [x] qid와 node ordering이 모든 partial failure에서 보존되고 system mismatch/schema corruption이 atomic publish를 막는 테스트를 작성한다.
- [x] 실패 테스트를 확인한 뒤 direct-first, generic bottom-up orchestration과 answer artifact validation을 구현한다.
- [x] depth-1 `suba_list/conf_suba` projection이 IDs `0..3` selected values와 정확히 같음을 확인한다.

## Task 8: Artifact namespace, manifest, dataset dependency 통합

**Files**

- Modify: `util/path.py`
- Modify: `util/artifacts.py`
- Modify: `dataset/base_dataset.py`
- Modify: `pipeline.py`
- Modify: `evaluation/c2r.py`
- Modify: `tests/test_artifacts.py`
- Modify: `tests/test_dataset_dependencies.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_c2r_evaluation.py`

### TDD 순서

- [x] depth 1은 기존 `N=4_M=2_K=4` 경로이고 depth>1은 canonical config SHA-256 앞 12자리 `D=2_H=<hash>`를 포함하는 테스트를 작성한다.
- [x] hierarchy semantics가 같으면 input mapping order와 무관하게 hash가 같고, 각 hierarchy field/fallback policy 차이는 namespace가 달라지는 테스트를 작성한다.
- [x] manifest가 모든 canonical hierarchy field를 기록하고 mismatch를 generation/load 전에 거부하는 테스트를 작성한다.
- [x] hierarchy field가 전혀 없는 legacy manifest는 depth 1 config에서만 허용하고 partial hierarchy manifest는 거부하는 테스트를 작성한다.
- [x] 명시적 `N=5` depth-1 config가 기존 namespace를 계속 주소 지정하는 회귀 테스트를 작성한다.
- [x] suba loader가 schema v2 tree를 검증해 hierarchy runner에 전달하고 malformed tree/qid를 거부하는 테스트를 작성한다.
- [x] refined loader는 hierarchy-only fields를 sample에 누출하지 않고 exact depth-1 flat projection만 전달하는 테스트를 작성한다.
- [x] hierarchy subq/suba rerun이 기존 DAG와 generation lineage에 따라 stale descendants를 무효화하는 테스트를 작성한다.
- [x] evaluator가 hierarchy config가 다른 dev/validation pair를 scoring 전에 거부하되 denominator, threshold, scorer 결과는 바뀌지 않는 테스트를 작성한다.
- [x] 실패 테스트를 확인한 뒤 schema canonicalization을 path/manifest/loader/evaluator가 공유하게 구현한다.
- [x] focused artifact/dependency/evaluator tests를 green으로 만든다.

## Task 9: Fake contract matrix와 factory/pipeline 일반성

**Files**

- Modify: `tests/test_hierarchical_subqa.py`
- Modify: `tests/test_pipeline.py`
- Modify: `model/__init__.py` only if production factory support is required

### TDD 순서

- [x] text multiple-choice, text open-ended, image multiple-choice, image open-ended, video multiple-choice의 다섯 fake sample matrix를 작성한다.
- [x] 각 case에서 template 호출, depth tree, bottom-up selection, flat refined input까지 end-to-end contract를 검증한다.
- [x] 모든 captured hierarchy prompt에서 GT와 main candidate 값이 없음을 검증한다.
- [x] vision이 text에서는 생략되고 image/video에서는 identity-preserving payload로 template에 전달되는지 검증한다.
- [x] `MODEL_REGISTRY`에 임시 arbitrary-name fake model을 등록하고 factory와 multi-stage pipeline이 Qwen/dataset 이름 검사 없이 동작하는 테스트를 작성한다.
- [x] test cleanup에서 registry entry를 복구해 다른 test에 상태를 누출하지 않는다.
- [x] focused matrix와 전체 pipeline regression을 green으로 만든다.

## Task 10: 문서/CPU 통합 검증과 독립 code review

**Files**

- Modify: `README.md`
- Modify: `tests/test_readme_commands.py`
- Modify only files required by review findings

### 검증 순서

- [x] README에 기본 `4/2/4`, depth-1 compatibility, primary depth-2 command, hierarchy artifact/namespace를 문서화한다.
- [x] README command test가 모든 일반 명령의 `4/2/4`와 depth-2 smoke의 exact hierarchy profile을 검증하도록 갱신한다.
- [x] 아래 focused CPU suite를 실행한다.

  ```bash
  /home/ywjang/miniconda3/envs/qwen2vl/bin/python -m unittest \
    tests.test_hierarchical_subqa \
    tests.test_pipeline \
    tests.test_dataset_dependencies \
    tests.test_artifacts \
    tests.test_c2r_evaluation \
    tests.test_vllm_config \
    tests.test_readme_commands
  ```

- [x] 아래 full CPU suite를 실행한다.

  ```bash
  /home/ywjang/miniconda3/envs/qwen2vl/bin/python \
    -m unittest discover -s tests -p 'test_*.py'
  ```

- [x] changed Python files에 `compileall`을 실행하고 `git diff --check`, `git status --short`, scoped diff를 검사한다.
- [x] 독립 reviewer에게 bug/regression, model/dataset coupling, artifact atomicity/lineage, missing validation을 우선하도록 code review를 요청한다.
- [x] blocker/high finding을 TDD로 수정하고 focused/full CPU gate를 다시 실행한다.

## Task 11: Qwen2.5-VL/MMMU 1문항 GPU acceptance

**Artifacts**

- Generate ignored artifacts under `output/smoke-hierarchical/MMMU/qwen2.5-vl-7b/val/...`
- Inspect: generated `run_manifest.json`, `subq_outputs.json`, `suba_outputs.json`, `refined_outputs.json`

### 검증 순서

- [x] `nvidia-smi -L`로 GPU 6 가용성을 확인한다.
- [x] 승인된 top-level wrapper로 다음 명령을 실행한다.

  ```bash
  /home/ywjang/.codex/bin/run_gpu.sh 6 -- env \
    HF_HOME=/home/ywjang/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_USE_V1=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    /home/ywjang/miniconda3/envs/qwen2vl/bin/python main.py --options \
    runner.mode=multi_stage \
    model.model_name=qwen2.5-vl-7b \
    model.tensor_parallel_size=1 \
    model.enforce_eager=true \
    dataset.dataset_name=MMMU \
    dataset.split=val \
    dataset.num_data=1 \
    runner.N=4 runner.M=2 runner.K=4 \
    runner.subqa_depth=2 \
    'runner.branching_by_depth=[4,3]' \
    runner.suba_M=2 runner.suba_K=3 \
    runner.suba_confidence_type=token_min_prob \
    runner.condition_on_direct_suba=true \
    runner.output_dir=output/smoke-hierarchical
  ```

- [x] manifest의 `subq`, `suba`, `base`, `refined`가 모두 현재 generation에서 completed인지 확인한다.
- [x] qid가 하나도 누락되지 않았고 tree가 depth-1 4개/depth-2 12개와 parent/depth/preorder 불변식을 만족하는지 검사한다.
- [x] 모든 depth-1 node에 valid selected answer와 `confidence` 또는 명시적 `direct_fallback` source가 있는지 검사한다.
- [x] final refined 입력/산출물이 depth-1 QA 네 개만 참조하고 descendant/GT/main choices가 hierarchy context에 누출되지 않았는지 검사한다.
- [x] routine application log가 main rank에서만 한 번씩 출력되었는지 검사한다.
- [x] GPU smoke 뒤 full CPU suite와 `git diff --check`를 다시 실행하고 `CLOSURE.md` evidence를 갱신한다.

## 완료 조건

- Task 1–10의 CPU test/evidence가 모두 green이다.
- Task 11의 Qwen2.5-VL/MMMU val 1문항이 네 stage를 완료하고 artifact invariant를 통과한다.
- 독립 code review blocker/high finding이 남지 않는다.
- Qwen3, MMLU, DramaQA full run이나 MMMU 150/900 성능 결과는 closure blocker가 아니다.
