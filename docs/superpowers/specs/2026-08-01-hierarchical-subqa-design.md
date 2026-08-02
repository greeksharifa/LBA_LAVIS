# 모델·벤치마크 독립 Hierarchical Sub-QA 설계

## Objective

기존 flat Sub-QA 경로와 최종 refined/evaluation 계약을 보존하면서, 공통 model/dataset protocol 위에서 임의 깊이의 Sub-QA tree를 생성하고 bottom-up으로 답을 정제한다. 주 설정은 `N=4, M=2, K=4`, hierarchy depth 2, branching `[4,3]`이며 Qwen2.5-VL/MMMU는 완료 검증 대상일 뿐 orchestration의 분기 조건이 아니다.

성공한 depth-1 네 노드의 selected Sub-A만 기존 refined stage에 전달한다. hierarchy prompt에는 main multiple-choice 선택지와 GT를 노출하지 않는다.

## Scope

포함 범위:

- 전역 `N/M/K` 기본값을 `4/2/4`로 변경하고 hierarchy 설정을 검증한다.
- 공통 `GenerationResult`와 `generate_results()` protocol을 정의한다.
- text/image/video를 구분하는 Qwen chat-template adapter 결함을 수정한다.
- stable node ID, preorder tree schema, deterministic evidence subset을 구현한다.
- top-down question expansion과 bottom-up answer refinement를 구현한다.
- schema v2 artifact, hierarchy namespace, manifest compatibility를 기존 stage DAG에 통합한다.
- fake model과 text/image/video dataset contract matrix로 일반성을 검증한다.
- Qwen2.5-VL/MMMU val 1문항 GPU smoke로 실제 adapter 경계를 검증한다.

## Non-Goals

- 모델명이나 dataset 이름별 hierarchy 분기
- raw direct/refined confidence 간 threshold gate
- generic default question으로 부족한 child를 padding하는 fallback
- hierarchy stage 중간 checkpoint/resume
- Qwen3, MMLU, DramaQA adapter 신규 구현 또는 실제 full run
- MMMU 150/900 성능 개선·정확도 비교

## Success Criteria

- 기본 설정이 `N=4, M=2, K=4`이고 기존 `N=5` artifact는 명시적 `runner.N=5`로 계속 주소 지정할 수 있다.
- depth 1은 기존 flat pipeline과 artifact 경로를 유지한다.
- depth 2, branching `[4,3]`은 qid마다 depth-1 노드 4개와 depth-2 노드 12개를 stable ID와 preorder 순서로 만든다.
- hierarchy orchestration은 model/dataset 이름, Qwen token, vLLM completion shape를 참조하지 않는다.
- 모든 question node는 direct answer를 가지며 internal node는 가능한 경우 immediate child-QA 2개로 최대 3개 refined candidate를 만든다.
- normalized confidence 최고 후보를 first-maximum 규칙으로 선택하고, 유효 후보가 없으면 direct answer를 명시적으로 선택한다.
- depth-1 projection은 `subq_list`, `suba_list`, `conf_suba`가 ID `0,1,2,3`과 정확히 대응하며 기존 refined/evaluator만 이 projection을 소비한다.
- descendant의 partial failure는 sibling이나 다른 qid를 제거하지 않고, 필수 root expansion 또는 depth-1 direct answer 실패는 stage를 완료 처리하지 않는다.
- 전체 CPU test suite와 fake contract matrix가 통과하고 Qwen2.5-VL/MMMU val 1문항의 네 stage가 완료된다.

## Related Docs

- 기존 multi-stage 경계: `docs/superpowers/specs/2026-08-01-multi-stage-mmmu-design.md`
- artifact lineage 경계: `docs/superpowers/specs/2026-08-01-artifact-lineage-invalidation-design.md`
- provenance 검증 경계: `docs/superpowers/specs/2026-08-01-artifact-provenance-hardening-design.md`

별도 `docs/principles/` 문서는 현재 저장소에 없다. 이 설계는 기존 stage DAG, fail-closed manifest 검증, main-rank routine logging 규칙을 상위 제약으로 따른다.

## System Area

주 변경 경계는 다음과 같다.

```text
config/configs.py, config/default.yaml
        |
        v
subqa/schema.py --------> util/path.py, util/artifacts.py
        |
        +--> subqa/prompts.py + subqa/parsing.py
        |
        +--> subqa/orchestrator.py <--> model GenerationResult protocol
        |                              <--> dataset sample protocol
        v
subq_outputs.json -> suba_outputs.json -> existing refined -> evaluator
```

`pipeline.py`는 depth에 따라 legacy flat runner와 hierarchy runner를 dispatch한다. `dataset/base_dataset.py`는 schema v2 tree를 검증해 `suba`에 전달하지만, `refined`에는 depth-1 flat projection만 노출한다. `util/artifacts.py`와 `util/path.py`는 hierarchy config를 namespace와 compatibility에 포함한다.

## Configuration Contract

`config/default.yaml`의 runner 기본값은 다음과 같다.

```yaml
runner:
  N: 4
  M: 2
  K: 4

  subqa_depth: 1
  branching_by_depth: null

  suba_M: 2
  suba_K: 3
  suba_confidence_type: token_min_prob
  condition_on_direct_suba: true

  subqa_max_nodes: 64
  subqa_repair_attempts: 1
  subqa_generation_batch_size: 64
  subqa_schema_version: 2
```

Normalization 규칙:

- `subqa_depth`는 positive non-bool integer다.
- `subqa_depth == 1`이고 `branching_by_depth == null`이면 canonical branching은 `[N]`이다. 이 경우 기존 flat `subq/suba` runner를 사용한다.
- `subqa_depth > 1`이면 branching을 명시해야 하며 길이는 depth와 같아야 한다.
- branching이 null이 아니면 모든 depth에서 길이가 `subqa_depth`와 같아야 한다. 따라서 depth 1에서 `[N, x]`는 거부한다.
- 모든 branching 값은 positive non-bool integer이고 `branching[0] == N`이어야 한다.
- root를 제외한 question node 수 `sum(prod(branching[:level]))`가 `subqa_max_nodes` 이하여야 한다. `subqa_max_nodes`는 positive non-bool integer다.
- SubA 조합 검증은 child question을 가진 question-node parent level, 즉 canonical `branching[1:]`에만 적용한다. 각 `child_count`에 대해 `1 <= suba_M <= child_count`와 `1 <= suba_K <= C(child_count, suba_M)`가 성립해야 한다. Root anchor만 expansion하는 depth 1에는 이 검증을 적용하지 않는다.
- `subqa_generation_batch_size >= 1`, `subqa_repair_attempts >= 0`이며 bool은 정수로 허용하지 않는다.
- `suba_confidence_type`은 공통 confidence utility가 지원하는 metric만 허용한다. 초기 지원값은 `token_min_prob`, `seq_ppl`이다.
- `condition_on_direct_suba`는 strict bool이고 `subqa_schema_version`은 구현이 지원하는 정수 버전 `2`여야 한다.
- `M/K`는 최종 refined 후보 전용이고 `suba_M/suba_K`는 internal node 후보 전용이다.

Config validation은 모델 로드와 artifact 접근보다 먼저 수행한다. 잘못된 조합은 조용히 보정하지 않고 field를 포함한 `ValueError`로 실패한다.

## Model Protocol

공통 generation 결과는 vLLM 타입을 노출하지 않는다. `GenerationResult`는
`model/protocol.py` 같은 lightweight module에 두어 `MODEL_REGISTRY`나 CPU-only
tooling import가 torch/vLLM을 eager import하지 않게 한다. Fake model은
`C2RFramework.__init__()` 또는 model loading lifecycle을 상속할 필요가 없는
structural protocol로 검증한다.

```python
@dataclass(frozen=True)
class GenerationResult:
    text: str
    confidence: dict[str, float]

class C2RFramework:
    def apply_chat_template(
        self,
        text_prompt: str,
        vision=None,
        mm_uuids: str | None = None,
    ) -> object: ...

    def generate_results(
        self,
        prompts: list[object],
    ) -> list[GenerationResult]: ...
```

계약:

- `generate_results()`는 입력 순서와 동일한 길이·순서의 `GenerationResult`를 반환한다.
- `text`는 문자열이고 `confidence`는 adapter가 계산한 지원 metric의 finite numeric raw 값이다. 모든 hierarchy 결과는 최소한 `{runner.confidence_type, suba_confidence_type}`의 union에 해당하는 key를 제공해야 한다.
- `token_min_prob`은 finite non-bool numeric `[0,1]`, `seq_ppl`은 finite non-bool numeric `>=0`이어야 한다. 필수 metric 누락, non-finite 값, domain 밖 값은 node-level invalid answer가 아니라 output protocol corruption이므로 stage 전체를 실패시킨다.
- Qwen adapter는 vLLM output의 completion text, cumulative log probability, token log probability를 `GenerationResult`로 변환한다.
- hierarchy code는 `.outputs[0]`, token IDs, logprobs, Qwen placeholder를 읽지 않는다.
- 기존 `generate()`와 legacy postprocessor는 depth-1 호환성을 위해 유지한다.
- fake 또는 신규 모델은 `apply_chat_template()`과 `generate_results()`만 충족하면 hierarchy runner에서 동작한다.
- prompt/output count mismatch는 전체 stage system error다.
- depth-1 legacy fake/model은 `generate()`만 구현해도 기존 flat runner에서 계속 동작한다. `generate_results()` 요구는 depth > 1 hierarchy dispatch 경계에만 적용한다.

공통 `util/confidence.py`는 `normalize_confidence(value, metric)`과 deterministic first-maximum selection을 소유한다. `token_min_prob`는 higher-is-better raw score를 그대로 검증·사용하고, `seq_ppl`은 `min(1, 1 / max(value, 1e-12))`로 변환한다. evaluator와 hierarchy는 이 구현을 함께 사용한다. Artifact는 adapter가 제공한 지원 raw metric을 보존하며 flat projection은 최소한 기존 refined consumer의 `runner.confidence_type`과 internal selection의 `suba_confidence_type`을 포함한다.

### Qwen modality adapter

- `data_type=text`이며 vision이 `None`이면 vision token, `multi_modal_data`, `multi_modal_uuids` 없이 text-only chat prompt를 반환한다.
- `data_type=image`는 image placeholder와 image payload/UUID를 유지한다.
- `data_type=video`는 video placeholder와 video payload/UUID를 유지한다.
- 지원하지 않는 modality는 값이 포함된 명확한 unsupported-modality 오류를 낸다.
- image/video에서 vision payload가 잘못된 경우 template 경계에서 명시적으로 실패한다.
- 기존 adapters가 video를 ndarray 또는 `(video, metadata)` wrapper로 전달할 수 있으므로 text fix 과정에서 video payload shape나 placeholder count를 임의로 재정의하지 않는다. 기존 representation별 regression fixture를 먼저 고정한다.

## Dataset Protocol

Hierarchy runner가 sample에서 읽는 필드는 다음뿐이다.

```text
required: qid, main_q, question_type, data_type
optional: vision
```

`qid`는 artifact key로 사용하기 전에 문자열로 canonicalize한다. `main_q`, `question_type`, `data_type`은 non-empty string이어야 한다. text sample은 `vision` 없이 유효하다. image/video payload는 runner가 검사·변형하지 않고 `apply_chat_template()`에 그대로 전달한다.

`candidate_list`, `gt_ans`, dataset-specific metadata는 hierarchy prompt builder에 전달하지 않는다. 기존 base/refined prompt는 multiple-choice 선택지를 계속 사용할 수 있다. Hierarchy runner는 dataset registry 이름을 검사하지 않는다.

## Canonical Tree Contract

Stable path ID를 사용한다. root ID는 `root`, depth-1 sibling은 `0..N-1`, descendant는 parent path 뒤에 `.<sibling_index>`를 붙인다.

```text
root
├── 0
│   ├── 0.0
│   ├── 0.1
│   └── 0.2
├── 1
├── 2
└── 3
```

Serialization은 preorder이며 sibling index 오름차순이다. `root`는 orchestration anchor이고 `nodes[]`에는 생성된 question node만 저장한다. 각 node는 다음 필드를 가진다.

```text
id, parent_id, depth, sibling_index, question, child_ids,
expansion_status, expansion_confidence, failure_reason
```

불변식:

- ID, parent, depth, sibling index가 서로 일관되고 중복 ID가 없다.
- `child_ids`는 sibling index 순서이며 모두 해당 node를 parent로 참조한다.
- 모든 node depth는 `1..max_depth`이고 preorder 순서를 만족한다.
- `expanded` node의 child count는 목표 branching 이하이며 status가 실제 count를 표현한다.
- `failure_reason`은 실패/partial 상태에서만 non-empty이고 성공 상태에서는 `null`이다.
- 한 completion이 여러 child question을 만들므로 child별 token confidence를 위조하지 않는다. raw confidence는 parent의 `expansion_confidence`에 귀속한다.

`expansion_status` enum은 `leaf`, `expanded`, `partial`, `failed`다. `leaf`는 max-depth이며 child가 없고, `expanded`는 목표 child count를 만족하며, `partial`은 1개 이상이지만 목표보다 적고, `failed`는 child가 0개다. Answer와 candidate의 `status` enum은 `valid/invalid`이며 invalid record만 non-empty `failure_reason`을 갖는다. Descendant direct answer가 invalid이고 fallback도 불가능하면 `selected.status=invalid`로 남겨 부모 evidence에서 제외한다.

`subqa/schema.py`가 config normalization, stable ID, tree validation, preorder serialization, depth-1 projection을 소유한다.

## Question Expansion and Parsing

Top-down expansion context는 다음과 같다.

- root child: main question + modality + 요구 child 수
- descendant child: main question + ancestor path + target parent question + modality + 요구 child 수
- GT와 main choices는 포함하지 않는다.

Parser는 다음 순서로 동작한다.

1. completion에서 non-empty candidate line을 추출한다.
2. list number/bullet prefix를 제거하고 whitespace를 하나로 정규화한다.
3. 비교 key는 casefold 후 끝 문장부호와 주변 공백을 제거한다.
4. 첫 출현을 보존하며 중복을 제거하고 목표 개수까지만 취한다.
5. 부족하면 현재 유효 질문을 포함한 repair prompt를 최대 `subqa_repair_attempts`만큼 실행한다.
6. repair 결과도 같은 parser와 중복 규칙을 거친다.

Expansion confidence는 accepted question을 하나 이상 새로 추가한 마지막 generation attempt의 raw confidence를 기록한다. 최초 completion만 기여했다면 최초 confidence를, repair가 새 질문을 추가했다면 마지막으로 기여한 repair confidence를 사용한다. Root에는 같은 규칙으로 `conf_subq`를 기록한다.

Generic default 질문 padding은 금지한다. root가 repair 후 정확히 `N`개가 아니면 subq stage 실패다. descendant는 목표보다 적은 child를 가진 partial node 또는 zero-child failed node가 될 수 있다.

Depth별 parent frontier의 prompt는 canonical qid/node 순서로 만들고 최대 `subqa_generation_batch_size` 단위로 `generate_results()`에 전달한다. 각 chunk의 metadata와 output을 위치 기반으로 엄격히 대조한다.

## Answer Generation and Selection

모든 question node에 direct answer를 먼저 생성한다.

- leaf direct context: main question + ancestor question path + target question + modality
- internal direct context: 동일한 direct context이며 child answer를 포함하지 않는다.
- 답은 최대 한 문장으로 요청하되 유효성은 non-empty normalized text로 판단한다.

그 뒤 deepest internal level부터 bottom-up으로 처리한다.

1. leaf는 valid direct answer를 `leaf_direct`로 선택한다.
2. internal node의 valid immediate child selected QA를 sibling 순서로 수집한다.
3. child가 `suba_M`개 이상이면 lexicographic combination을 만들고 앞에서 `suba_K`개까지만 사용한다.
4. 각 subset마다 main question, target question, 선택한 child-QA, 선택적으로 direct draft를 포함한 refined candidate prompt를 만든다.
5. `condition_on_direct_suba=false`이면 candidate prompt에서만 direct draft를 제외한다. direct answer 생성과 fallback은 유지한다.
6. valid candidate 중 normalized confidence의 first maximum을 선택하고 source를 `confidence`로 기록한다.
7. valid candidate가 없거나 child가 부족하면 valid direct answer를 `direct_fallback`으로 선택한다.

`N=3, suba_M=2, suba_K=3`의 evidence subset은 정확히 `(0,1)`, `(0,2)`, `(1,2)`다. 무작위 sampling은 사용하지 않는다.

Direct/candidate failure 정책:

- depth-1 direct answer는 필수다. 최초 생성과 한 번의 repair 후에도 invalid하면 suba stage를 완료하지 않는다.
- descendant direct answer 실패는 해당 node를 invalid로 기록하며 sibling과 다른 qid는 유지한다.
- valid child-QA가 `suba_M`보다 적으면 direct fallback한다.
- 후보가 일부 실패하면 성공 후보 중에서만 선택한다.
- system exception, output count mismatch, schema corruption은 stage 전체 실패다.
- 어떤 failure path에서도 qid record를 조용히 누락하지 않는다.

## Artifact Contract

### `subq_outputs.json`

qid별 record는 legacy projection과 canonical tree를 함께 가진다.

```json
{
  "subq_list": ["..."],
  "conf_subq": {"seq_ppl": 0.0, "token_min_prob": 0.0},
  "subq_tree": {
    "schema_version": 2,
    "root_id": "root",
    "max_depth": 2,
    "nodes": []
  }
}
```

`subq_list`는 depth-1 ID 순서다. `conf_subq`는 legacy field를 유지하며 root expansion의 raw confidence mapping을 사용한다. Tree node의 expansion confidence가 더 상세한 source of truth다.

### `suba_outputs.json`

qid별 record는 legacy projection과 node answer map을 함께 가진다.

```json
{
  "suba_list": ["..."],
  "conf_suba": {"seq_ppl": [], "token_min_prob": []},
  "answers_by_node": {
    "0": {
      "direct": {},
      "candidates": [],
      "selected": {}
    }
  }
}
```

Direct answer와 candidate는 최소 다음을 가진다.

```text
answer, confidence, status, failure_reason
```

Candidate에는 `support_node_ids`가 추가된다. `selected`는 선택된 answer/confidence와 `source`를 가지며 source는 `leaf_direct`, `confidence`, `direct_fallback` 중 하나다.

Flat `suba_list`와 각 `conf_suba[metric]`은 depth-1 ID `0..N-1`의 selected 값과 같은 순서를 갖는다. descendant answer는 flat projection에 들어가지 않는다.

Stage artifact는 모든 qid의 canonical record를 검증한 뒤 destination directory의 temporary file에서 atomic replace한다. 중간 canonical artifact를 publish하지 않는다. 현재 generation이 유효할 때만 기존 `mark_stage_complete(..., artifact_writer=...)` 경계 안에서 publish한다.

## Namespace and Manifest Compatibility

- depth 1의 경로는 기존 `{output_dir}/{dataset}/{model}/{split}/N=4_M=2_K=4` 형식을 유지한다.
- depth > 1은 기존 signature directory 바로 아래에 단일 child directory를 추가한다: `{...}/N=<N>_M=<M>_K=<K>/D=<depth>_H=<hash>`.
- `<hash>`는 canonical hierarchy config를 stable-key JSON으로 serialize한 UTF-8 bytes의 SHA-256 앞 12자리 lowercase hex다.
- canonical hash input에는 depth, branching, `suba_M/K`, internal confidence, direct conditioning, max nodes, repair attempts, batch size, schema version, fallback policy가 포함된다.
- `output_dir`, dataset/model name, split처럼 기존 namespace segment이거나 hierarchy semantics와 무관한 값은 hash input에 중복 포함하지 않는다.

Manifest `config`에는 canonical hierarchy fields와 fallback policy를 명시적으로 기록한다. 생성/loader/evaluator가 같은 canonicalization 함수를 공유한다. dev/validation pair evaluator는 hierarchy config가 다르면 scoring 전에 거부한다.

Hierarchy field가 전혀 없는 legacy manifest는 depth 1로만 해석한다. 일부 hierarchy field만 있는 malformed manifest는 legacy로 추정하지 않고 거부한다. 기존 `N=5` flat artifact는 사용자가 동일한 기존 config를 명시했을 때만 같은 namespace와 manifest validation을 통과한다.

## Pipeline Integration

- `runner.mode in {subq, suba}`이고 `subqa_depth > 1`일 때만 hierarchy orchestrator로 dispatch한다.
- depth 1의 `subq/suba`, 모든 `base/refined`는 legacy runner를 유지한다.
- stage order와 dependency DAG `subq -> suba`, `{subq,suba,base} -> refined`는 변하지 않는다.
- hierarchy `suba` dataset loader는 schema v2 tree와 dependency generation을 검증해 orchestrator에 전달한다.
- refined dataset loader는 flat depth-1 projection만 기존 sample keys로 전달한다.
- 새 subq generation은 기존 invalidation 규칙으로 stale suba/refined를 무효화한다. 새 suba generation은 refined를 무효화한다.
- evaluator의 question denominator, answer scoring, threshold selection은 변경하지 않는다. manifest pair compatibility에 hierarchy fields만 추가한다.
- routine application log는 unset/`RANK=0` main process에서만 한 번 출력한다.

## Failure Modes

| Failure | Scope | Required result |
|---|---|---|
| invalid hierarchy config | run | model load 전 `ValueError` |
| root parse/repair 부족 | subq stage | incomplete, artifact 미게시 |
| descendant expansion 부족 | node/subtree | partial/failed 기록, sibling 유지 |
| generation output count mismatch | stage | incomplete, artifact 미게시 |
| depth-1 direct invalid after repair | suba stage | incomplete, artifact 미게시 |
| descendant direct invalid | node | invalid 기록, parent가 나머지 child 사용 |
| child 부족 또는 모든 candidate invalid | internal node | direct fallback |
| malformed tree/answer schema | stage/load | fail closed |
| manifest hierarchy mismatch | load/eval | scoring/generation 전 거부 |
| unsupported modality | model template | 명시적 오류 |

Stage failure는 manifest의 해당 generation을 completed로 표시하지 않는다. 실패한 시도 때문에 이전 generation artifact를 새 generation으로 오인해서는 안 된다.

## Validation and Rollout

### CPU contract validation

Fake model/dataset matrix는 다음을 포함한다.

| Modality | Question type | Vision |
|---|---|---|
| text | multiple choice | 없음 |
| text | open ended | 없음 |
| image | multiple choice | image list |
| image | open ended | image list |
| video | multiple choice | video payload |

모든 case에서 prompt/template 호출, tree 생성, bottom-up selection, flat projection을 검증하고 hierarchy prompt에 GT와 choices가 없음을 검사한다. 임시 fake model registry entry로 factory/pipeline이 이름 독립적으로 동작함도 검증한다.

Focused CPU gate:

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

Full CPU gate:

```bash
/home/ywjang/miniconda3/envs/qwen2vl/bin/python \
  -m unittest discover -s tests -p 'test_*.py'
```

추가로 changed Python compile, `git diff --check`, 독립 code review를 수행한다.

### GPU acceptance

Qwen2.5-VL/MMMU val 1문항만 primary completion gate다.

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

Acceptance evidence는 네 stage completed manifest, 4/12 tree invariant, 네 depth-1 selected answers, 각 internal selection source, final refined가 네 flat QA만 사용한다는 trace/artifact 검사, main-rank routine log 단일 출력을 포함한다.

## Risks and Mitigations

- Prompt volume은 depth와 branching에 따라 빠르게 증가한다. Config validation의 max-node bound와 generation batch size로 제한한다.
- Partial tree가 artifact consumer를 복잡하게 만들 수 있다. Node status/failure reason을 필수화하고 flat projection은 depth-1 completeness를 강제한다.
- Confidence 의미가 adapter마다 달라질 수 있다. Raw metric을 보존하고 공통 normalization/지원 metric validation만 orchestration에서 수행한다.
- Legacy artifact가 hierarchy artifact로 오인될 수 있다. Depth-1-only legacy rule과 partial hierarchy manifest fail-closed 정책을 적용한다.
- Qwen modality fix가 image/video payload를 회귀시킬 수 있다. text/image/video template unit test와 실제 MMMU image smoke를 모두 gate로 둔다.

## Alternatives Considered

- Qwen/MMMU 전용 hierarchy runner: 구현은 빠르지만 model/dataset protocol 목표를 위반하므로 제외한다.
- 모든 depth를 flat list로만 저장: parent/partial failure provenance와 generic recursion을 검증할 수 없어 제외한다.
- 부족 child를 generic question으로 padding: 의미 없는 evidence를 정상 node처럼 보이게 하므로 제외한다.
- random child subset: seed와 무관한 artifact 재현성을 해치므로 lexicographic combination을 선택한다.
- direct/refined confidence threshold gate: 서로 다른 prompt의 raw confidence calibration을 전제하므로 이번 범위에서 제외한다.
