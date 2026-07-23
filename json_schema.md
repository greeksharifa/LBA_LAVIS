# SQ-InstructBLIP

각 파일의 첫 번째 json element를 그대로 출력하면 아래와 같습니다.

  $VQA_INTROSPECT_ROOT/ywjang_sub_qas/vqa_introspect_questioner_subquestions_val_58f05e6001b8.json

  {
    "question_id": "565248001",
    "main_question": "are they at the beach?",
    "image_path": "$COCO_ROOT/images/val2014/COCO_val2014_000000565248.jpg",
    "sub-questions": [
      "are they on the beach?",
      "are they on the beach?",
      "is there a beach in the background?"
    ]
  }

  $AOKVQA_ROOT/ywjang_sub_qas/aok_vqa_questioner_subquestions_val_58f05e6001b8.json

  {
    "question_id": "22jbM6gDxdaMaunuzgrsBB",
    "main_question": "What is in the motorcyclist's mouth?",
    "image_path": "$COCO_ROOT/images/val2014/COCO_val2014_000000461751.jpg",
    "sub-questions": [
      "is there a cigarette in the man's mouth?",
      "is there a cigarette in the man's mouth?",
      "is there a cigarette in the man's mouth?"
    ]
  }

## VQA-Introspect
  - $VQA_INTROSPECT_ROOT/ywjang_sub_qas/judge_ready_val_sq_instructblip_recovered_20240503024_nonempty.json
  - 전체 22,793개 중 generated_sub_qa가 비어 있지 않은 21,492개만 남겼습니다.

  GT와 generated sub-QA가 전부 같은 건 아닙니다. non-empty 21,492개 기준으로:

  - exact 동일: 4,584개 (21.3%)
  - 순서만 무시하고 같은 pair 집합: 8,704개 (40.5%)
  - 즉 12,788개 (59.5%)는 GT와 pair 집합 자체가 다릅니다.

  예를 들면:

  - 565248001: generated는 1개만 있고 GT는 2개입니다.
  - 276693003: generated는 1개, GT는 4개입니다.
  - 119210000: generated는 1개, GT는 2개입니다.

  그래서 결론은 이렇습니다.

  - generated_sub_qa가 GT를 그대로 복사한 파일은 아닙니다.
  - 다만 일부 샘플에서는 모델 출력이 GT와 정확히 같을 수는 있습니다. 그 비율이 위 숫자입니다.

## AOK-VQA


# C2R (UniQA)

모든 경로는 sub_qas 기준 상대경로입니다. 표에 없는 조합은 현재 최종 sub_qas*.json 파일이 없습니다. 실제 폴더명은 VQAIntrospect, MMMUPro입니다.

  | 데이터셋 | 남긴 모델 기준 파일 |
  |---|---|
  | AOKVQA | qwen2.5-vl-7b: AOKVQA/qwen2.5-vl-7b/test/sub_qas.json<br>llava-onevision-7b: AOKVQA/llava-onevision-7b/test/sub_qas.json<br>gemma-3-4b-it: AOKVQA/gemma-3-4b-it/test/sub_qas.json |
  | VQAIntrospect | 선택한 4개 모델 하위 파일은 없음<br>dataset-level: VQAIntrospect/sub_qas_GT.json, VQAIntrospect/sub_qas_irrSubQ.json, VQAIntrospect/sub_qas_wrongSubA.json |
  | MMLU | qwen2.5-vl-7b: MMLU/qwen2.5-vl-7b/test/sub_qas.json, MMLU/qwen2.5-vl-7b/test/sub_qas_cog.json, MMLU/qwen2.5-vl-7b/test/sub_qas_llm_judge.json, MMLU/qwen2.5-vl-7b/val/sub_qas.json<br>qwen2-vl-2b: MMLU/qwen2-vl-2b/test/sub_qas.json<br>llava-
  onevision-7b: MMLU/llava-onevision-7b/test/sub_qas.json, MMLU/llava-onevision-7b/test/sub_qas_llm_judge.json<br>gemma-3-4b-it: MMLU/gemma-3-4b-it/test/sub_qas.json |
  | MMMU | qwen2.5-vl-7b: MMMU/qwen2.5-vl-7b/test/sub_qas.json, MMMU/qwen2.5-vl-7b/test/sub_qas_cog.json, MMMU/qwen2.5-vl-7b/test/sub_qas_llm_judge.json, MMMU/qwen2.5-vl-7b/test_blind/sub_qas.json, MMMU/qwen2.5-vl-7b/val/sub_qas.json<br>qwen2-vl-2b: MMMU/
  qwen2-vl-2b/test/sub_qas.json<br>llava-onevision-7b: MMMU/llava-onevision-7b/test/sub_qas.json, MMMU/llava-onevision-7b/test/sub_qas_llm_judge.json, MMMU/llava-onevision-7b/test_blind/sub_qas.json<br>gemma-3-4b-it: MMMU/gemma-3-4b-it/test/sub_qas.json,
  MMMU/gemma-3-4b-it/test_blind/sub_qas.json |
  | MMMUPro | qwen2.5-vl-7b: MMMUPro/qwen2.5-vl-7b/test/sub_qas.json, MMMUPro/qwen2.5-vl-7b/10_options/test/sub_qas.json |
  | StrategyQA | qwen2.5-vl-7b: StrategyQA/qwen2.5-vl-7b/test/sub_qas.json, StrategyQA/qwen2.5-vl-7b/test/sub_qas_cog.json, StrategyQA/qwen2.5-vl-7b/test/sub_qas_llm_judge.json<br>qwen2-vl-2b: StrategyQA/qwen2-vl-2b/test/sub_qas.json<br>llava-onevision-
  7b: StrategyQA/llava-onevision-7b/test/sub_qas.json, StrategyQA/llava-onevision-7b/test/sub_qas_llm_judge.json<br>gemma-3-4b-it: StrategyQA/gemma-3-4b-it/test/sub_qas.json |
  | EgoSchema | qwen2.5-vl-7b: EgoSchema/qwen2.5-vl-7b/test/sub_qas.json, EgoSchema/qwen2.5-vl-7b/test/sub_qas_cog.json, EgoSchema/qwen2.5-vl-7b/test/sub_qas_llm_judge.json, EgoSchema/qwen2.5-vl-7b/test_blind/sub_qas.json, EgoSchema/qwen2.5-vl-7b/val/
  sub_qas.json<br>qwen2-vl-2b: EgoSchema/qwen2-vl-2b/test/sub_qas.json<br>llava-onevision-7b: EgoSchema/llava-onevision-7b/test/sub_qas.json, EgoSchema/llava-onevision-7b/test/sub_qas_llm_judge.json, EgoSchema/llava-onevision-7b/test_blind/
  sub_qas.json<br>gemma-3-4b-it: EgoSchema/gemma-3-4b-it/test/sub_qas.json, EgoSchema/gemma-3-4b-it/test_blind/sub_qas.json |

  Schema
  대부분은 아래 표준 구조입니다. sub_qas.json, sub_qas_cog.json, sub_qas_GT.json, sub_qas_irrSubQ.json, sub_qas_wrongSubA.json이 여기에 해당합니다.

  {
    "<qid>": {
      "main_q": "string",
      "sub_q_list": ["string", "..."],
      "sub_a_list": ["string", "..."],
      "sub_a_conf_list": [number, "..."],
      "sub_a_ppl_list": [number, "..."],
      "sub_a_min_prob_list": [number, "..."]
    }
  }

  sub_qas_llm_judge.json은 표준 구조에 아래 필드가 추가됩니다.

  {
    "<qid>": {
      "...": "...",
      "judged_sub_q_indices": [int, int],
      "judged_sub_a_indices": [int, int],
      "judged_sub_qa_indices": [int, int],
      "sub_q_success": true,
      "sub_a_success": true,
      "sub_qa_success": true
    }
  }

  예외로, 이번 필터 범위 안에서는 아래 3개 파일이 main_q 없이 저장된 compact 형태입니다.

  - EgoSchema/qwen2.5-vl-7b/test/sub_qas.json
  - MMMU/gemma-3-4b-it/test/sub_qas.json
  - MMMU/qwen2.5-vl-7b/test/sub_qas.json

  {
    "<qid>": {
      "sub_q_list": ["string", "..."],
      "sub_a_list": ["string", "..."],
      "sub_a_conf_list": [number, "..."],
      "sub_a_ppl_list": [number, "..."],
      "sub_a_min_prob_list": [number, "..."]
    }
  }

  보충하면:

  - 각 리스트는 같은 index끼리 한 sub-QA를 이룹니다.
  - number는 파일에 따라 float 또는 int입니다.
  - GT/irrSubQ/wrongSubA도 키 구조는 표준과 같습니다. 다만 리스트 길이가 보통 2~3개입니다.

---

# INQUIRER
새 QA 원본은 repo 바깥의 $INQUIRER_SOURCE_ROOT 아래에 저장돼 있다. 학습 코드 기준으로 보면 KG/QGen 산출물 경로는 다음이 핵심이다.

  - DramaQA scene-level: $INQUIRER_SOURCE_ROOT/prompts/AnotherMissOhQA_train_set_add_sceneprob.json
  - DramaQA shot-level: $INQUIRER_SOURCE_ROOT/prompts/AnotherMissOhQA_train_set_add_shotprob.json
  - STAR: $INQUIRER_SOURCE_ROOT/gen_starQA/STAR_train_add_prob.json
  - TVQA: $INQUIRER_SOURCE_ROOT/gen_tvqa/tvqa_train_add_prob.jsonl
  - TVQA q_type 포함 파생본: $INQUIRER_SOURCE_ROOT/gen_tvqa/tvqa_train_add_prob_with_qtype.jsonl
  - How2QA KG 후처리본: $INQUIRER_SOURCE_ROOT/gen_how2qakg/how2qa_train_kg_prob.json
  - How2QA KG 정규화 원본: $INQUIRER_SOURCE_ROOT/gen_how2qakg/results_kg_filled.json

  repo 안에는 원본 QA와 합쳐진 병합본도 있다.

  - DramaQA KG 병합본: .
  - STAR KG 병합본 디렉터리: filtered
  - TVQA KG 병합본 디렉터리: filtered

  구체적으로는 아래 패턴이다.

  - DramaQA: dramaqa_train_unfiltered_KG.json, dramaqa_train_filtered05_KG.json, dramaqa_train_filtered075_KG.json, dramaqa_train_filtered0875_KG.json
  - STAR: filtered/STAR_train_unfiltered_KG.json, filtered/STAR_train_filtered05_KG.json, filtered/STAR_train_filtered075_KG.json, filtered/
    STAR_train_filtered0875_KG.json
  - TVQA: filtered/TVQA_train_unfiltered_KG.jsonl, filtered/TVQA_train_filtered05_KG.jsonl, filtered/TVQA_train_filtered075_KG.jsonl, filtered/
    TVQA_train_filtered0875_KG.jsonl

  근거는 학습 로더가 이 외부 경로들을 직접 읽는다는 점이다: dataloader/dramaqa.py:13, dataloader/dramaqa.py:14, dataloader/star.py:30, dataloader/tvqa.py:31, dataloader/
  how2qa.py:11. DramaQA/STAR/TVQA 병합본은 split_data.py:53 같은 스크립트로 repo 안에 따로 써 둔 형태다.
