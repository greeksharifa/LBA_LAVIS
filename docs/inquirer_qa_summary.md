# INQUIRER QA Files

## 묶어서 본 원본 구조

### DramaQA raw files
- 공통: 원본 DramaQA train은 `qid`, `vid`, `que`, `answers`, `correct_idx`를 가집니다. 생성 파일도 질문 필드는 `question`이 아니라 `que`입니다.
- KG scene-level: generated file은 `qid` 없이 scene `vid`와 새 `que`만 있습니다. prompt가 scene 전체 `scene_qas`에 condition되므로, 정규화에서는 같은 scene의 원본 `qid`마다 동일한 새 QA 묶음을 복제했습니다.
- KG shot-level: generated file은 `qid`가 없지만 shot `vid`별 질문 개수와 내부 순서는 원본과 맞습니다. 정규화에서는 `vid`별 queue alignment로 원본 `qid`와 `que`를 복원했습니다.
- naive scene/shot: generated file 안 `qid`는 원본 DramaQA `qid`이고, 그 안의 `que`는 새로 생성된 질문입니다. 정규화에서는 원본 dataset의 동일 `qid`에서 `main_question`을 복원했습니다.

### STAR raw files
- 공통: JSON list, 각 element는 `question_id`, `video_id`, `start`, `end`, `question`, `answer`, `choices`, `perplex`를 가집니다.
- 차이: KG 파일은 `q_type`과 `situations`가 붙을 수 있고, naive 파일은 보통 더 compact합니다.
- 정규화: `question_id`를 원본 main question id로 보고, 원본 `STAR_train_ori.json`의 `question`을 `main_question`으로 사용했습니다.

### TVQA raw files
- 공통: JSONL, 각 line은 `qid`, `vid_name`, `ts`, `show_name`, `q`, `answer_idx`, `a0..a4`, `perplex`를 가집니다.
- 차이: `tvqa_train_add_prob_with_qtype.jsonl`은 `q_type`이 추가됩니다. KG 계열은 같은 `(vid_name, ts)`가 연속 block으로 붙고, naive는 `chatgpt_result/results.json`에 원 요청 key가 남아 있습니다.
- 정규화: KG 계열은 연속 `(vid_name, ts)` block 순서를 `integrate_oricapkg_train.jsonl` 순서에 맞췄고, naive는 `results.json`의 exact object match를 우선 사용하고 일부 누락만 `qid-1` fallback을 적용했습니다.

### How2QA raw files
- 공통: JSON list, 각 element는 `qid`, `video_id`, `start`, `end`, `question`, `a0..a3`, `answer_id` 중심입니다.
- 차이: `how2qa_train_kg_prob.json`은 `perplex`가 있고, `results_kg_filled.json`은 정규화 후 파일이라 `perplex`가 없습니다. 원본 파일에 `answer_id=4`인데 `a4`가 없는 malformed row가 1건 있습니다.
- 정규화: `gen_how2qa/train.json`의 동일 `qid`를 원본 main question으로 사용했습니다.

## 예시

### DramaQA example
- source: `$INQUIRER_SOURCE_ROOT/prompts/AnotherMissOhQA_train_set_add_sceneprob.json`
```json
{
  "key": "8288",
  "value": {
    "main_question": "Why was Deogi in the kitchen?",
    "new Q": [
      "What was Deogi doing when Haeyoung1 announced she was not getting married?",
      "What was the expected reaction of Deogi and Kyungsu to the news of Haeyoung1's marriage before she announced she wasn't getting married?",
      "What caused Kyungsu to close his eyes and lower his head?",
      "How did Jeongsuk's emotional state compare before and after Haeyoung1's announcement?",
      "What was the immediate consequence of Haeyoung1's announcement on the atmosphere in the house?"
    ],
    "new A": [
      "Deogi was cooking on the floor.",
      "Deogi and Kyungsu were expected to be happy and celebrate Haeyoung1's marriage.",
      "Kyungsu closed his eyes and lowered his head likely due to the shock or sadness from Haeyoung1's announcement that she was not getting married.",
      "Before the announcement, Jeongsuk was happy and optimistic, but after Haeyoung1 said she was not getting married, Jeongsuk was surprised.",
      "The immediate consequence was a shift from a celebratory mood to one of surprise and anger, as indicated by Deogi's and Jeongsuk's reactions."
    ],
    "image_or_video_path": [
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0078/IMAGE_0000004327.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0079/IMAGE_0000004434.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0080/IMAGE_0000004523.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0081/IMAGE_0000004567.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0082/IMAGE_0000004591.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0083/IMAGE_0000004627.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0084/IMAGE_0000004681.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0085/IMAGE_0000004719.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0086/IMAGE_0000004742.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0087/IMAGE_0000004774.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0088/IMAGE_0000004816.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0089/IMAGE_0000004924.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0090/IMAGE_0000005032.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0091/IMAGE_0000005088.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0092/IMAGE_0000005140.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0093/IMAGE_0000005206.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0094/IMAGE_0000005282.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0095/IMAGE_0000005616.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0096/IMAGE_0000006025.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0097/IMAGE_0000006153.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0098/IMAGE_0000006228.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0099/IMAGE_0000006293.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0100/IMAGE_0000006418.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0101/IMAGE_0000006525.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0102/IMAGE_0000006563.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0103/IMAGE_0000006593.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0104/IMAGE_0000006783.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0105/IMAGE_0000007020.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0106/IMAGE_0000007135.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0107/IMAGE_0000007196.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0108/IMAGE_0000007261.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0109/IMAGE_0000007333.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0110/IMAGE_0000007378.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0111/IMAGE_0000007430.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0112/IMAGE_0000007477.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0113/IMAGE_0000007519.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0114/IMAGE_0000007571.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0115/IMAGE_0000007612.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0116/IMAGE_0000007650.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0117/IMAGE_0000007680.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0118/IMAGE_0000007705.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0119/IMAGE_0000007719.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0120/IMAGE_0000007741.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0121/IMAGE_0000007774.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0122/IMAGE_0000007813.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0123/IMAGE_0000007840.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0124/IMAGE_0000007864.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0125/IMAGE_0000007890.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0126/IMAGE_0000007934.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0127/IMAGE_0000007980.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0128/IMAGE_0000008042.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0129/IMAGE_0000008093.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0130/IMAGE_0000008128.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0131/IMAGE_0000008150.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0132/IMAGE_0000008164.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0133/IMAGE_0000008179.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0134/IMAGE_0000008194.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0135/IMAGE_0000008219.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0136/IMAGE_0000008265.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0137/IMAGE_0000008318.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0138/IMAGE_0000008452.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0139/IMAGE_0000008586.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0140/IMAGE_0000008659.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0141/IMAGE_0000008714.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0142/IMAGE_0000008754.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0143/IMAGE_0000008782.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0144/IMAGE_0000008816.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0145/IMAGE_0000008939.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0146/IMAGE_0000009049.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0147/IMAGE_0000009070.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0148/IMAGE_0000009093.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0149/IMAGE_0000009124.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0150/IMAGE_0000009170.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0151/IMAGE_0000009212.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0152/IMAGE_0000009234.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0153/IMAGE_0000009251.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0154/IMAGE_0000009295.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0155/IMAGE_0000009359.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0156/IMAGE_0000009403.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0157/IMAGE_0000009442.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0158/IMAGE_0000009532.jpg",
      "$DRAMAQA_ROOT/AnotherMissOh_images/AnotherMissOh01/001/0159/IMAGE_0000009621.jpg"
    ]
  }
}
```

### STAR example
- source: `$INQUIRER_SOURCE_ROOT/gen_starQA/STAR_train_add_prob.json`
```json
{
  "key": "Feasibility_T2_10",
  "value": {
    "main_question": "What is another action the person can perform with the mirror?",
    "new Q": [
      "What is another action the person can perform with the mirror?"
    ],
    "new A": [
      "Check their reflection."
    ],
    "image_or_video_path": "$STAR_VIDEO_ROOT/AYZS4.mp4"
  }
}
```

### TVQA example
- source: `$INQUIRER_SOURCE_ROOT/gen_tvqa/tvqa_train_add_prob.jsonl`
```json
{
  "key": "0",
  "value": {
    "main_question": "Where is Meredith when George approaches her?",
    "new Q": [
      "Where is George when he expresses worry about Meredith?",
      "What does George feel about his fight with Meredith?"
    ],
    "new A": [
      "Outside",
      "Feels guilty about their fight"
    ],
    "image_or_video_path": "$TVQA_VIDEO_ROOT/grey_s03e20_seg02_clip_14.mp4"
  }
}
```

### How2QA example
- source: `$INQUIRER_SOURCE_ROOT/gen_how2qakg/how2qa_train_kg_prob.json`
```json
{
  "key": "1",
  "value": {
    "main_question": "How many regulators are there in the stove?",
    "new Q": [
      "What ingredient is mentioned as being added gradually?"
    ],
    "new A": [
      "chili powder"
    ],
    "image_or_video_path": "$HOW2QA_VIDEO_ROOT/1cotW8EvpwU_180_240.mp4"
  }
}
```

## 정규화 JSON 경로

| Dataset | Variant | Source | Normalized | Record count | Notes |
|---|---|---|---|---:|---|
| DramaQA | KG scene-level | `$INQUIRER_SOURCE_ROOT/prompts/AnotherMissOhQA_train_set_add_sceneprob.json` | `$INQUIRER_WORKSPACE/normalized/dramaqa_kg_scene_normalized.json` | 5100 | scene prompt가 scene 전체 QA에 condition되므로, 생성된 새 QA 묶음을 해당 scene의 원본 `qid`들에 복제했습니다. |
| DramaQA | KG shot-level | `$INQUIRER_SOURCE_ROOT/prompts/AnotherMissOhQA_train_set_add_shotprob.json` | `$INQUIRER_WORKSPACE/normalized/dramaqa_kg_shot_normalized.json` | 13025 | shot `vid`별 질문 개수/순서를 원본 DramaQA와 맞춰 `qid`와 `main_question`을 복원했습니다. |
| DramaQA | Naive scene-level | `$INQUIRER_SOURCE_ROOT/prompts/AnotherMissOhQA_train_set_naive_sceneprob.json` | `$INQUIRER_WORKSPACE/normalized/dramaqa_naive_scene_normalized.json` | 473 | generated file의 `qid`는 원본 DramaQA `qid`이므로, 원본 `que`를 `main_question`으로 복원했습니다. |
| DramaQA | Naive shot-level | `$INQUIRER_SOURCE_ROOT/prompts/AnotherMissOhQA_train_set_naive_shotprob.json` | `$INQUIRER_WORKSPACE/normalized/dramaqa_naive_shot_normalized.json` | 11054 | generated file의 `qid`는 원본 DramaQA `qid`이므로, 원본 `que`를 `main_question`으로 복원했습니다. |
| STAR | KG | `$INQUIRER_SOURCE_ROOT/gen_starQA/STAR_train_add_prob.json` | `$INQUIRER_WORKSPACE/normalized/star_kg_normalized.json` | 47495 | `question_id` 기준으로 원본 STAR question에 생성 질문들을 묶음. |
| STAR | Naive | `$INQUIRER_SOURCE_ROOT/gen_starQA/STAR_train_naive_prob_filtered.json` | `$INQUIRER_WORKSPACE/normalized/star_naive_normalized.json` | 20738 | `question_id` 기준으로 원본 STAR question에 생성 질문들을 묶음. |
| TVQA | KG | `$INQUIRER_SOURCE_ROOT/gen_tvqa/tvqa_train_add_prob.jsonl` | `$INQUIRER_WORKSPACE/normalized/tvqa_kg_normalized.json` | 121850 | 연속 `(vid_name, ts)` block을 원본 TVQA train 순서에 맞춰 묶음. |
| TVQA | KG + q_type | `$INQUIRER_SOURCE_ROOT/gen_tvqa/tvqa_train_add_prob_with_qtype.jsonl` | `$INQUIRER_WORKSPACE/normalized/tvqa_kg_with_qtype_normalized.json` | 60823 | 연속 `(vid_name, ts)` block을 원본 TVQA train 순서에 맞춰 묶음. q_type은 summary markdown에서만 설명. |
| TVQA | Naive | `$INQUIRER_SOURCE_ROOT/gen_tvqa/tvqa_train_naive_filtered.jsonl` | `$INQUIRER_WORKSPACE/normalized/tvqa_naive_normalized.json` | 119198 | `chatgpt_result/results.json` exact match를 우선 사용해 원본 question id를 복원함. |
| How2QA | KG prob | `$INQUIRER_SOURCE_ROOT/gen_how2qakg/how2qa_train_kg_prob.json` | `$INQUIRER_WORKSPACE/normalized/how2qa_kg_prob_normalized.json` | 24403 | `qid` 기준으로 원본 How2QA train question에 생성 질문들을 묶음. |
| How2QA | KG filled | `$INQUIRER_SOURCE_ROOT/gen_how2qakg/results_kg_filled.json` | `$INQUIRER_WORKSPACE/normalized/how2qa_kg_filled_normalized.json` | 24403 | 정규화/보정 후 파일. `qid` 기준으로 원본 How2QA train question에 생성 질문들을 묶음. |

## Media path 규칙

- DramaQA: `$DRAMAQA_ROOT/AnotherMissOh_images/...` 아래의 대표 frame path를 사용했습니다. scene는 shot별 대표 frame list, shot은 대표 frame 1개입니다.
- STAR: `$STAR_VIDEO_ROOT/{video_id}.mp4`를 사용했습니다.
- TVQA: 존재 확인된 `$TVQA_VIDEO_ROOT/{vid_name}.mp4`를 우선 사용했습니다.
- How2QA: 서버에서 clip root를 직접 확인하지 못해 canonical config path `$HOW2QA_VIDEO_ROOT/{video_id}.mp4`를 사용했습니다.

## 병합본 메모

- repo 안의 `dramaqa_train_*`, `filtered/STAR_*`, `filtered/TVQA_*`는 원본 QA와 새 QA를 합친 학습용 병합본입니다.
- 이번 산출물은 INQUIRER가 만든 raw new-QA 파일을 정규화한 결과입니다. 병합본 자체는 raw source linkage가 일부 손실돼 별도 정규화 대상으로 두지 않았습니다.
