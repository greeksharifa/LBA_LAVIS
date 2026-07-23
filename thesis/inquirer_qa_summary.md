# Thesis QA Exports

## JSON Schema

```json
[{"<qid>": {"main_question": "...", "new q": ["..."], "new a": ["..."], "media_path": "/path/or/list"}}]
```

## Created

| Dataset | Model | Output | Sources | Notes |
|---|---|---|---|---|
| DramaQA | INQUIRER | `thesis/DramaQA/INQUIRER/new_qas.json` | `$INQUIRER_WORKSPACE/normalized/dramaqa_kg_scene_normalized.json`, `$INQUIRER_WORKSPACE/normalized/dramaqa_kg_shot_normalized.json` | scene-level은 원본 scene `qid`들에 동일한 새 QA 묶음을 복제했고, shot-level은 shot `vid`별 원본 `qid`에 맞춰 복원해 합쳤습니다. |
| DramaQA | naive | `thesis/DramaQA/naive/new_qas.json` | `$INQUIRER_WORKSPACE/normalized/dramaqa_naive_scene_normalized.json`, `$INQUIRER_WORKSPACE/normalized/dramaqa_naive_shot_normalized.json` | scene-level과 shot-level naive 결과 모두 generated `qid`로 원본 DramaQA `main_question`을 복원해 합쳤습니다. |
| STAR | INQUIRER | `thesis/STAR/INQUIRER/new_qas.json` | `$INQUIRER_WORKSPACE/normalized/star_kg_normalized.json` | STAR INQUIRER add_prob 결과입니다. |
| STAR | naive | `thesis/STAR/naive/new_qas.json` | `$INQUIRER_WORKSPACE/normalized/star_naive_normalized.json` | STAR naive filtered 결과입니다. |
| TVQA | INQUIRER | `thesis/TVQA/INQUIRER/new_qas.json` | `$INQUIRER_WORKSPACE/normalized/tvqa_kg_normalized.json` | TVQA INQUIRER add_prob 결과입니다. q_type 파생본은 같은 모델 변형이라 제외했습니다. |
| TVQA | naive | `thesis/TVQA/naive/new_qas.json` | `$INQUIRER_WORKSPACE/normalized/tvqa_naive_normalized.json` | TVQA naive filtered 결과입니다. |
| How2QA | INQUIRER | `thesis/How2QA/INQUIRER/new_qas.json` | `$INQUIRER_WORKSPACE/normalized/how2qa_kg_filled_normalized.json` | How2QA는 cleaned `results_kg_filled`를 우선 사용하고, 없으면 `kg_prob`를 사용합니다. |
| How2QA | naive | `thesis/How2QA/naive/new_qas.json` | `$INQUIRER_SOURCE_ROOT/gen_how2qa/how2qa_train_naive_prob.json` | How2QA naive prob 결과입니다. |
