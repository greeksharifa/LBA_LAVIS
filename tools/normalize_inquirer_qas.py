from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def path_from_env(name: str, default: str) -> Path:
    configured = os.environ.get(name)
    return Path(configured).expanduser() if configured else REPO_ROOT / default


INQUIRER_SOURCE_ROOT = path_from_env("INQUIRER_SOURCE_ROOT", "data/inquirer-source")
INQUIRER_WORKSPACE = path_from_env("INQUIRER_WORKSPACE", "artifacts/inquirer")
DRAMAQA_ROOT = path_from_env("DRAMAQA_ROOT", "data/dramaqa")
STAR_VIDEO_ROOT = path_from_env("STAR_VIDEO_ROOT", "data/star/videos")
TVQA_VIDEO_ROOT = path_from_env("TVQA_VIDEO_ROOT", "data/tvqa/videos")
HOW2QA_VIDEO_ROOT = path_from_env("HOW2QA_VIDEO_ROOT", "data/how2qa/clips")

DEFAULT_OUTPUT_DIR = INQUIRER_WORKSPACE / "normalized"
DEFAULT_SUMMARY_PATH = REPO_ROOT / "docs" / "inquirer_qa_summary.md"
_ITEM_CACHE: dict[Path, list[dict[str, Any]]] = {}
_JSON_CACHE: dict[Path, Any] = {}

DRAMAQA_ORIGINAL_TRAIN_PATH = DRAMAQA_ROOT / "AnotherMissOhQA_train_set_ori_scsh.json"
DRAMAQA_IMAGE_ROOT = DRAMAQA_ROOT / "AnotherMissOh_images"


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    variant: str
    source_path: Path
    output_name: str
    group_kind: str
    notes: str


def load_items(path: Path) -> list[dict[str, Any]]:
    if path in _ITEM_CACHE:
        return _ITEM_CACHE[path]
    if path.suffix == ".jsonl":
        with path.open() as f:
            items = [json.loads(line) for line in f]
    else:
        with path.open() as f:
            items = json.load(f)
    _ITEM_CACHE[path] = items
    return items


def load_json(path: Path) -> Any:
    if path in _JSON_CACHE:
        return _JSON_CACHE[path]
    with path.open() as f:
        value = json.load(f)
    _JSON_CACHE[path] = value
    return value


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def pick_middle_image(dir_path: Path) -> str:
    images = sorted(dir_path.glob("*.jpg"))
    if not images:
        return str(dir_path)
    return str(images[len(images) // 2])


def dramaqa_media_path(vid: str, root_dir: Path = DRAMAQA_ROOT) -> str | list[str]:
    image_root = root_dir / "AnotherMissOh_images"
    if vid.endswith("0000"):
        scene_root = image_root / Path(vid.replace("_", "/"))
        scene_root = Path(str(scene_root)[:-4])
        shot_dirs = sorted([p for p in scene_root.iterdir() if p.is_dir()]) if scene_root.exists() else []
        return [pick_middle_image(shot_dir) for shot_dir in shot_dirs]
    shot_dir = image_root / Path(vid.replace("_", "/"))
    return pick_middle_image(shot_dir)


def star_media_path(video_id: str) -> str:
    return str(STAR_VIDEO_ROOT / f"{video_id}.mp4")


def tvqa_media_path(vid_name: str) -> str:
    return str(TVQA_VIDEO_ROOT / f"{vid_name}.mp4")


def how2qa_media_path(video_id: str) -> str:
    return str(HOW2QA_VIDEO_ROOT / f"{video_id}.mp4")


def correct_answer_from_choices(item: dict[str, Any], answer_key: str, prefix: str) -> str:
    return item[f"{prefix}{item[answer_key]}"]


def _dramaqa_answer_text(item: dict[str, Any]) -> str:
    return item["answers"][item["correct_idx"]]


def build_dramaqa_qid_records(
    generated_items: list[dict[str, Any]],
    original_items: list[dict[str, Any]],
    root_dir: Path = DRAMAQA_ROOT,
) -> list[dict[str, Any]]:
    original_map = {str(item["qid"]): item for item in original_items}
    grouped: dict[str, dict[str, Any]] = {}
    for item in generated_items:
        key = str(item["qid"])
        original = original_map.get(key)
        main_question = original["que"] if original is not None else ""
        media_vid = original["vid"] if original is not None else item["vid"]
        if key not in grouped:
            grouped[key] = {
                "key": key,
                "value": {
                    "main_question": main_question,
                    "new Q": [],
                    "new A": [],
                    "image_or_video_path": dramaqa_media_path(media_vid, root_dir=root_dir),
                },
            }
        grouped[key]["value"]["new Q"].append(item["que"])
        grouped[key]["value"]["new A"].append(_dramaqa_answer_text(item))
    return [grouped[key] for key in sorted(grouped, key=lambda x: int(x))]


def build_dramaqa_generated_shot_records(
    generated_items: list[dict[str, Any]],
    original_items: list[dict[str, Any]],
    root_dir: Path = DRAMAQA_ROOT,
) -> list[dict[str, Any]]:
    original_shot_items = [item for item in original_items if not item["vid"].endswith("0000")]
    if len(generated_items) != len(original_shot_items):
        raise ValueError(
            f"DramaQA shot generated rows ({len(generated_items)}) do not match original shot rows ({len(original_shot_items)})"
        )

    original_by_vid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in original_shot_items:
        original_by_vid[item["vid"]].append(item)

    records = []
    for index, generated in enumerate(generated_items):
        vid = generated["vid"]
        original_candidates = original_by_vid.get(vid)
        if not original_candidates:
            raise ValueError(
                f"DramaQA shot row alignment failed at index {index}: no remaining original question for vid {vid}"
            )
        original = original_candidates.pop(0)
        records.append(
            {
                "key": str(original["qid"]),
                "value": {
                    "main_question": original["que"],
                    "new Q": [generated["que"]],
                    "new A": [_dramaqa_answer_text(generated)],
                    "image_or_video_path": dramaqa_media_path(original["vid"], root_dir=root_dir),
                },
            }
        )
    return records


def build_dramaqa_generated_scene_records(
    generated_items: list[dict[str, Any]],
    original_items: list[dict[str, Any]],
    root_dir: Path = DRAMAQA_ROOT,
) -> list[dict[str, Any]]:
    generated_by_vid: dict[str, dict[str, Any]] = {}
    for item in generated_items:
        vid = item["vid"]
        if vid not in generated_by_vid:
            generated_by_vid[vid] = {
                "new Q": [],
                "new A": [],
                "image_or_video_path": dramaqa_media_path(vid, root_dir=root_dir),
            }
        generated_by_vid[vid]["new Q"].append(item["que"])
        generated_by_vid[vid]["new A"].append(_dramaqa_answer_text(item))

    records = []
    for original in original_items:
        vid = original["vid"]
        if not vid.endswith("0000"):
            continue
        generated_group = generated_by_vid.get(vid)
        if generated_group is None:
            continue
        records.append(
            {
                "key": str(original["qid"]),
                "value": {
                    "main_question": original["que"],
                    "new Q": list(generated_group["new Q"]),
                    "new A": list(generated_group["new A"]),
                    "image_or_video_path": generated_group["image_or_video_path"],
                },
            }
        )
    return records


def build_star_records(generated_items: list[dict[str, Any]], original_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original_map = {item["question_id"]: item for item in original_items}
    grouped: dict[str, dict[str, Any]] = {}
    for item in generated_items:
        key = item["question_id"]
        original = original_map.get(key, item)
        if key not in grouped:
            grouped[key] = {
                "key": key,
                "value": {
                    "main_question": original["question"],
                    "new Q": [],
                    "new A": [],
                    "image_or_video_path": star_media_path(original["video_id"]),
                },
            }
        grouped[key]["value"]["new Q"].append(item["question"])
        grouped[key]["value"]["new A"].append(item["answer"])
    return [grouped[key] for key in sorted(grouped)]


def _tvqa_group_consecutive_items(generated_items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current_group: list[dict[str, Any]] = []
    current_key: tuple[str, str] | None = None
    for item in generated_items:
        item_key = (item["vid_name"], item["ts"])
        if current_key is None or item_key == current_key:
            current_group.append(item)
        else:
            groups.append(current_group)
            current_group = [item]
        current_key = item_key
    if current_group:
        groups.append(current_group)
    return groups


def build_tvqa_block_records(generated_items: list[dict[str, Any]], original_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for original, block in zip(original_items, _tvqa_group_consecutive_items(generated_items)):
        records.append(
            {
                "key": str(original["qid"]),
                "value": {
                    "main_question": original["q"],
                    "new Q": [item["q"] for item in block],
                    "new A": [item[f"a{item['answer_idx']}"] for item in block],
                    "image_or_video_path": tvqa_media_path(original["vid_name"]),
                },
            }
        )
    return records


def _normalized_tvqa_signature(item: dict[str, Any]) -> str:
    normalized = {key: value for key, value in item.items() if key != "perplex"}
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def _tvqa_result_match_key(item: dict[str, Any]) -> str:
    match_fields = {
        key: item.get(key)
        for key in ("qid", "q", "show_name", "ts", "vid_name")
        if key in item
    }
    return json.dumps(match_fields, ensure_ascii=False, sort_keys=True)


def _coerce_tvqa_result_item(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    if "options" in value:
        return None
    return value


def build_tvqa_naive_records(
    generated_items: list[dict[str, Any]],
    original_items: list[dict[str, Any]],
    results_index: dict[str, list[dict[str, Any]]],
    results_by_source_key: dict[str, dict[str, Any]] | None = None,
    results_by_query_key: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    original_map = {str(item["qid"]): item for item in original_items}
    results_by_source_key = results_by_source_key or {}
    results_by_query_key = results_by_query_key or {}
    grouped: dict[str, dict[str, Any]] = {}

    for item in generated_items:
        signature = _normalized_tvqa_signature(item)
        matched_results = results_index.get(signature, [])
        target_key = None
        matched_result_item: dict[str, Any] | None = None
        if matched_results:
            matched = matched_results.pop(0)
            target_key = matched["source_key"]
            matched_result_item = matched["item"]
        else:
            query_key = _tvqa_result_match_key(item)
            query_matches = results_by_query_key.get(query_key, [])
            if query_matches:
                matched = query_matches.pop(0)
                target_key = matched["source_key"]
                matched_result_item = matched["item"]
            else:
                fallback_key = str(item["qid"] - 1)
                if fallback_key in original_map:
                    target_key = fallback_key
                    matched_result_item = results_by_source_key.get(fallback_key)
        if target_key is None:
            continue

        original = original_map.get(target_key)
        if original is None:
            continue
        if target_key not in grouped:
            grouped[target_key] = {
                "key": target_key,
                "value": {
                    "main_question": original["q"],
                    "new Q": [],
                    "new A": [],
                    "image_or_video_path": tvqa_media_path(original["vid_name"]),
                },
            }
        grouped[target_key]["value"]["new Q"].append(item["q"])
        answer_key = f"a{item['answer_idx']}"
        answer_text = item.get(answer_key)
        if answer_text is None and matched_result_item is not None:
            answer_text = matched_result_item.get(answer_key)
        if answer_text is None:
            answer_text = original.get(answer_key, "")
        if answer_text is None:
            answer_text = ""
        grouped[target_key]["value"]["new A"].append(answer_text)
    return [grouped[key] for key in sorted(grouped, key=lambda x: int(x))]


def _resolve_how2qa_answer(item: dict[str, Any], original: dict[str, Any]) -> str:
    answer_idx = item.get("answer_id")
    answer_key = f"a{answer_idx}" if answer_idx is not None else None
    if answer_key is None or answer_key not in item:
        answer_idx = original.get("answer_id")
        answer_key = f"a{answer_idx}" if answer_idx is not None else None
    if answer_key is None:
        return ""
    return item.get(answer_key, original.get(answer_key, ""))


def build_how2qa_records(generated_items: list[dict[str, Any]], original_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    original_map = {str(item["qid"]): item for item in original_items}
    grouped: dict[str, dict[str, Any]] = {}
    for item in generated_items:
        key = str(item["qid"])
        original = original_map.get(key, item)
        if key not in grouped:
            grouped[key] = {
                "key": key,
                "value": {
                    "main_question": original["question"],
                    "new Q": [],
                    "new A": [],
                    "image_or_video_path": how2qa_media_path(original["video_id"]),
                },
            }
        grouped[key]["value"]["new Q"].append(item["question"])
        grouped[key]["value"]["new A"].append(_resolve_how2qa_answer(item, original))
    return [grouped[key] for key in sorted(grouped, key=lambda x: (len(x), x))]


def summarize_example(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, indent=2)


def specs() -> list[SourceSpec]:
    base = INQUIRER_SOURCE_ROOT
    return [
        SourceSpec(
            dataset="DramaQA",
            variant="KG scene-level",
            source_path=base / "prompts/AnotherMissOhQA_train_set_add_sceneprob.json",
            output_name="dramaqa_kg_scene_normalized.json",
            group_kind="dramaqa_scene_generated",
            notes="scene prompt가 scene 전체 QA에 condition되므로, 생성된 새 QA 묶음을 해당 scene의 원본 `qid`들에 복제했습니다.",
        ),
        SourceSpec(
            dataset="DramaQA",
            variant="KG shot-level",
            source_path=base / "prompts/AnotherMissOhQA_train_set_add_shotprob.json",
            output_name="dramaqa_kg_shot_normalized.json",
            group_kind="dramaqa_shot_generated",
            notes="shot `vid`별 질문 개수/순서를 원본 DramaQA와 맞춰 `qid`와 `main_question`을 복원했습니다.",
        ),
        SourceSpec(
            dataset="DramaQA",
            variant="Naive scene-level",
            source_path=base / "prompts/AnotherMissOhQA_train_set_naive_sceneprob.json",
            output_name="dramaqa_naive_scene_normalized.json",
            group_kind="dramaqa_qid",
            notes="generated file의 `qid`는 원본 DramaQA `qid`이므로, 원본 `que`를 `main_question`으로 복원했습니다.",
        ),
        SourceSpec(
            dataset="DramaQA",
            variant="Naive shot-level",
            source_path=base / "prompts/AnotherMissOhQA_train_set_naive_shotprob.json",
            output_name="dramaqa_naive_shot_normalized.json",
            group_kind="dramaqa_qid",
            notes="generated file의 `qid`는 원본 DramaQA `qid`이므로, 원본 `que`를 `main_question`으로 복원했습니다.",
        ),
        SourceSpec(
            dataset="STAR",
            variant="KG",
            source_path=base / "gen_starQA/STAR_train_add_prob.json",
            output_name="star_kg_normalized.json",
            group_kind="star",
            notes="`question_id` 기준으로 원본 STAR question에 생성 질문들을 묶음.",
        ),
        SourceSpec(
            dataset="STAR",
            variant="Naive",
            source_path=base / "gen_starQA/STAR_train_naive_prob_filtered.json",
            output_name="star_naive_normalized.json",
            group_kind="star",
            notes="`question_id` 기준으로 원본 STAR question에 생성 질문들을 묶음.",
        ),
        SourceSpec(
            dataset="TVQA",
            variant="KG",
            source_path=base / "gen_tvqa/tvqa_train_add_prob.jsonl",
            output_name="tvqa_kg_normalized.json",
            group_kind="tvqa",
            notes="연속 `(vid_name, ts)` block을 원본 TVQA train 순서에 맞춰 묶음.",
        ),
        SourceSpec(
            dataset="TVQA",
            variant="KG + q_type",
            source_path=base / "gen_tvqa/tvqa_train_add_prob_with_qtype.jsonl",
            output_name="tvqa_kg_with_qtype_normalized.json",
            group_kind="tvqa",
            notes="연속 `(vid_name, ts)` block을 원본 TVQA train 순서에 맞춰 묶음. q_type은 summary markdown에서만 설명.",
        ),
        SourceSpec(
            dataset="TVQA",
            variant="Naive",
            source_path=base / "gen_tvqa/tvqa_train_naive_filtered.jsonl",
            output_name="tvqa_naive_normalized.json",
            group_kind="tvqa",
            notes="`chatgpt_result/results.json` exact match를 우선 사용해 원본 question id를 복원함.",
        ),
        SourceSpec(
            dataset="How2QA",
            variant="KG prob",
            source_path=base / "gen_how2qakg/how2qa_train_kg_prob.json",
            output_name="how2qa_kg_prob_normalized.json",
            group_kind="how2qa",
            notes="`qid` 기준으로 원본 How2QA train question에 생성 질문들을 묶음.",
        ),
        SourceSpec(
            dataset="How2QA",
            variant="KG filled",
            source_path=base / "gen_how2qakg/results_kg_filled.json",
            output_name="how2qa_kg_filled_normalized.json",
            group_kind="how2qa",
            notes="정규화/보정 후 파일. `qid` 기준으로 원본 How2QA train question에 생성 질문들을 묶음.",
        ),
    ]


def build_records_for_spec(spec: SourceSpec) -> list[dict[str, Any]]:
    items = load_items(spec.source_path)
    base = INQUIRER_SOURCE_ROOT
    if spec.group_kind in {"dramaqa_scene_generated", "dramaqa_shot_generated", "dramaqa_qid"}:
        original_items = load_items(DRAMAQA_ORIGINAL_TRAIN_PATH)
        if spec.group_kind == "dramaqa_scene_generated":
            return build_dramaqa_generated_scene_records(items, original_items)
        if spec.group_kind == "dramaqa_shot_generated":
            return build_dramaqa_generated_shot_records(items, original_items)
        return build_dramaqa_qid_records(items, original_items)
    if spec.group_kind == "star":
        original_items = load_items(base / "gen_starQA/STAR_train_ori.json")
        return build_star_records(items, original_items)
    if spec.group_kind == "tvqa":
        original_items = load_items(base / "gen_tvqa/integrate_oricapkg_train.jsonl")
        if spec.variant == "Naive":
            results_path = base / "gen_tvqa/chatgpt_result/results.json"
            raw_results = load_json(results_path)
            results_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
            results_by_source_key: dict[str, dict[str, Any]] = {}
            results_by_query_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for source_key in sorted(raw_results, key=lambda x: int(x)):
                result_item = _coerce_tvqa_result_item(raw_results[source_key])
                if result_item is None:
                    continue
                results_by_source_key[str(source_key)] = result_item
                results_index[_normalized_tvqa_signature(result_item)].append(
                    {"source_key": source_key, "item": result_item}
                )
                results_by_query_key[_tvqa_result_match_key(result_item)].append(
                    {"source_key": source_key, "item": result_item}
                )
            return build_tvqa_naive_records(
                items,
                original_items,
                results_index,
                results_by_source_key=results_by_source_key,
                results_by_query_key=results_by_query_key,
            )
        return build_tvqa_block_records(items, original_items)
    if spec.group_kind == "how2qa":
        original_items = load_items(base / "gen_how2qa/train.json")
        return build_how2qa_records(items, original_items)
    raise ValueError(f"Unsupported group kind: {spec.group_kind}")


def build_summary(spec_to_records: list[tuple[SourceSpec, list[dict[str, Any]]]]) -> str:
    lines = [
        "# INQUIRER QA Files",
        "",
        "## 묶어서 본 원본 구조",
        "",
        "### DramaQA raw files",
        "- 공통: 원본 DramaQA train은 `qid`, `vid`, `que`, `answers`, `correct_idx`를 가집니다. 생성 파일도 질문 필드는 `question`이 아니라 `que`입니다.",
        "- KG scene-level: generated file은 `qid` 없이 scene `vid`와 새 `que`만 있습니다. prompt가 scene 전체 `scene_qas`에 condition되므로, 정규화에서는 같은 scene의 원본 `qid`마다 동일한 새 QA 묶음을 복제했습니다.",
        "- KG shot-level: generated file은 `qid`가 없지만 shot `vid`별 질문 개수와 내부 순서는 원본과 맞습니다. 정규화에서는 `vid`별 queue alignment로 원본 `qid`와 `que`를 복원했습니다.",
        "- naive scene/shot: generated file 안 `qid`는 원본 DramaQA `qid`이고, 그 안의 `que`는 새로 생성된 질문입니다. 정규화에서는 원본 dataset의 동일 `qid`에서 `main_question`을 복원했습니다.",
        "",
        "### STAR raw files",
        "- 공통: JSON list, 각 element는 `question_id`, `video_id`, `start`, `end`, `question`, `answer`, `choices`, `perplex`를 가집니다.",
        "- 차이: KG 파일은 `q_type`과 `situations`가 붙을 수 있고, naive 파일은 보통 더 compact합니다.",
        "- 정규화: `question_id`를 원본 main question id로 보고, 원본 `STAR_train_ori.json`의 `question`을 `main_question`으로 사용했습니다.",
        "",
        "### TVQA raw files",
        "- 공통: JSONL, 각 line은 `qid`, `vid_name`, `ts`, `show_name`, `q`, `answer_idx`, `a0..a4`, `perplex`를 가집니다.",
        "- 차이: `tvqa_train_add_prob_with_qtype.jsonl`은 `q_type`이 추가됩니다. KG 계열은 같은 `(vid_name, ts)`가 연속 block으로 붙고, naive는 `chatgpt_result/results.json`에 원 요청 key가 남아 있습니다.",
        "- 정규화: KG 계열은 연속 `(vid_name, ts)` block 순서를 `integrate_oricapkg_train.jsonl` 순서에 맞췄고, naive는 `results.json`의 exact object match를 우선 사용하고 일부 누락만 `qid-1` fallback을 적용했습니다.",
        "",
        "### How2QA raw files",
        "- 공통: JSON list, 각 element는 `qid`, `video_id`, `start`, `end`, `question`, `a0..a3`, `answer_id` 중심입니다.",
        "- 차이: `how2qa_train_kg_prob.json`은 `perplex`가 있고, `results_kg_filled.json`은 정규화 후 파일이라 `perplex`가 없습니다. 원본 파일에 `answer_id=4`인데 `a4`가 없는 malformed row가 1건 있습니다.",
        "- 정규화: `gen_how2qa/train.json`의 동일 `qid`를 원본 main question으로 사용했습니다.",
        "",
        "## 예시",
        "",
    ]

    example_specs = {}
    for spec, records in spec_to_records:
        if records:
            example_specs.setdefault(spec.dataset, (spec, records[0]))
    for dataset in ["DramaQA", "STAR", "TVQA", "How2QA"]:
        if dataset in example_specs:
            spec, example = example_specs[dataset]
            lines.append(f"### {dataset} example")
            lines.append(f"- source: `{spec.source_path}`")
            lines.append("```json")
            lines.append(summarize_example(example))
            lines.append("```")
            lines.append("")

    lines.extend(
        [
            "## 정규화 JSON 경로",
            "",
            "| Dataset | Variant | Source | Normalized | Record count | Notes |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for spec, records in spec_to_records:
        normalized_path = DEFAULT_OUTPUT_DIR / spec.output_name
        lines.append(
            f"| {spec.dataset} | {spec.variant} | `{spec.source_path}` | `{normalized_path}` | {len(records)} | {spec.notes} |"
        )

    lines.extend(
        [
            "",
            "## Media path 규칙",
            "",
            "- DramaQA: `$DRAMAQA_ROOT/AnotherMissOh_images/...` 아래의 대표 frame path를 사용합니다. scene는 shot별 대표 frame list, shot은 대표 frame 1개입니다.",
            "- STAR: `$STAR_VIDEO_ROOT/{video_id}.mp4`를 사용합니다.",
            "- TVQA: `$TVQA_VIDEO_ROOT/{vid_name}.mp4`를 사용합니다.",
            "- How2QA: `$HOW2QA_VIDEO_ROOT/{video_id}.mp4`를 사용합니다.",
            "",
            "## 병합본 메모",
            "",
            "- repo 안의 `dramaqa_train_*`, `filtered/STAR_*`, `filtered/TVQA_*`는 원본 QA와 새 QA를 합친 학습용 병합본입니다.",
            "- 이번 산출물은 INQUIRER가 만든 raw new-QA 파일을 정규화한 결과입니다. 병합본 자체는 raw source linkage가 일부 손실돼 별도 정규화 대상으로 두지 않았습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_to_records = []
    for spec in specs():
        records = build_records_for_spec(spec)
        write_json(output_dir / spec.output_name, records)
        spec_to_records.append((spec, records))
    DEFAULT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SUMMARY_PATH.write_text(build_summary(spec_to_records))


if __name__ == "__main__":
    main()
