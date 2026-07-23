from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.normalize_inquirer_qas import (
    DEFAULT_OUTPUT_DIR,
    INQUIRER_SOURCE_ROOT,
    INQUIRER_WORKSPACE,
    REPO_ROOT,
    build_how2qa_records,
    load_items,
    render_path_with_placeholder,
    render_path_with_placeholders,
)


THESIS_ROOT = REPO_ROOT / "thesis"
THESIS_SUMMARY_PATH = THESIS_ROOT / "inquirer_qa_summary.md"
NORMALIZED_DIR = DEFAULT_OUTPUT_DIR


@dataclass(frozen=True)
class ThesisSpec:
    dataset: str
    model: str
    normalized_inputs: tuple[Path, ...] = ()
    notes: str = ""
    special_builder: str | None = None


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def merge_record_sets(record_sets: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for records in record_sets:
        merged.extend(records)
    return merged


def convert_normalized_records_to_thesis_entries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries = []
    for record in records:
        value = record["value"]
        entries.append(
            {
                str(record["key"]): {
                    "main_question": value["main_question"],
                    "new q": value["new Q"],
                    "new a": value["new A"],
                    "media_path": value["image_or_video_path"],
                }
            }
        )
    return entries


def build_how2qa_naive_records() -> list[dict[str, Any]]:
    base = INQUIRER_SOURCE_ROOT / "gen_how2qa"
    generated_items = load_items(base / "how2qa_train_naive_prob.json")
    original_items = load_items(base / "train.json")
    return build_how2qa_records(generated_items, original_items)


def export_specs() -> list[ThesisSpec]:
    how2qa_inquirer_candidates = (
        NORMALIZED_DIR / "how2qa_kg_filled_normalized.json",
        NORMALIZED_DIR / "how2qa_kg_prob_normalized.json",
    )
    how2qa_inquirer_input = tuple(path for path in how2qa_inquirer_candidates if path.exists())[:1]
    return [
        ThesisSpec(
            dataset="DramaQA",
            model="INQUIRER",
            normalized_inputs=(
                NORMALIZED_DIR / "dramaqa_kg_scene_normalized.json",
                NORMALIZED_DIR / "dramaqa_kg_shot_normalized.json",
            ),
            notes="scene-level은 원본 scene `qid`들에 동일한 새 QA 묶음을 복제했고, shot-level은 shot `vid`별 원본 `qid`에 맞춰 복원해 합쳤습니다.",
        ),
        ThesisSpec(
            dataset="DramaQA",
            model="naive",
            normalized_inputs=(
                NORMALIZED_DIR / "dramaqa_naive_scene_normalized.json",
                NORMALIZED_DIR / "dramaqa_naive_shot_normalized.json",
            ),
            notes="scene-level과 shot-level naive 결과 모두 generated `qid`로 원본 DramaQA `main_question`을 복원해 합쳤습니다.",
        ),
        ThesisSpec(
            dataset="STAR",
            model="INQUIRER",
            normalized_inputs=(NORMALIZED_DIR / "star_kg_normalized.json",),
            notes="STAR INQUIRER add_prob 결과입니다.",
        ),
        ThesisSpec(
            dataset="STAR",
            model="naive",
            normalized_inputs=(NORMALIZED_DIR / "star_naive_normalized.json",),
            notes="STAR naive filtered 결과입니다.",
        ),
        ThesisSpec(
            dataset="TVQA",
            model="INQUIRER",
            normalized_inputs=(NORMALIZED_DIR / "tvqa_kg_normalized.json",),
            notes="TVQA INQUIRER add_prob 결과입니다. q_type 파생본은 같은 모델 변형이라 제외했습니다.",
        ),
        ThesisSpec(
            dataset="TVQA",
            model="naive",
            normalized_inputs=(NORMALIZED_DIR / "tvqa_naive_normalized.json",),
            notes="TVQA naive filtered 결과입니다.",
        ),
        ThesisSpec(
            dataset="How2QA",
            model="INQUIRER",
            normalized_inputs=how2qa_inquirer_input,
            notes="How2QA는 cleaned `results_kg_filled`를 우선 사용하고, 없으면 `kg_prob`를 사용합니다.",
        ),
        ThesisSpec(
            dataset="How2QA",
            model="naive",
            special_builder="how2qa_naive",
            notes="How2QA naive prob 결과입니다.",
        ),
    ]


def build_records_for_spec(spec: ThesisSpec) -> list[dict[str, Any]]:
    if spec.special_builder == "how2qa_naive":
        return build_how2qa_naive_records()
    record_sets = [load_json(path) for path in spec.normalized_inputs]
    return merge_record_sets(record_sets)


def spec_is_available(spec: ThesisSpec) -> bool:
    if spec.special_builder == "how2qa_naive":
        return (INQUIRER_SOURCE_ROOT / "gen_how2qa/how2qa_train_naive_prob.json").exists()
    return bool(spec.normalized_inputs) and all(path.exists() for path in spec.normalized_inputs)


def thesis_output_path(dataset: str, model: str) -> Path:
    return THESIS_ROOT / dataset / model / "new_qas.json"


def render_export_path(path: str | Path) -> str:
    return render_path_with_placeholders(
        path,
        (
            (INQUIRER_SOURCE_ROOT, "$INQUIRER_SOURCE_ROOT"),
            (INQUIRER_WORKSPACE, "$INQUIRER_WORKSPACE"),
        ),
    )


def render_export_sources(sources: str | tuple[Path, ...]) -> str:
    if isinstance(sources, str):
        return sources
    return ", ".join(f"`{render_export_path(path)}`" for path in sources)


def build_summary(created: list[dict[str, Any]], skipped: list[dict[str, str]]) -> str:
    lines = [
        "# Thesis QA Exports",
        "",
        "## JSON Schema",
        "",
        "```json",
        '[{"<qid>": {"main_question": "...", "new q": ["..."], "new a": ["..."], "media_path": "/path/or/list"}}]',
        "```",
        "",
        "## Created",
        "",
        "| Dataset | Model | Output | Sources | Notes |",
        "|---|---|---|---|---|",
    ]
    for row in created:
        output_path = render_path_with_placeholder(row["output"], REPO_ROOT, "")
        sources = render_export_sources(row["sources"])
        lines.append(
            f"| {row['dataset']} | {row['model']} | `{output_path}` | {sources} | {row['notes']} |"
        )
    if skipped:
        lines.extend(["", "## Skipped", "", "| Dataset | Model | Reason |", "|---|---|---|"])
        for row in skipped:
            lines.append(f"| {row['dataset']} | {row['model']} | {row['reason']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    THESIS_ROOT.mkdir(parents=True, exist_ok=True)
    created: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for spec in export_specs():
        if not spec_is_available(spec):
            skipped.append(
                {
                    "dataset": spec.dataset,
                    "model": spec.model,
                    "reason": "source files not found",
                }
            )
            continue
        records = build_records_for_spec(spec)
        output_path = thesis_output_path(spec.dataset, spec.model)
        write_json(output_path, convert_normalized_records_to_thesis_entries(records))
        sources = spec.normalized_inputs
        if spec.special_builder == "how2qa_naive":
            sources = (
                INQUIRER_SOURCE_ROOT
                / "gen_how2qa/how2qa_train_naive_prob.json",
            )
        created.append(
            {
                "dataset": spec.dataset,
                "model": spec.model,
                "output": output_path,
                "sources": sources,
                "notes": spec.notes,
            }
        )

    THESIS_SUMMARY_PATH.write_text(build_summary(created, skipped))


if __name__ == "__main__":
    main()
