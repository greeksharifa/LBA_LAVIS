import argparse
import json
from pathlib import Path

from subqa.vqa_introspect_data import (
    build_vqa_introspect_records,
    get_vqa_introspect_image_path,
)


def _split_lines(text):
    if text is None:
        return []

    return [line.strip() for line in str(text).splitlines() if line.strip()]


def build_compact_subqa_mapping(results, dedupe=True):
    compact = {}

    for result in results:
        question_id = str(result["question_id"])
        questions = _split_lines(result.get("sub_q_list"))
        answers = _split_lines(result.get("sub_a_list"))

        if len(questions) != len(answers):
            raise ValueError(
                f"Mismatched sub-Q/sub-A counts for question_id={question_id}: "
                f"{len(questions)} questions vs {len(answers)} answers"
            )

        pairs = []
        seen = set()
        for question, answer in zip(questions, answers):
            pair = (question, answer)
            if dedupe and pair in seen:
                continue
            seen.add(pair)
            pairs.append([question, answer])

        compact[question_id] = pairs

    return compact


def build_gt_subqa_pairs(item):
    pairs = []
    seen_questions = set()

    for introspect in item.get("introspect", []):
        for sub_qa in introspect.get("sub_qa", []):
            question = sub_qa["sub_question"]
            if question in seen_questions:
                continue

            seen_questions.add(question)
            pairs.append([question, sub_qa["sub_answer"]])

    return pairs


def build_judge_ready_subqa_mapping(
    json_data,
    generated_subqa,
    vis_root,
    split="val",
    sample_stride=1,
):
    records = build_vqa_introspect_records(json_data, sample_stride=sample_stride)
    valid_question_ids = {record["question_id"] for record in records}
    unknown_question_ids = sorted(set(map(str, generated_subqa)) - valid_question_ids)
    if unknown_question_ids:
        raise ValueError(
            "Generated sub-QA contains unknown question_ids: "
            + ", ".join(unknown_question_ids[:10])
        )

    judge_ready = {}
    for record in records:
        question_id = record["question_id"]
        judge_ready[question_id] = {
            "generated_sub_qa": generated_subqa.get(question_id, []),
            "main_question": record["question"],
            "gt_sub_qa": build_gt_subqa_pairs(json_data[question_id]),
            "image_path": get_vqa_introspect_image_path(
                vis_root, record["image_id"], split
            ),
        }

    return judge_ready


def export_judge_ready_subqa_json(
    ann_path,
    generated_subqa_path,
    output_path,
    vis_root,
    split="val",
    sample_stride=1,
):
    ann_path = Path(ann_path)
    generated_subqa_path = Path(generated_subqa_path)
    output_path = Path(output_path)

    with ann_path.open("r") as f:
        json_data = json.load(f)
    with generated_subqa_path.open("r") as f:
        generated_subqa = json.load(f)

    judge_ready = build_judge_ready_subqa_mapping(
        json_data,
        generated_subqa,
        vis_root=vis_root,
        split=split,
        sample_stride=sample_stride,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(judge_ready, f, indent=4)

    return judge_ready


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export judge-ready VQA-Introspect sub-QA JSON."
    )
    parser.add_argument(
        "--ann-path",
        default="data/VQA-Introspect/VQAIntrospect_valv1.0.json",
        help="Path to the VQA-Introspect annotation JSON.",
    )
    parser.add_argument(
        "--generated-subqa-path",
        required=True,
        help="Path to compact generated sub-QA JSON.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to write the judge-ready JSON.",
    )
    parser.add_argument(
        "--vis-root",
        default="data/coco/images",
        help="COCO image root used by the VQA-Introspect dataloader.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split for image path resolution.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Keep every Nth VQA-Introspect example, matching the dataloader behavior.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_judge_ready_subqa_json(
        ann_path=args.ann_path,
        generated_subqa_path=args.generated_subqa_path,
        output_path=args.output_path,
        vis_root=args.vis_root,
        split=args.split,
        sample_stride=args.sample_stride,
    )


if __name__ == "__main__":
    main()
