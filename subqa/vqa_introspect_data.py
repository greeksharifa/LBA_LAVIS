import os


def build_vqa_introspect_records(json_data, sample_stride=1):
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")

    records = []
    for idx, (question_id, value) in enumerate(json_data.items()):
        if idx % sample_stride != 0:
            continue

        records.append(
            {
                "image_id": value["image_id"],
                "question_id": question_id,
                "question": value["reasoning_question"],
                "answer": value["reasoning_answer_most_common"],
            }
        )

    return records


def get_vqa_introspect_image_path(vis_root, image_id, split):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    image_split = "train2014" if split == "train" else "val2014"
    image_prefix = "COCO_train2014" if split == "train" else "COCO_val2014"

    return os.path.join(
        vis_root,
        image_split,
        f"{image_prefix}_{image_id:012}.jpg",
    )
