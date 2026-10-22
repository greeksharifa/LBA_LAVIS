import json

import pytest

from subqa.vqa_introspect_data import (
    build_vqa_introspect_records,
    get_vqa_introspect_image_path,
)
from subqa.vqa_introspect_export import (
    build_compact_subqa_mapping,
    build_gt_subqa_pairs,
    build_judge_ready_subqa_mapping,
    export_judge_ready_subqa_json,
)


def test_build_vqa_introspect_records_keeps_all_questions_by_default():
    json_data = {
        "101": {
            "image_id": 1,
            "reasoning_question": "q1",
            "reasoning_answer_most_common": "a1",
        },
        "102": {
            "image_id": 2,
            "reasoning_question": "q2",
            "reasoning_answer_most_common": "a2",
        },
        "103": {
            "image_id": 3,
            "reasoning_question": "q3",
            "reasoning_answer_most_common": "a3",
        },
    }

    records = build_vqa_introspect_records(json_data)

    assert [record["question_id"] for record in records] == ["101", "102", "103"]
    assert [record["answer"] for record in records] == ["a1", "a2", "a3"]


def test_build_vqa_introspect_records_applies_sample_stride():
    json_data = {
        str(100 + idx): {
            "image_id": idx,
            "reasoning_question": f"q{idx}",
            "reasoning_answer_most_common": f"a{idx}",
        }
        for idx in range(5)
    }

    records = build_vqa_introspect_records(json_data, sample_stride=2)

    assert [record["question_id"] for record in records] == ["100", "102", "104"]


def test_get_vqa_introspect_image_path_uses_train2014_for_train_split():
    image_path = get_vqa_introspect_image_path("/dataset/coco/images", 265814, "train")

    assert image_path == "/dataset/coco/images/train2014/COCO_train2014_000000265814.jpg"


def test_get_vqa_introspect_image_path_uses_val2014_for_val_split():
    image_path = get_vqa_introspect_image_path("/dataset/coco/images", 565248, "val")

    assert image_path == "/dataset/coco/images/val2014/COCO_val2014_000000565248.jpg"


def test_build_compact_subqa_mapping_zips_lines_and_deduplicates():
    results = [
        {
            "question_id": "565248001",
            "sub_q_list": "is there sand?\nis there sand?\nare they riding horses?",
            "sub_a_list": "yes\nyes\nyes",
        },
        {
            "question_id": "20906001",
            "sub_q_list": "is it a sandwich?",
            "sub_a_list": "yes",
        },
    ]

    compact = build_compact_subqa_mapping(results)

    assert compact == {
        "565248001": [["is there sand?", "yes"], ["are they riding horses?", "yes"]],
        "20906001": [["is it a sandwich?", "yes"]],
    }


def test_build_compact_subqa_mapping_rejects_mismatched_line_counts():
    results = [
        {
            "question_id": "565248001",
            "sub_q_list": "is there sand?\nare they riding horses?",
            "sub_a_list": "yes",
        }
    ]

    with pytest.raises(ValueError, match="565248001"):
        build_compact_subqa_mapping(results)



def test_build_gt_subqa_pairs_deduplicates_by_question_text():
    item = {
        "introspect": [
            {
                "sub_qa": [
                    {"sub_question": "is there a fence?", "sub_answer": "yes"},
                    {
                        "sub_question": "what surrounds the plane?",
                        "sub_answer": "walls",
                    },
                ]
            },
            {
                "sub_qa": [
                    {"sub_question": "is there a fence?", "sub_answer": "no"},
                ]
            },
        ]
    }

    assert build_gt_subqa_pairs(item) == [
        ["is there a fence?", "yes"],
        ["what surrounds the plane?", "walls"],
    ]


def test_build_judge_ready_subqa_mapping_merges_generated_gt_and_image_paths():
    json_data = {
        "565248001": {
            "image_id": 565248,
            "reasoning_question": "are they at the beach?",
            "reasoning_answer_most_common": "yes",
            "introspect": [
                {
                    "sub_qa": [
                        {
                            "sub_question": "are the women on horses near the sea?",
                            "sub_answer": "yes",
                        },
                        {
                            "sub_question": "is there sand and whitewater?",
                            "sub_answer": "yes",
                        },
                    ]
                }
            ],
        },
        "20906001": {
            "image_id": 20906,
            "reasoning_question": "which food is this?",
            "reasoning_answer_most_common": "sandwich",
            "introspect": [
                {
                    "sub_qa": [
                        {
                            "sub_question": "is the food a sandwich?",
                            "sub_answer": "yes",
                        }
                    ]
                }
            ],
        },
    }
    generated_subqa = {
        "565248001": [["are the women on horses near the sea?", "yes"]],
    }

    judge_ready = build_judge_ready_subqa_mapping(
        json_data,
        generated_subqa,
        vis_root="/dataset/coco/images",
        split="val",
    )

    assert judge_ready == {
        "565248001": {
            "generated_sub_qa": [["are the women on horses near the sea?", "yes"]],
            "main_question": "are they at the beach?",
            "gt_sub_qa": [
                ["are the women on horses near the sea?", "yes"],
                ["is there sand and whitewater?", "yes"],
            ],
            "image_path": "/dataset/coco/images/val2014/COCO_val2014_000000565248.jpg",
        },
        "20906001": {
            "generated_sub_qa": [],
            "main_question": "which food is this?",
            "gt_sub_qa": [["is the food a sandwich?", "yes"]],
            "image_path": "/dataset/coco/images/val2014/COCO_val2014_000000020906.jpg",
        },
    }


def test_build_judge_ready_subqa_mapping_rejects_unknown_generated_question_ids():
    json_data = {
        "565248001": {
            "image_id": 565248,
            "reasoning_question": "are they at the beach?",
            "reasoning_answer_most_common": "yes",
            "introspect": [],
        }
    }

    with pytest.raises(ValueError, match="999"):
        build_judge_ready_subqa_mapping(
            json_data,
            {"999": [["is there sand?", "yes"]]},
            vis_root="/dataset/coco/images",
            split="val",
        )


def test_export_judge_ready_subqa_json_writes_file(tmp_path):
    ann_path = tmp_path / "ann.json"
    generated_path = tmp_path / "generated.json"
    output_path = tmp_path / "judge_ready.json"

    ann_path.write_text(
        json.dumps(
            {
                "565248001": {
                    "image_id": 565248,
                    "reasoning_question": "are they at the beach?",
                    "reasoning_answer_most_common": "yes",
                    "introspect": [
                        {
                            "sub_qa": [
                                {
                                    "sub_question": "is there sand and whitewater?",
                                    "sub_answer": "yes",
                                }
                            ]
                        }
                    ],
                }
            }
        )
    )
    generated_path.write_text(
        json.dumps({"565248001": [["is there sand and whitewater?", "yes"]]})
    )

    export_judge_ready_subqa_json(
        ann_path=ann_path,
        generated_subqa_path=generated_path,
        output_path=output_path,
        vis_root="/dataset/coco/images",
        split="val",
    )

    assert json.loads(output_path.read_text()) == {
        "565248001": {
            "generated_sub_qa": [["is there sand and whitewater?", "yes"]],
            "main_question": "are they at the beach?",
            "gt_sub_qa": [["is there sand and whitewater?", "yes"]],
            "image_path": "/dataset/coco/images/val2014/COCO_val2014_000000565248.jpg",
        }
    }


def test_build_vqa_introspect_records_samples_deterministically():
    annotations = {
        str(index): {"image_id": index, "reasoning_question": "q", "reasoning_answer_most_common": "a"}
        for index in range(20)
    }
    rows = build_vqa_introspect_records(annotations, sample_fraction=0.25, sample_seed=42)
    assert len(rows) == 5
    assert rows == build_vqa_introspect_records(annotations, sample_fraction=0.25, sample_seed=42)
    assert rows != build_vqa_introspect_records(annotations, sample_fraction=0.25, sample_seed=43)


@pytest.mark.parametrize("fraction", [0, -1, 1.1])
def test_build_vqa_introspect_records_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError, match="sample_fraction"):
        build_vqa_introspect_records({}, sample_fraction=fraction)
