import json

from subqa.questioner_subquestions import (
    build_questioner_prompt,
    build_rank_output_record,
    load_questioner_prompt_config,
    merge_rank_output_records,
    normalize_aokvqa_records,
    normalize_vqa_introspect_records,
)


def test_build_questioner_prompt_without_previous_questions(tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            {
                "Questioner_MultipleSubQ": {
                    "init_prompt": "Generate 1 sub-question about image that will help answer the main question, when main-question is '{}'.",
                    "after_prompt": " Create a question that asks about different information than the following questions. ",
                    "pair_prompt": "sub-question-{}: '{}'",
                }
            }
        )
    )
    config = load_questioner_prompt_config(prompt_path)

    assert (
        build_questioner_prompt("are they at the beach?", [], config)
        == "Generate 1 sub-question about image that will help answer the main question, when main-question is 'are they at the beach?'."
    )


def test_build_questioner_prompt_with_previous_questions(tmp_path):
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps(
            {
                "Questioner_MultipleSubQ": {
                    "init_prompt": "Generate 1 sub-question about image that will help answer the main question, when main-question is '{}'.",
                    "after_prompt": " Create a question that asks about different information than the following questions. ",
                    "pair_prompt": "sub-question-{}: '{}'",
                }
            }
        )
    )
    config = load_questioner_prompt_config(prompt_path)

    assert (
        build_questioner_prompt("are they at the beach?", ["is there sand and whitewater?"], config)
        == "Generate 1 sub-question about image that will help answer the main question, when main-question is 'are they at the beach?'. Create a question that asks about different information than the following questions. sub-question-1: 'is there sand and whitewater?'."
    )
    assert (
        build_questioner_prompt(
            "are they at the beach?",
            ["is there sand and whitewater?", "are the women on horses near the sea?"],
            config,
        )
        == "Generate 1 sub-question about image that will help answer the main question, when main-question is 'are they at the beach?'. Create a question that asks about different information than the following questions. sub-question-1: 'is there sand and whitewater?', sub-question-2: 'are the women on horses near the sea?'."
    )


def test_normalize_vqa_introspect_records_builds_absolute_paths():
    records = normalize_vqa_introspect_records(
        {
            "565248001": {
                "reasoning_question": "are they at the beach?",
                "reasoning_answer_most_common": "yes",
                "image_id": 565248,
                "introspect": [],
            }
        },
        vis_root="/dataset/coco/images",
        split="val",
    )

    assert records == [
        {
            "source_index": 0,
            "question_id": "565248001",
            "main_question": "are they at the beach?",
            "image_path": "/dataset/coco/images/val2014/COCO_val2014_000000565248.jpg",
        }
    ]


def test_normalize_aokvqa_records_uses_annotation_image_field():
    records = normalize_aokvqa_records(
        [
            {
                "question_id": "22jbM6gDxdaMaunuzgrsBB",
                "question": "What is in the motorcyclist's mouth?",
                "image": "val2014/COCO_val2014_000000461751.jpg",
            }
        ],
        vis_root="/dataset/coco/images",
    )

    assert records == [
        {
            "source_index": 0,
            "question_id": "22jbM6gDxdaMaunuzgrsBB",
            "main_question": "What is in the motorcyclist's mouth?",
            "image_path": "/dataset/coco/images/val2014/COCO_val2014_000000461751.jpg",
        }
    ]


def test_merge_rank_output_records_sorts_and_strips_source_index():
    merged = merge_rank_output_records(
        [
            build_rank_output_record(
                {
                    "source_index": 2,
                    "question_id": "q2",
                    "main_question": "mq2",
                    "image_path": "/tmp/2.jpg",
                },
                ["sq2-1", "sq2-2", "sq2-3"],
            ),
            build_rank_output_record(
                {
                    "source_index": 0,
                    "question_id": "q0",
                    "main_question": "mq0",
                    "image_path": "/tmp/0.jpg",
                },
                ["sq0-1", "sq0-2", "sq0-3"],
            ),
        ]
    )

    assert merged == [
        {
            "question_id": "q0",
            "main_question": "mq0",
            "image_path": "/tmp/0.jpg",
            "sub-questions": ["sq0-1", "sq0-2", "sq0-3"],
        },
        {
            "question_id": "q2",
            "main_question": "mq2",
            "image_path": "/tmp/2.jpg",
            "sub-questions": ["sq2-1", "sq2-2", "sq2-3"],
        },
    ]
