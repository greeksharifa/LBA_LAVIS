import json

from subqa.baseline_single_subq import (
    build_baseline_export_record,
    build_baseline_prompt,
    build_baseline_rank_output_record,
    load_baseline_prompt_config,
    merge_baseline_export_records,
)
from subqa.thesis_export import to_thesis_sub_qas


def test_build_baseline_prompt_uses_verbatim_exemplar_and_single_subq_suffix(tmp_path):
    prompt_path = tmp_path / "baseline_prompts_single_subq.json"
    prompt_path.write_text(
        json.dumps(
            {
                "Baseline_SingleSubQ": {
                    "exemplar": " Context : is the sky blue ? no . are there clouds in the sky ? yes . Question : what weather is likely ? Short answer : rain ",
                    "suffix": "Generate 1 sub-question about image that will help answer the main question. Question : {question}",
                }
            }
        )
    )

    config = load_baseline_prompt_config(prompt_path)

    assert (
        build_baseline_prompt("are they at the beach?", config)
        == " Context : is the sky blue ? no . are there clouds in the sky ? yes . Question : what weather is likely ? Short answer : rain Generate 1 sub-question about image that will help answer the main question. Question : are they at the beach?"
    )


def test_merge_baseline_export_records_sorts_and_strips_source_index():
    merged = merge_baseline_export_records(
        [
            build_baseline_export_record(
                source_index=2,
                question_id="q2",
                main_question="mq2",
                media_path="/tmp/2.jpg",
                sub_question="sq2",
            ),
            build_baseline_export_record(
                source_index=0,
                question_id="q0",
                main_question="mq0",
                media_path="/tmp/0.jpg",
                sub_question="sq0",
            ),
        ]
    )

    assert merged == [
        {
            "question_id": "q0",
            "main_question": "mq0",
            "media_path": "/tmp/0.jpg",
            "sub_q": ["sq0"],
            "sub_a": [],
        },
        {
            "question_id": "q2",
            "main_question": "mq2",
            "media_path": "/tmp/2.jpg",
            "sub_q": ["sq2"],
            "sub_a": [],
        },
    ]


def test_build_baseline_rank_output_record_wraps_single_question():
    record = build_baseline_rank_output_record(
        {
            "source_index": 3,
            "question_id": "q3",
            "main_question": "mq3",
            "image_path": "/tmp/3.jpg",
        },
        "sq3",
    )

    assert record == {
        "source_index": 3,
        "question_id": "q3",
        "main_question": "mq3",
        "image_path": "/tmp/3.jpg",
        "sub-questions": ["sq3"],
    }


def test_baseline_export_records_convert_to_thesis_schema():
    thesis_payload = to_thesis_sub_qas(
        [
            build_baseline_export_record(
                source_index=0,
                question_id="565248001",
                main_question="are they at the beach?",
                media_path="/dataset/565248.jpg",
                sub_question="is there sand?",
            )
        ]
    )

    assert thesis_payload == [
        {
            "565248001": {
                "main_question": "are they at the beach?",
                "sub_q": ["is there sand?"],
                "sub_a": [],
                "media_path": "/dataset/565248.jpg",
            }
        }
    ]
