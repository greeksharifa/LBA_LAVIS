from subqa.thesis_export import (
    build_answerer_generation_kwargs,
    build_answerer_rank_output_record,
    merge_answerer_rank_output_records,
    to_thesis_sub_qas,
)


def test_merge_answerer_rank_output_records_sorts_and_strips_source_index():
    merged = merge_answerer_rank_output_records(
        [
            build_answerer_rank_output_record(
                source_index=2,
                question_id="q2",
                main_question="mq2",
                media_path="/tmp/2.jpg",
                sub_questions=["sq2-1", "sq2-2", "sq2-3"],
                sub_answers=["sa2-1", "sa2-2", "sa2-3"],
            ),
            build_answerer_rank_output_record(
                source_index=0,
                question_id="q0",
                main_question="mq0",
                media_path="/tmp/0.jpg",
                sub_questions=["sq0-1", "sq0-2", "sq0-3"],
                sub_answers=["sa0-1", "sa0-2", "sa0-3"],
            ),
        ]
    )

    assert merged == [
        {
            "question_id": "q0",
            "main_question": "mq0",
            "media_path": "/tmp/0.jpg",
            "sub_q": ["sq0-1", "sq0-2", "sq0-3"],
            "sub_a": ["sa0-1", "sa0-2", "sa0-3"],
        },
        {
            "question_id": "q2",
            "main_question": "mq2",
            "media_path": "/tmp/2.jpg",
            "sub_q": ["sq2-1", "sq2-2", "sq2-3"],
            "sub_a": ["sa2-1", "sa2-2", "sa2-3"],
        },
    ]


def test_to_thesis_sub_qas_uses_qid_wrapped_list_elements():
    thesis_payload = to_thesis_sub_qas(
        [
            {
                "question_id": "565248001",
                "main_question": "are they at the beach?",
                "media_path": "/dataset/565248.jpg",
                "sub_q": ["q1", "q2", "q3"],
                "sub_a": ["a1", "a2", "a3"],
            }
        ]
    )

    assert thesis_payload == [
        {
            "565248001": {
                "main_question": "are they at the beach?",
                "sub_q": ["q1", "q2", "q3"],
                "sub_a": ["a1", "a2", "a3"],
                "media_path": "/dataset/565248.jpg",
            }
        }
    ]


def test_build_answerer_generation_kwargs_uses_short_answer_defaults():
    assert build_answerer_generation_kwargs() == {
        "use_nucleus_sampling": False,
        "num_beams": 5,
        "max_length": 10,
        "min_length": 1,
    }
