import json

from subqa.c2r_baseline_single_subq import (
    build_c2r_output_path,
    load_c2r_manifest_records,
    resolve_visual_source,
)


def test_load_c2r_manifest_records_reads_public_thesis_schema(tmp_path):
    source_path = tmp_path / "sub_qas.json"
    source_path.write_text(
        json.dumps(
            [
                {
                    "MMLU_0": {
                        "main_question": "What is algebra?",
                        "sub_q": ["old"],
                        "sub_a": ["old"],
                        "media_path": None,
                    }
                },
                {
                    "validation_Accounting_1": {
                        "main_question": "<image 1> What is shown?",
                        "sub_q": ["old"],
                        "sub_a": ["old"],
                        "media_path": ["/data/MMMU/image1.png", "/data/MMMU/image2.png"],
                    }
                },
            ]
        )
    )

    records = load_c2r_manifest_records(source_path)

    assert records == [
        {
            "source_index": 0,
            "question_id": "MMLU_0",
            "main_question": "What is algebra?",
            "media_path": None,
        },
        {
            "source_index": 1,
            "question_id": "validation_Accounting_1",
            "main_question": "<image 1> What is shown?",
            "media_path": ["/data/MMMU/image1.png", "/data/MMMU/image2.png"],
        },
    ]


def test_resolve_visual_source_uses_dummy_for_text_first_image_for_multi_image_and_video_path_for_video():
    assert resolve_visual_source(None, "/tmp/dummy.png") == ("/tmp/dummy.png", None)
    assert resolve_visual_source(["/tmp/a.png", "/tmp/b.png"], "/tmp/dummy.png") == (
        "/tmp/a.png",
        ["/tmp/a.png", "/tmp/b.png"],
    )
    assert resolve_visual_source("/tmp/video.mp4", "/tmp/dummy.png") == ("/tmp/video.mp4", "/tmp/video.mp4")


def test_build_c2r_output_path_uses_dataset_folder_and_baseline_model_name():
    assert str(build_c2r_output_path("/home/ywjang/C2R", "MMLU-Pro")) == (
        "/home/ywjang/C2R/MMLU-Pro/Baseline-SingleSubQ/sub_qas.json"
    )
