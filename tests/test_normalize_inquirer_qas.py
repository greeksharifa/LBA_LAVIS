import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import normalize_inquirer_qas
from tools.normalize_inquirer_qas import (
    HOW2QA_VIDEO_ROOT,
    TVQA_VIDEO_ROOT,
    _tvqa_result_match_key,
    build_dramaqa_generated_scene_records,
    build_dramaqa_generated_shot_records,
    build_dramaqa_qid_records,
    build_how2qa_records,
    build_star_records,
    build_tvqa_block_records,
    build_tvqa_naive_records,
    path_from_env,
)


def test_build_star_records_groups_generated_questions_by_original_qid(monkeypatch):
    monkeypatch.setenv("STAR_VIDEO_ROOT", "fixtures/star/videos")
    configured_root = path_from_env("STAR_VIDEO_ROOT", "data/star/videos")
    monkeypatch.setattr(normalize_inquirer_qas, "STAR_VIDEO_ROOT", configured_root)
    original_items = [
        {
            "question_id": "Interaction_T1_4",
            "video_id": "TJZ0P",
            "question": "Which object was eaten by the person?",
        }
    ]
    generated_items = [
        {
            "question_id": "Interaction_T1_4",
            "video_id": "TJZ0P",
            "question": "What type of object was present in the scene?",
            "answer": "A sandwich.",
        },
        {
            "question_id": "Interaction_T1_4",
            "video_id": "TJZ0P",
            "question": "How many objects were present in the scene?",
            "answer": "Six objects.",
        },
    ]

    records = build_star_records(generated_items, original_items)

    assert records == [
        {
            "key": "Interaction_T1_4",
            "value": {
                "main_question": "Which object was eaten by the person?",
                "new Q": [
                    "What type of object was present in the scene?",
                    "How many objects were present in the scene?",
                ],
                "new A": ["A sandwich.", "Six objects."],
                "image_or_video_path": "fixtures/star/videos/TJZ0P.mp4",
            },
        }
    ]


def test_build_dramaqa_generated_scene_records_replicates_scene_qas_for_original_qids(tmp_path):
    image_root = tmp_path / "AnotherMissOh_images" / "AnotherMissOh01" / "001"
    for shot_name in ["0078", "0159"]:
        shot_dir = image_root / shot_name
        shot_dir.mkdir(parents=True, exist_ok=True)
        (shot_dir / "frame.jpg").write_bytes(b"jpg")

    generated_items = [
        {
            "vid": "AnotherMissOh01_001_0000",
            "que": "What was Deogi doing in the kitchen?",
            "answers": ["Cooking", "Sleeping"],
            "correct_idx": 0,
        },
        {
            "vid": "AnotherMissOh01_001_0000",
            "que": "Who was in the kitchen with Deogi?",
            "answers": ["Haeyoung1", "Dokyung"],
            "correct_idx": 1,
        },
    ]
    original_items = [
        {
            "qid": 1086,
            "vid": "AnotherMissOh01_001_0000",
            "que": "Why was Deogi in the kitchen?",
        },
        {
            "qid": 1087,
            "vid": "AnotherMissOh01_001_0000",
            "que": "Why did Deogi make food a lot?",
        },
    ]

    records = build_dramaqa_generated_scene_records(generated_items, original_items, root_dir=tmp_path)

    assert records == [
        {
            "key": "1086",
            "value": {
                "main_question": "Why was Deogi in the kitchen?",
                "new Q": [
                    "What was Deogi doing in the kitchen?",
                    "Who was in the kitchen with Deogi?",
                ],
                "new A": ["Cooking", "Dokyung"],
                "image_or_video_path": [
                    str(image_root / "0078" / "frame.jpg"),
                    str(image_root / "0159" / "frame.jpg"),
                ],
            },
        },
        {
            "key": "1087",
            "value": {
                "main_question": "Why did Deogi make food a lot?",
                "new Q": [
                    "What was Deogi doing in the kitchen?",
                    "Who was in the kitchen with Deogi?",
                ],
                "new A": ["Cooking", "Dokyung"],
                "image_or_video_path": [
                    str(image_root / "0078" / "frame.jpg"),
                    str(image_root / "0159" / "frame.jpg"),
                ],
            },
        },
    ]


def test_build_dramaqa_generated_shot_records_aligns_to_original_row_order(tmp_path):
    shot_dir = tmp_path / "AnotherMissOh_images" / "AnotherMissOh01" / "001" / "0078"
    shot_dir.mkdir(parents=True, exist_ok=True)
    (shot_dir / "frame.jpg").write_bytes(b"jpg")

    generated_items = [
        {
            "vid": "AnotherMissOh01_001_0078",
            "que": "What style of collar does Haeyoung1's shirt have?",
            "answers": ["Bow tie collar", "Peter Pan collar"],
            "correct_idx": 0,
        },
        {
            "vid": "AnotherMissOh01_001_0078",
            "que": "What type of clothing is Haeyoung1 wearing?",
            "answers": ["A shirt", "A scarf"],
            "correct_idx": 0,
        },
    ]
    original_items = [
        {
            "qid": 995169,
            "vid": "AnotherMissOh01_001_0078",
            "que": "What is the color of Haeyoung1's shirt?",
        },
        {
            "qid": 9947760,
            "vid": "AnotherMissOh01_001_0078",
            "que": "What kind of jewelry is Haeyoung1 wearing?",
        },
    ]

    records = build_dramaqa_generated_shot_records(generated_items, original_items, root_dir=tmp_path)

    assert records == [
        {
            "key": "995169",
            "value": {
                "main_question": "What is the color of Haeyoung1's shirt?",
                "new Q": ["What style of collar does Haeyoung1's shirt have?"],
                "new A": ["Bow tie collar"],
                "image_or_video_path": str(shot_dir / "frame.jpg"),
            },
        },
        {
            "key": "9947760",
            "value": {
                "main_question": "What kind of jewelry is Haeyoung1 wearing?",
                "new Q": ["What type of clothing is Haeyoung1 wearing?"],
                "new A": ["A shirt"],
                "image_or_video_path": str(shot_dir / "frame.jpg"),
            },
        },
    ]


def test_build_dramaqa_qid_records_use_original_question_text(tmp_path):
    shot_dir = tmp_path / "AnotherMissOh_images" / "AnotherMissOh01" / "001" / "0078"
    shot_dir.mkdir(parents=True, exist_ok=True)
    (shot_dir / "frame.jpg").write_bytes(b"jpg")

    generated_items = [
        {
            "qid": 9947760,
            "vid": "AnotherMissOh01_001_0078",
            "que": "What type of clothing is Haeyoung1 wearing?",
            "answers": ["A shirt", "A scarf"],
            "correct_idx": 0,
        }
    ]
    original_items = [
        {
            "qid": 9947760,
            "vid": "AnotherMissOh01_001_0078",
            "que": "What kind of jewelry is Haeyoung1 wearing?",
        }
    ]

    records = build_dramaqa_qid_records(generated_items, original_items, root_dir=tmp_path)

    assert records == [
        {
            "key": "9947760",
            "value": {
                "main_question": "What kind of jewelry is Haeyoung1 wearing?",
                "new Q": ["What type of clothing is Haeyoung1 wearing?"],
                "new A": ["A shirt"],
                "image_or_video_path": str(shot_dir / "frame.jpg"),
            },
        }
    ]


def test_build_tvqa_block_records_aligns_consecutive_blocks_to_original_order():
    original_items = [
        {
            "qid": 0,
            "q": "Where is Meredith when George approaches her?",
            "vid_name": "grey_s03e20_seg02_clip_14",
            "ts": "76.01-84.2",
        },
        {
            "qid": 1,
            "q": "What was in the back of Zoey's van after she opened the doors?",
            "vid_name": "met_s06e05_seg02_clip_09",
            "ts": "45.05-61.29",
        },
    ]
    generated_items = [
        {
            "qid": 122039,
            "q": "Where is George when he expresses worry about Meredith?",
            "answer_idx": 4,
            "a4": "Outside",
            "vid_name": "grey_s03e20_seg02_clip_14",
            "ts": "76.01-84.2",
        },
        {
            "qid": 122040,
            "q": "What does George feel about his fight with Meredith?",
            "answer_idx": 3,
            "a3": "Feels guilty about their fight",
            "vid_name": "grey_s03e20_seg02_clip_14",
            "ts": "76.01-84.2",
        },
        {
            "qid": 122041,
            "q": "What did Zoey liberate from the cosmetics company?",
            "answer_idx": 2,
            "a2": "Bunnies.",
            "vid_name": "met_s06e05_seg02_clip_09",
            "ts": "45.05-61.29",
        },
        {
            "qid": 122042,
            "q": "What cause is Zoey associated with?",
            "answer_idx": 0,
            "a0": "Environmental protection",
            "vid_name": "met_s06e05_seg02_clip_09",
            "ts": "45.05-61.29",
        },
        {
            "qid": 122043,
            "q": "What item is visible in the back of the van?",
            "answer_idx": 1,
            "a1": "A cage.",
            "vid_name": "met_s06e05_seg02_clip_09",
            "ts": "45.05-61.29",
        },
    ]

    records = build_tvqa_block_records(generated_items, original_items)

    assert records == [
        {
            "key": "0",
            "value": {
                "main_question": "Where is Meredith when George approaches her?",
                "new Q": [
                    "Where is George when he expresses worry about Meredith?",
                    "What does George feel about his fight with Meredith?",
                ],
                "new A": ["Outside", "Feels guilty about their fight"],
                "image_or_video_path": str(TVQA_VIDEO_ROOT / "grey_s03e20_seg02_clip_14.mp4"),
            },
        },
        {
            "key": "1",
            "value": {
                "main_question": "What was in the back of Zoey's van after she opened the doors?",
                "new Q": [
                    "What did Zoey liberate from the cosmetics company?",
                    "What cause is Zoey associated with?",
                    "What item is visible in the back of the van?",
                ],
                "new A": ["Bunnies.", "Environmental protection", "A cage."],
                "image_or_video_path": str(TVQA_VIDEO_ROOT / "met_s06e05_seg02_clip_09.mp4"),
            },
        },
    ]


def test_build_tvqa_naive_records_prefers_results_json_exact_matches():
    original_items = [
        {
            "qid": 0,
            "q": "Where is Meredith when George approaches her?",
            "vid_name": "grey_s03e20_seg02_clip_14",
            "ts": "76.01-84.2",
        },
        {
            "qid": 1,
            "q": "What was in the back of Zoey's van after she opened the doors?",
            "vid_name": "met_s06e05_seg02_clip_09",
            "ts": "45.05-61.29",
        },
    ]
    generated_items = [
        {
            "qid": 20,
            "q": "Where does George think Meredith might be?",
            "answer_idx": 0,
            "a0": "At the lab",
            "vid_name": "grey_s03e20_seg02_clip_14",
            "ts": "76.01-84.2",
            "show_name": "Grey's Anatomy",
            "perplex": 0.1,
        },
        {
            "qid": 99,
            "q": "What kind of group did Zoey say she was part of?",
            "answer_idx": 5,
            "a0": "A study group.",
            "a1": "A volunteer group.",
            "a2": "A travel group.",
            "a3": "A sports group.",
            "a4": "A charity group.",
            "vid_name": "met_s06e05_seg02_clip_09",
            "ts": "45.05-61.29",
            "show_name": "How I Met You Mother",
            "perplex": 0.2,
        },
    ]
    results_index = {
        '{"a0": "At the lab", "answer_idx": 0, "q": "Where does George think Meredith might be?", "qid": 20, "show_name": "Grey\'s Anatomy", "ts": "76.01-84.2", "vid_name": "grey_s03e20_seg02_clip_14"}': [
            {"source_key": "0"}
        ],
        '{"a0": "A study group.", "a1": "A volunteer group.", "a2": "A travel group.", "a3": "A sports group.", "a4": "A charity group.", "answer_idx": 5, "q": "What kind of group did Zoey say she was part of?", "qid": 99, "show_name": "How I Met You Mother", "ts": "45.05-61.29", "vid_name": "met_s06e05_seg02_clip_09"}': [
            {
                "source_key": "1",
                "item": {
                    "a0": "A study group.",
                    "a1": "A volunteer group.",
                    "a2": "A travel group.",
                    "a3": "A sports group.",
                    "a4": "A charity group.",
                    "a5": "Bunnies.",
                    "answer_idx": 5,
                    "q": "What kind of group did Zoey say she was part of?",
                    "qid": 99,
                    "show_name": "How I Met You Mother",
                    "ts": "45.05-61.29",
                    "vid_name": "met_s06e05_seg02_clip_09",
                },
            }
        ],
    }
    results_index['{"a0": "At the lab", "answer_idx": 0, "q": "Where does George think Meredith might be?", "qid": 20, "show_name": "Grey\'s Anatomy", "ts": "76.01-84.2", "vid_name": "grey_s03e20_seg02_clip_14"}'][0]["item"] = {
        "a0": "At the lab",
        "answer_idx": 0,
        "q": "Where does George think Meredith might be?",
        "qid": 20,
        "show_name": "Grey's Anatomy",
        "ts": "76.01-84.2",
        "vid_name": "grey_s03e20_seg02_clip_14",
    }

    records = build_tvqa_naive_records(generated_items, original_items, results_index)

    assert records == [
        {
            "key": "0",
            "value": {
                "main_question": "Where is Meredith when George approaches her?",
                "new Q": ["Where does George think Meredith might be?"],
                "new A": ["At the lab"],
                "image_or_video_path": str(TVQA_VIDEO_ROOT / "grey_s03e20_seg02_clip_14.mp4"),
            },
        },
        {
            "key": "1",
            "value": {
                "main_question": "What was in the back of Zoey's van after she opened the doors?",
                "new Q": ["What kind of group did Zoey say she was part of?"],
                "new A": ["Bunnies."],
                "image_or_video_path": str(TVQA_VIDEO_ROOT / "met_s06e05_seg02_clip_09.mp4"),
            },
        },
    ]


def test_build_tvqa_naive_records_recovers_answers_from_query_and_source_fallbacks():
    original_items = [
        {
            "qid": 0,
            "q": "How does House try to take attention off of himself when they question him about his symptoms?",
            "vid_name": "house_s04e08_seg02_clip_19",
            "ts": "39.19-85.19",
            "a0": "Throws his tea at them.",
            "a1": "Reminds them of their dying patient.",
        },
        {
            "qid": 98987,
            "q": "Original Sheldon question",
            "vid_name": "s09e01_seg01_clip_02",
            "ts": "0-5.17",
            "a0": "Wrong 0",
            "a1": "Wrong 1",
            "a2": "Wrong 2",
            "a3": "Wrong 3",
            "a4": "Wrong 4",
        },
    ]
    generated_items = [
        {
            "qid": 49207,
            "q": "What does House do to redirect the conversation when questioned about his own health?",
            "answer_idx": 1,
            "vid_name": "house_s04e08_seg02_clip_19",
            "ts": "39.19-85.19",
            "show_name": "House M.D.",
            "perplex": 0.1,
        },
        {
            "qid": 98989,
            "q": "What emotion does Sheldon express regarding Amy during his conversation?",
            "answer_idx": 5,
            "a0": "There's a spider in the apartment.",
            "a1": "He made a discovery at work.",
            "a2": "The apartment got robbed.",
            "a3": "He took the wrong bus home.",
            "a4": "Leonard is upset about his job.",
            "vid_name": "s09e01_seg01_clip_02",
            "ts": "0-5.17",
            "show_name": "The Big Bang Theory",
            "perplex": 0.2,
        },
    ]
    results_index = {}
    results_by_source_key = {
        "0": {
            "answer_idx": 1,
            "q": "What does House do to redirect the conversation when questioned about his own health?",
            "qid": 49207,
            "show_name": "House M.D.",
            "ts": "39.19-85.19",
            "vid_name": "house_s04e08_seg02_clip_19",
        },
        "98987": {
            "answer_idx": 5,
            "q": "What emotion does Sheldon express regarding Amy during his conversation?",
            "qid": 98989,
            "show_name": "The Big Bang Theory",
            "ts": "0-5.17",
            "vid_name": "s09e01_seg01_clip_02",
            "a5": "Amy gets upset about Leonard's marriage.",
        },
    }
    results_by_query_key = {
        _tvqa_result_match_key(generated_items[0]): [
            {"source_key": "0", "item": results_by_source_key["0"]},
        ],
        _tvqa_result_match_key(generated_items[1]): [
            {"source_key": "98987", "item": results_by_source_key["98987"]},
        ],
    }

    records = build_tvqa_naive_records(
        generated_items,
        original_items,
        results_index,
        results_by_source_key=results_by_source_key,
        results_by_query_key=results_by_query_key,
    )

    assert records == [
        {
            "key": "0",
            "value": {
                "main_question": "How does House try to take attention off of himself when they question him about his symptoms?",
                "new Q": ["What does House do to redirect the conversation when questioned about his own health?"],
                "new A": ["Reminds them of their dying patient."],
                "image_or_video_path": str(TVQA_VIDEO_ROOT / "house_s04e08_seg02_clip_19.mp4"),
            },
        },
        {
            "key": "98987",
            "value": {
                "main_question": "Original Sheldon question",
                "new Q": ["What emotion does Sheldon express regarding Amy during his conversation?"],
                "new A": ["Amy gets upset about Leonard's marriage."],
                "image_or_video_path": str(TVQA_VIDEO_ROOT / "s09e01_seg01_clip_02.mp4"),
            },
        },
    ]


def test_build_how2qa_records_uses_original_qid_mapping():
    original_items = [
        {
            "qid": "10325",
            "video_id": "0WA58hZ21b4_0_60",
            "question": "What is the main part of thevideo?",
            "answer_id": "3",
        }
    ]
    generated_items = [
        {
            "qid": "10325",
            "video_id": "0WA58hZ21b4_0_60",
            "question": "What is the main focus of the video?",
            "answer_id": "0",
            "a0": "Installing blinds",
        },
        {
            "qid": "10325",
            "video_id": "0WA58hZ21b4_0_60",
            "question": "What task is the speaker struggling with?",
            "answer_id": "4",
            "a0": "Painting",
            "a1": "Cleaning",
            "a2": "Cooking",
            "a3": "Driving",
        }
    ]

    records = build_how2qa_records(generated_items, original_items)

    assert records == [
        {
            "key": "10325",
            "value": {
                "main_question": "What is the main part of thevideo?",
                "new Q": [
                    "What is the main focus of the video?",
                    "What task is the speaker struggling with?",
                ],
                "new A": ["Installing blinds", "Driving"],
                "image_or_video_path": str(HOW2QA_VIDEO_ROOT / "0WA58hZ21b4_0_60.mp4"),
            },
        }
    ]


def test_build_how2qa_records_falls_back_to_original_answer_id_when_missing():
    original_items = [
        {
            "qid": "37561",
            "video_id": "OF8eYE-aHIk_60_120",
            "question": "What is a tell?",
            "answer_id": "3",
        }
    ]
    generated_items = [
        {
            "qid": "37561",
            "video_id": "OF8eYE-aHIk_60_120",
            "question": "Who is explaining the steps?",
            "a0": "John",
            "a1": "Bravey",
            "a2": "Thomas",
            "a3": "The man",
        }
    ]

    records = build_how2qa_records(generated_items, original_items)

    assert records == [
        {
            "key": "37561",
            "value": {
                "main_question": "What is a tell?",
                "new Q": ["Who is explaining the steps?"],
                "new A": ["The man"],
                "image_or_video_path": str(HOW2QA_VIDEO_ROOT / "OF8eYE-aHIk_60_120.mp4"),
            },
        }
    ]
