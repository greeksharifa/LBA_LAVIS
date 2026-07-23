import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.export_thesis_qas import build_summary, convert_normalized_records_to_thesis_entries, merge_record_sets


def test_convert_normalized_records_to_thesis_entries_uses_requested_keys():
    normalized_records = [
        {
            "key": "123",
            "value": {
                "main_question": "What happened?",
                "new Q": ["Who arrived?"],
                "new A": ["John."],
                "image_or_video_path": "/tmp/example.mp4",
            },
        }
    ]

    assert convert_normalized_records_to_thesis_entries(normalized_records) == [
        {
            "123": {
                "main_question": "What happened?",
                "new q": ["Who arrived?"],
                "new a": ["John."],
                "media_path": "/tmp/example.mp4",
            }
        }
    ]


def test_merge_record_sets_concatenates_multiple_sources_for_one_model():
    record_sets = [
        [{"key": "scene_1", "value": {"main_question": "", "new Q": ["s"], "new A": ["a"], "image_or_video_path": ["i1"]}}],
        [{"key": "shot_1", "value": {"main_question": "", "new Q": ["t"], "new A": ["b"], "image_or_video_path": "i2"}}],
    ]

    assert merge_record_sets(record_sets) == [
        {"key": "scene_1", "value": {"main_question": "", "new Q": ["s"], "new A": ["a"], "image_or_video_path": ["i1"]}},
        {"key": "shot_1", "value": {"main_question": "", "new Q": ["t"], "new A": ["b"], "image_or_video_path": "i2"}},
    ]


def test_build_summary_omits_skipped_section_when_nothing_is_skipped():
    created = [
        {
            "dataset": "DramaQA",
            "model": "INQUIRER",
            "output": "/tmp/new_qas.json",
            "sources": "`/tmp/source.json`",
            "notes": "ok",
        }
    ]

    summary = build_summary(created, [])

    assert "## Skipped" not in summary
    assert "baseline" not in summary
