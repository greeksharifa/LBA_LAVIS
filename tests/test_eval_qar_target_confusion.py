import csv
import json

import pytest

from scripts import eval_qar_checkpoint_sweep as sweep


def test_target_confusion_metrics_join_naive_rows_and_flag_failures():
    qar_rows = [
        {
            "eval_index": 2,
            "sub_question": "The cat!",
            "sub_answer": "blue",
            "gt_answer": "green",
            "pred_answer": "Blue.",
        },
        {
            "eval_index": 5,
            "sub_question": "is the object blue",
            "sub_answer": "yes",
            "gt_answer": "yes",
            "pred_answer": "yes",
        },
        {
            "eval_index": 7,
            "sub_question": "where is the main answer?",
            "sub_answer": "left",
            "gt_answer": "main answer",
            "pred_answer": "main answer",
        },
        {
            "eval_index": 8,
            "sub_question": "what color is it?",
            "sub_answer": "blue",
            "gt_answer": "green",
            "pred_answer": "Blue.",
        },
    ]
    naive_rows = [
        {"eval_index": 5, "pred_answer": "not two"},
        {"eval_index": 2, "pred_answer": "cat"},
        {"eval_index": 7, "pred_answer": "different"},
        {"eval_index": 8, "pred_answer": "different"},
    ]

    metrics = sweep.apply_target_confusion_metrics(qar_rows, naive_rows=naive_rows)

    assert qar_rows[0]["questioner_pred_naive_main_answer"] == 1
    assert qar_rows[1]["questioner_pred_naive_main_answer"] == 0
    assert qar_rows[2]["questioner_pred_naive_main_answer"] == 0
    assert qar_rows[3]["questioner_pred_naive_main_answer"] == 0

    assert qar_rows[0]["questioner_output_is_question"] == 0
    assert qar_rows[1]["questioner_output_is_question"] == 1
    assert qar_rows[2]["questioner_output_is_question"] == 1
    assert qar_rows[3]["questioner_output_is_question"] == 1
    assert qar_rows[0]["questioner_not_question"] == 1
    assert qar_rows[1]["questioner_not_question"] == 0
    assert qar_rows[2]["questioner_not_question"] == 0
    assert qar_rows[3]["questioner_not_question"] == 0

    assert qar_rows[0]["reasoner_question_form_candidate"] == 0
    assert qar_rows[0]["reasoner_pred_sub_answer_instead_of_main"] == 0
    assert qar_rows[0]["reasoner_target_ambiguous"] == 0
    assert qar_rows[1]["reasoner_question_form_candidate"] == 1
    assert qar_rows[1]["reasoner_pred_sub_answer_instead_of_main"] == 0
    assert qar_rows[1]["reasoner_target_ambiguous"] == 1
    assert qar_rows[2]["reasoner_question_form_candidate"] == 1
    assert qar_rows[2]["reasoner_pred_sub_answer_instead_of_main"] == 0
    assert qar_rows[2]["reasoner_target_ambiguous"] == 0
    assert qar_rows[3]["reasoner_question_form_candidate"] == 1
    assert qar_rows[3]["reasoner_pred_sub_answer_instead_of_main"] == 1
    assert qar_rows[3]["reasoner_target_ambiguous"] == 0

    assert metrics["questioner_naive_main_answer_count"] == 1
    assert metrics["questioner_naive_main_answer_rate"] == pytest.approx(1 / 4)
    assert metrics["questioner_not_question_count"] == 1
    assert metrics["questioner_not_question_rate"] == pytest.approx(1 / 4)
    assert metrics["reasoner_sub_answer_instead_count"] == 1
    assert metrics["reasoner_sub_answer_instead_rate"] == pytest.approx(1 / 2)
    assert metrics["reasoner_question_form_examples"] == 3
    assert metrics["reasoner_non_question_excluded_count"] == 1
    assert metrics["reasoner_eligible_examples"] == 2
    assert metrics["reasoner_ambiguous_count"] == 1

def test_question_form_detection_accepts_question_mark_and_question_stems():
    assert sweep.is_question_form("What color is the bus")
    assert sweep.is_question_form("is the light on.")
    assert sweep.is_question_form("there is a bus?")
    assert not sweep.is_question_form("blue")
    assert not sweep.is_question_form("")


def test_target_confusion_requires_matching_naive_eval_indices():
    qar_rows = [
        {
            "eval_index": 10,
            "sub_question": "red",
            "sub_answer": "blue",
            "gt_answer": "green",
            "pred_answer": "blue",
        }
    ]
    naive_rows = [{"eval_index": 9, "pred_answer": "red"}]

    with pytest.raises(ValueError, match="missing naive eval_index"):
        sweep.apply_target_confusion_metrics(qar_rows, naive_rows=naive_rows)


def test_summarize_checkpoint_writes_target_confusion_json_and_csv(tmp_path):
    rows = [
        {
            "eval_index": 0,
            "sub_question": "The cat",
            "sub_answer": "blue",
            "gt_answer": "green",
            "pred_answer": "blue",
            "exact_match": 0,
        }
    ]
    result_file = tmp_path / "checkpoint_2_results.json"
    result_file.write_text(json.dumps(rows))

    summary = []
    sweep.summarize_checkpoint(
        tmp_path,
        "summary",
        summary,
        2,
        "q.pth",
        "r.pth",
        result_file,
        0.0,
        naive_rows=[{"eval_index": 0, "pred_answer": "cat"}],
    )

    written = json.loads((tmp_path / "summary.json").read_text())
    assert written[0]["questioner_naive_main_answer_count"] == 1
    assert written[0]["questioner_naive_main_answer_rate"] == 1.0
    assert written[0]["questioner_not_question_count"] == 1
    assert written[0]["questioner_not_question_rate"] == 1.0
    assert written[0]["reasoner_sub_answer_instead_count"] == 0
    assert written[0]["reasoner_sub_answer_instead_rate"] == 0.0
    assert written[0]["reasoner_question_form_examples"] == 0
    assert written[0]["reasoner_non_question_excluded_count"] == 1
    assert written[0]["reasoner_eligible_examples"] == 0
    assert written[0]["reasoner_ambiguous_count"] == 0

    with open(tmp_path / "summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert "questioner_naive_main_answer_rate" in reader.fieldnames
        assert "questioner_not_question_rate" in reader.fieldnames
        assert "reasoner_sub_answer_instead_rate" in reader.fieldnames
        assert "reasoner_question_form_examples" in reader.fieldnames
        assert "reasoner_non_question_excluded_count" in reader.fieldnames
        csv_rows = list(reader)
    assert csv_rows[0]["questioner_naive_main_answer_count"] == "1"
