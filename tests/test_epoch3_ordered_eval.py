import csv
import json

from scripts.run_epoch3_ordered_eval import (
    aggregate_ordered_summary,
    latest_checkpoint_dir,
    parse_devices,
    run_naive,
    run_qar,
)


def test_latest_checkpoint_dir_returns_newest_dir_with_checkpoint_2(tmp_path):
    root = tmp_path / "lavis" / "out"
    old = root / "questioner_small" / "20260101000"
    missing = root / "questioner_small" / "20260103000"
    new = root / "questioner_small" / "20260102000"

    for path in [old, missing, new]:
        path.mkdir(parents=True)
    (old / "checkpoint_2.pth").write_text("old")
    (new / "checkpoint_2.pth").write_text("new")

    assert latest_checkpoint_dir(root, "questioner", "small") == new


def test_aggregate_ordered_summary_preserves_naive_small_medium_large_order(tmp_path):
    root = tmp_path / "lavis" / "out"
    ordered = root / "ordered_eval"
    naive_dir = ordered / "01_naive_base"
    small_dir = ordered / "02_small_ckpt2"
    medium_dir = ordered / "03_medium_ckpt2"
    large_dir = ordered / "04_large_ckpt2"
    for path in [naive_dir, small_dir, medium_dir, large_dir]:
        path.mkdir(parents=True)

    (naive_dir / "summary.json").write_text(
        json.dumps(
            {
                "setting": "naive",
                "checkpoint": "base",
                "num_examples": 7,
                "accuracy": 0.1,
                "model_checkpoint": "/model/base.pth",
            }
        )
    )
    for eval_dir, acc in [(small_dir, 0.2), (medium_dir, 0.3), (large_dir, 0.4)]:
        (eval_dir / "summary_ckpt2.json").write_text(
            json.dumps(
                [
                    {
                        "checkpoint": 2,
                        "num_examples": 7,
                        "accuracy": acc,
                        "questioner_checkpoint": "q.pth",
                        "reasoner_checkpoint": "r.pth",
                        "questioner_naive_main_answer_count": 2,
                        "questioner_naive_main_answer_rate": 2 / 7,
                        "questioner_not_question_count": 1,
                        "questioner_not_question_rate": 1 / 7,
                        "reasoner_sub_answer_instead_count": 3,
                        "reasoner_sub_answer_instead_rate": 0.5,
                        "reasoner_question_form_examples": 7,
                        "reasoner_non_question_excluded_count": 0,
                        "reasoner_eligible_examples": 6,
                        "reasoner_ambiguous_count": 1,
                    }
                ]
            )
        )

    rows = aggregate_ordered_summary(
        root=root,
        ordered_dir=ordered,
        base_checkpoint="/model/base.pth",
        versions=("small", "medium", "large"),
    )

    assert [row["setting"] for row in rows] == ["naive", "small", "medium", "large"]
    assert [row["order"] for row in rows] == [1, 2, 3, 4]
    assert rows[0]["eval_type"] == "naive_instructblip"
    assert rows[1]["checkpoint"] == 2
    assert rows[1]["questioner_naive_main_answer_count"] == 2
    assert rows[1]["questioner_not_question_count"] == 1
    assert rows[1]["reasoner_sub_answer_instead_rate"] == 0.5
    assert rows[1]["reasoner_question_form_examples"] == 7
    assert rows[1]["reasoner_non_question_excluded_count"] == 0
    assert (ordered / "summary_ordered_naive_small_medium_large.json").is_file()
    csv_path = ordered / "summary_ordered_naive_small_medium_large.csv"
    assert csv_path.is_file()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        assert "questioner_naive_main_answer_rate" in reader.fieldnames
        assert "questioner_not_question_rate" in reader.fieldnames
        assert "reasoner_sub_answer_instead_rate" in reader.fieldnames
        assert "reasoner_question_form_examples" in reader.fieldnames
        assert "reasoner_non_question_excluded_count" in reader.fieldnames


def test_checkpoint_dir_for_run_uses_placeholder_in_dry_run(tmp_path):
    from scripts.run_epoch3_ordered_eval import checkpoint_dir_for_run

    root = tmp_path / "lavis" / "out"

    assert checkpoint_dir_for_run(root, "questioner", "medium", dry_run=True) == (
        root / "questioner_medium" / "DRY_RUN_RUN_DIR"
    )


def test_parse_devices_prefers_explicit_multi_gpu_list():
    assert parse_devices("cuda:0", ["cuda:0", "cuda:1"]) == ["cuda:0", "cuda:1"]
    assert parse_devices("cuda:0", None) == ["cuda:0"]


def test_run_commands_pass_multi_gpu_devices_and_custom_ordered_subdir(tmp_path, monkeypatch):
    from argparse import Namespace

    root = tmp_path / "lavis" / "out"
    ordered = root / "ordered_eval_2gpu"
    calls = []

    monkeypatch.setattr(
        "scripts.run_epoch3_ordered_eval.run_cmd",
        lambda cmd, dry_run=False: calls.append([str(part) for part in cmd]),
    )

    args = Namespace(
        python_executable="python",
        cfg_path="cfg.yaml",
        base_checkpoint="/model/base.pth",
        answerer_checkpoint="/model/base.pth",
        llm_model="/model/vicuna",
        ann_val="/data/val.json",
        image_root="/images",
        eval_sample_fraction=0.1,
        eval_sample_seed=42,
        device="cuda:0",
        devices=["cuda:0", "cuda:1"],
        max_len=128,
        min_len=1,
        num_beams=5,
        dry_run=True,
    )

    run_naive(args, ordered)
    run_qar(args, root, ordered, "small", 2)

    assert str(ordered / "01_naive_base") in calls[0]
    assert str(ordered / "02_small_ckpt2") in calls[1]
    assert calls[0][calls[0].index("--devices") + 1 : calls[0].index("--devices") + 3] == [
        "cuda:0",
        "cuda:1",
    ]
    assert calls[1][calls[1].index("--devices") + 1 : calls[1].index("--devices") + 3] == [
        "cuda:0",
        "cuda:1",
    ]
    assert calls[1][calls[1].index("--naive-results") + 1] == str(
        ordered / "01_naive_base" / "results.json"
    )
