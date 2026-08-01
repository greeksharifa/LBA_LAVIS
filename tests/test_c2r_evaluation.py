import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.c2r import (
    TAU1_GRID,
    TAU2_GRID,
    apply_thresholds,
    evaluate_run_pair,
    load_run,
    normalize_confidence,
    paired_bootstrap_delta,
    prepare_samples,
    search_thresholds,
    select_refined_candidate,
)


def exact_scorer(prediction, gold, question_type):
    del question_type
    return prediction == gold


def record(
    qid,
    *,
    split="dev",
    gold="base",
    base="base",
    base_conf=0.5,
    refined=("wrong", "refined"),
    refined_conf=(0.2, 0.8),
    confidence_type="token_min_prob",
):
    return {
        "qid": qid,
        "split": split,
        "main_q": f"question {qid}",
        "gt_ans": gold,
        "question_type": "open_ended",
        "base_answer": base,
        "conf_base": {confidence_type: base_conf},
        "refined_answer_list": list(refined),
        "conf_refined": {confidence_type: list(refined_conf)},
    }


def manifest(split, qids, *, confidence_type="token_min_prob", model_id="model/id"):
    return {
        "config": {
            "dataset": "MMMU",
            "split": split,
            "model_name": "fixture-model",
            "model_id": model_id,
            "N": 5,
            "M": 2,
            "K": 8,
            "confidence_type": confidence_type,
            "annotation_paths": [f"MMMU/{split}.json"],
            "annotation_paths_resolved": [f"/data/MMMU/{split}.json"],
            "num_data": -1,
        },
        "qids": list(qids),
        "stages": {"refined": {"completed": True, "state": "completed"}},
    }


def write_run(root, name, split, records, **manifest_kwargs):
    run_dir = root / name
    run_dir.mkdir()
    qids = [item["qid"] for item in records]
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest(split, qids, **manifest_kwargs)), encoding="utf-8"
    )
    (run_dir / "refined_samples.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    return run_dir


class ConfidenceTests(unittest.TestCase):
    def test_token_min_prob_higher_candidate_wins(self):
        answer, confidence, index = select_refined_candidate(
            ["low", "high"], [0.1, 0.9], "token_min_prob"
        )
        self.assertEqual(("high", 0.9, 1), (answer, confidence, index))

    def test_seq_ppl_uses_inverse_clamped_confidence_for_candidates_and_base(self):
        answer, confidence, index = select_refined_candidate(
            ["large-ppl", "zero-ppl"], [1e300, 0.0], "seq_ppl"
        )
        self.assertEqual("zero-ppl", answer)
        self.assertEqual(1, index)
        self.assertEqual(1.0, confidence)
        self.assertEqual(1.0, normalize_confidence(0.0, "seq_ppl"))
        self.assertEqual(1.0, normalize_confidence(1e-300, "seq_ppl"))
        self.assertEqual(1e-300, normalize_confidence(1e300, "seq_ppl"))

        samples = prepare_samples(
            [
                record(
                    "q0",
                    gold="refined",
                    base_conf=4.0,
                    refined=("refined",),
                    refined_conf=(2.0,),
                    confidence_type="seq_ppl",
                )
            ],
            "seq_ppl",
            scorer=exact_scorer,
        )
        result = apply_thresholds(samples, tau1=0.3, tau2=0.2)
        self.assertEqual(0.25, samples[0]["base_confidence"])
        self.assertEqual(0.5, samples[0]["refined_confidence"])
        self.assertEqual(1, result["switch_count"])
        self.assertEqual(1, result["correct_count"])


class ThresholdTests(unittest.TestCase):
    def test_gate_switches_at_exact_decimal_margin_boundary(self):
        samples = prepare_samples(
            [
                record(
                    "boundary",
                    gold="refined",
                    base_conf=0.4,
                    refined=("refined",),
                    refined_conf=(0.6,),
                )
            ],
            "token_min_prob",
            scorer=exact_scorer,
        )

        result = apply_thresholds(samples, tau1=0.8, tau2=0.2)

        self.assertEqual([True], result["switched"])

    def test_gate_rejects_value_strictly_below_decimal_margin_boundary(self):
        samples = prepare_samples(
            [
                record(
                    "below-boundary",
                    gold="base",
                    base_conf=0.4,
                    refined=("refined",),
                    refined_conf=(0.5999999999999,),
                )
            ],
            "token_min_prob",
            scorer=exact_scorer,
        )

        result = apply_thresholds(samples, tau1=0.8, tau2=0.2)

        self.assertEqual([False], result["switched"])

    def test_gate_rule_obeys_base_short_circuit_and_refined_margin(self):
        samples = prepare_samples(
            [
                record("high-base", gold="base", base_conf=0.8, refined_conf=(0.1, 1.0)),
                record("margin", gold="refined", base_conf=0.4, refined_conf=(0.1, 0.6)),
                record("fallback", gold="base", base_conf=0.4, refined_conf=(0.1, 0.59)),
            ],
            "token_min_prob",
            scorer=exact_scorer,
        )

        result = apply_thresholds(samples, tau1=0.8, tau2=0.2)

        self.assertEqual([False, True, False], result["switched"])
        self.assertEqual(["base", "refined", "base"], result["answers"])

    def test_fixed_grid_and_ties_prefer_switches_then_low_tau1_then_high_tau2(self):
        self.assertEqual(tuple(round(i / 10, 1) for i in range(11)), TAU1_GRID)
        self.assertEqual(
            tuple(round(-1 + i / 10, 1) for i in range(21)), TAU2_GRID
        )
        all_base_correct = prepare_samples(
            [record("q0", gold="base", base_conf=0.5, refined_conf=(0.1, 0.9))],
            "token_min_prob",
            scorer=exact_scorer,
        )

        selected = search_thresholds(all_base_correct)

        self.assertEqual(1, selected["correct_count"])
        self.assertEqual(0, selected["switch_count"])
        self.assertEqual(0.0, selected["tau1"])
        self.assertEqual(1.0, selected["tau2"])

    def test_validation_uses_dev_threshold_even_when_validation_oracle_differs(self):
        dev = prepare_samples(
            [record("dev-q", gold="base", base_conf=0.5, refined_conf=(0.1, 0.9))],
            "token_min_prob",
            scorer=exact_scorer,
        )
        validation = prepare_samples(
            [
                record(
                    "val-q",
                    split="validation",
                    gold="refined",
                    base_conf=0.5,
                    refined_conf=(0.1, 0.9),
                )
            ],
            "token_min_prob",
            scorer=exact_scorer,
        )

        selected = search_thresholds(dev)
        validation_result = apply_thresholds(
            validation, selected["tau1"], selected["tau2"]
        )
        validation_oracle = search_thresholds(validation)

        self.assertEqual((0.0, 1.0), (selected["tau1"], selected["tau2"]))
        self.assertEqual(0, validation_result["correct_count"])
        self.assertEqual(1, validation_oracle["correct_count"])


class BootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_is_seeded_and_reports_percentile_ci(self):
        base = [1, 0, 1, 0]
        gated = [1, 1, 0, 1]

        first = paired_bootstrap_delta(base, gated, seed=42, count=10_000)
        second = paired_bootstrap_delta(base, gated, seed=42, count=10_000)

        self.assertEqual(first, second)
        self.assertEqual(0.25, first["delta"])
        self.assertEqual(42, first["seed"])
        self.assertEqual(10_000, first["count"])
        self.assertEqual(2, len(first["percentile_95_ci"]))


class RunLoadingTests(unittest.TestCase):
    def test_loader_rejects_malformed_confidence_schema_before_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed = record("q0")
            malformed["conf_refined"] = {"token_min_prob": [0.2]}
            run_dir = write_run(root, "run", "dev", [malformed])

            with self.assertRaisesRegex(ValueError, "count mismatch"):
                load_run(run_dir)

    def test_loader_accepts_selected_scalar_base_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scalar = record("q0")
            scalar["conf_base"] = 0.5
            run_dir = write_run(root, "run", "dev", [scalar])

            loaded = load_run(run_dir)

            self.assertEqual(["q0"], loaded["qids"])

    def test_loader_requires_completed_refined_stage_unique_exact_qids_and_split(self):
        cases = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good_records = [record("q0")]
            run_dir = write_run(root, "run", "dev", good_records)
            loaded = load_run(run_dir)
            self.assertEqual(["q0"], loaded["qids"])

            bad_stage = json.loads((run_dir / "run_manifest.json").read_text())
            bad_stage["stages"]["refined"]["state"] = "running"
            cases.append(("stage", bad_stage, good_records, "not completed"))
            cases.append(
                (
                    "duplicate",
                    manifest("dev", ["q0", "q0"]),
                    [record("q0"), record("q0")],
                    "duplicate qid",
                )
            )
            cases.append(
                (
                    "qid",
                    manifest("dev", ["q0"]),
                    [record("q1")],
                    "qid mismatch",
                )
            )
            cases.append(
                (
                    "split",
                    manifest("dev", ["q0"]),
                    [record("q0", split="validation")],
                    "split mismatch",
                )
            )

            for name, bad_manifest, bad_records, message in cases:
                case_dir = root / name
                case_dir.mkdir()
                (case_dir / "run_manifest.json").write_text(json.dumps(bad_manifest))
                (case_dir / "refined_samples.json").write_text(json.dumps(bad_records))
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, message):
                        load_run(case_dir)

    def test_pair_compatibility_fails_before_scorer_is_called(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_dir = write_run(root, "dev", "dev", [record("d")])
            val_dir = write_run(
                root,
                "validation",
                "validation",
                [record("v", split="validation")],
                model_id="other/id",
            )
            calls = []

            with self.assertRaisesRegex(ValueError, "incompatible run manifests.*model_id"):
                evaluate_run_pair(
                    dev_dir,
                    val_dir,
                    scorer=lambda *args: calls.append(args),
                    bootstrap_count=10,
                )
            self.assertEqual([], calls)

    def test_report_contains_metrics_threshold_source_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_dir = write_run(root, "dev", "dev", [record("d")])
            val_dir = write_run(
                root,
                "validation",
                "validation",
                [record("v", split="validation")],
            )

            report = evaluate_run_pair(
                dev_dir,
                val_dir,
                scorer=exact_scorer,
                bootstrap_count=100,
            )

            self.assertEqual("dev", report["threshold_source"]["split"])
            self.assertEqual(42, report["bootstrap_seed"])
            self.assertEqual(100, report["bootstrap_count"])
            for split_name in ("dev", "validation"):
                split_report = report[split_name]
                self.assertIn("base_accuracy", split_report)
                self.assertIn("raw_refined_accuracy", split_report)
                self.assertIn("gated_accuracy", split_report)
                self.assertIn("absolute_delta", split_report)
                self.assertIn("relative_delta", split_report)
                self.assertIn("paired_95_ci", split_report)
                self.assertIn("paired_delta", split_report)
                self.assertEqual("dev", split_report["threshold_source"]["split"])
                self.assertEqual(-1, split_report["num_data"])
                self.assertEqual(1, split_report["sample_count"])
                self.assertEqual(1, len(split_report["qids"]))
                self.assertIn("configured", split_report["annotation_paths"])
                self.assertIn("resolved", split_report["annotation_paths"])
                self.assertIn("manifest", split_report["provenance"])


class CliTests(unittest.TestCase):
    def test_cli_default_output_is_atomic_json_in_validation_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev_dir = write_run(root, "dev", "dev", [record("d")])
            val_dir = write_run(
                root,
                "validation",
                "validation",
                [record("v", split="validation")],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/evaluate_c2r.py",
                    "--dev-run",
                    str(dev_dir),
                    "--validation-run",
                    str(val_dir),
                ],
                cwd=Path(__file__).parents[1],
                text=True,
                capture_output=True,
            )

            self.assertEqual(0, result.returncode, result.stderr)
            output = val_dir / "c2r_evaluation.json"
            self.assertTrue(output.is_file())
            self.assertEqual("dev", json.loads(output.read_text())["threshold_source"]["split"])
            self.assertEqual([], list(val_dir.glob(".c2r_evaluation.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
