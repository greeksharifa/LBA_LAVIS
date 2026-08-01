import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_artifacts import make_config
from util.artifacts import (
    MANIFEST_FILENAME,
    STAGE_DEPENDENCIES,
    create_manifest,
    write_manifest,
)
from util.path import get_output_dir, get_output_filename


class FakeModel:
    def __init__(self):
        self.cfg = None
        self.prompts = []

    def apply_chat_template(self, prompt, vision=None, mm_uuids=None):
        rendered = {"prompt": prompt, "qid": mm_uuids}
        self.prompts.append(rendered)
        return rendered

    def generate(self, prompts):
        return list(prompts)


class VllmOutputModel(FakeModel):
    def __init__(self, output_texts):
        super().__init__()
        self.output_texts = output_texts

    def generate(self, prompts):
        return [fake_vllm_output(text) for text in self.output_texts]


def fake_vllm_output(text):
    token_id = 7
    completion = SimpleNamespace(
        text=text,
        cumulative_logprob=-0.25,
        token_ids=[token_id],
        logprobs=[{token_id: SimpleNamespace(logprob=math.log(0.8))}],
    )
    return SimpleNamespace(outputs=[completion])


def fixture_sample():
    return {
        "qid": "q0",
        "split": "sample-split-must-not-win",
        "main_q": "What is shown?",
        "gt_ans": "cat",
        "question_type": "open_ended",
        "data_type": "text",
        "base_answer": "dog",
        "conf_base": 0.25,
        "subq_list": ["What animal?"],
        "conf_subq_list": 0.5,
        "suba_list": ["A cat."],
        "conf_suba_list": 0.75,
    }


def fake_formatter(mode, outputs, qids, n):
    values = {
        "subq": {
            "subq_list": ["What animal?"],
            "conf_subq": {"seq_ppl": 2.0, "token_min_prob": 0.5},
        },
        "suba": {
            "suba_list": ["A cat."],
            "conf_suba": {"seq_ppl": [1.5], "token_min_prob": [0.75]},
        },
        "base": {
            "base_answer": "dog",
            "conf_base": {"seq_ppl": 2.0, "token_min_prob": 0.25},
        },
        "refined": {
            "refined_answer_list": ["cat"],
            "conf_refined": {"seq_ppl": [1.25], "token_min_prob": [0.95]},
        },
    }
    return {"q0": values[mode]}


class SnapshotDataset(list):
    def __init__(self, values, dependency_generations):
        super().__init__(values)
        self.dependency_generations = dependency_generations


def snapshot_dataset(cfg, sample=None):
    sample = sample or fixture_sample()
    manifest_path = get_output_dir(cfg) / MANIFEST_FILENAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = create_manifest(cfg, [sample["qid"]])
        for parent in STAGE_DEPENDENCIES[str(cfg.runner_cfg.mode)]:
            manifest["stages"][parent].update(
                {
                    "completed": True,
                    "state": "completed",
                    "generation_id": f"{parent}-fixture-generation",
                    "parent_generations": {
                        ancestor: f"{ancestor}-fixture-generation"
                        for ancestor in STAGE_DEPENDENCIES[parent]
                    },
                }
            )
            manifest["generation_history"][parent].append(
                f"{parent}-fixture-generation"
            )
        write_manifest(manifest_path, manifest)
    parent_generations = {
        parent: manifest["stages"][parent]["generation_id"]
        for parent in STAGE_DEPENDENCIES[str(cfg.runner_cfg.mode)]
    }
    return SnapshotDataset([sample], parent_generations)


class PipelineTests(unittest.TestCase):
    def test_suba_prompt_count_must_match_configured_n_before_generation(self):
        from pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), n=2, m=1, k=1).for_stage("suba")
            sample = fixture_sample()
            sample["subq_list"] = ["corrupted short list"]

            class GenerationMustNotRun(FakeModel):
                def generate(self, prompts):
                    raise AssertionError("generation must not run")

            with self.assertRaisesRegex(
                ValueError,
                "prompt count mismatch.*suba.*q0.*expected 2.*got 1",
            ):
                run_stage(
                    cfg,
                    GenerationMustNotRun(),
                    dataset_loader=lambda stage_cfg: [sample],
                )

    def test_refined_prompt_count_must_match_configured_k_before_generation(self):
        from pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), n=1, m=1, k=2).for_stage("refined")

            class GenerationMustNotRun(FakeModel):
                def generate(self, prompts):
                    raise AssertionError("generation must not run")

            with self.assertRaisesRegex(
                ValueError,
                "prompt count mismatch.*refined.*q0.*expected 2.*got 1",
            ):
                run_stage(
                    cfg,
                    GenerationMustNotRun(),
                    dataset_loader=lambda stage_cfg: [fixture_sample()],
                    prompt_builder=(
                        lambda mode, sample, stage_cfg, sampler: [
                            "refine" for _ in sampler.indices
                        ]
                    ),
                )

    def test_subq_and_base_require_one_prompt_per_qid(self):
        from pipeline import run_stage

        class GenerationMustNotRun(FakeModel):
            def generate(self, prompts):
                raise AssertionError("generation must not run")

        for mode in ("subq", "base"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                cfg = make_config(Path(tmp), n=1, m=1, k=1).for_stage(mode)
                with self.assertRaisesRegex(
                    ValueError,
                    f"prompt count mismatch.*{mode}.*q0.*expected 1.*got 2",
                ):
                    run_stage(
                        cfg,
                        GenerationMustNotRun(),
                        dataset_loader=lambda stage_cfg: [fixture_sample()],
                        prompt_builder=(
                            lambda mode, sample, stage_cfg, sampler: ["a", "b"]
                        ),
                        output_formatter=fake_formatter,
                    )

    def test_empty_dataset_is_rejected_before_manifest_creation(self):
        from pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), n=1, m=1, k=1).for_stage("base")
            with self.assertRaisesRegex(ValueError, "dataset.*no samples"):
                run_stage(
                    cfg,
                    FakeModel(),
                    dataset_loader=lambda stage_cfg: [],
                    output_formatter=fake_formatter,
                )

            run_dir = get_output_dir(cfg)
            self.assertFalse((run_dir / MANIFEST_FILENAME).exists())
            self.assertFalse((run_dir / get_output_filename(cfg)).exists())

    def test_real_formatter_keeps_singleton_suba_and_refined_values_as_lists(self):
        from pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode in ("suba", "refined"):
                with self.subTest(mode=mode):
                    cfg = make_config(root / mode, n=1, m=1, k=1).for_stage(mode)
                    result = run_stage(
                        cfg,
                        VllmOutputModel(["cat"]),
                        dataset_loader=snapshot_dataset,
                        prompt_builder=lambda mode, sample, stage_cfg, sampler: ["only"],
                    )

                    value_key = f"{mode}_list" if mode == "suba" else "refined_answer_list"
                    confidence = result["q0"][f"conf_{mode}"]
                    self.assertEqual(["cat"], result["q0"][value_key])
                    self.assertEqual(1, len(confidence["seq_ppl"]))
                    self.assertEqual([0.8], confidence["token_min_prob"])

    def test_missing_generated_output_preserves_artifact_and_marks_rerun_incomplete(self):
        from pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), n=1, m=1, k=1).for_stage("base")
            kwargs = {
                "dataset_loader": lambda stage_cfg: [fixture_sample()],
                "prompt_builder": lambda mode, sample, stage_cfg, sampler: "base",
            }
            run_stage(cfg, VllmOutputModel(["cat"]), **kwargs)
            output_path = get_output_dir(cfg) / get_output_filename(cfg)
            successful_artifact = output_path.read_bytes()

            with self.assertRaisesRegex(ValueError, "output count.*1.*0"):
                run_stage(cfg, VllmOutputModel([]), **kwargs)

            self.assertEqual(successful_artifact, output_path.read_bytes())
            manifest = json.loads(
                (get_output_dir(cfg) / MANIFEST_FILENAME).read_text()
            )
            self.assertFalse(manifest["stages"]["base"]["completed"])
            self.assertEqual("running", manifest["stages"]["base"]["state"])
            self.assertTrue(manifest["stages"]["base"]["generation_id"])

    def test_rerun_crash_exposes_running_manifest_and_leaves_incomplete(self):
        from pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), n=1, m=1, k=1).for_stage("base")
            kwargs = {
                "dataset_loader": lambda stage_cfg: [fixture_sample()],
                "prompt_builder": lambda mode, sample, stage_cfg, sampler: "base",
            }
            run_stage(cfg, VllmOutputModel(["cat"]), **kwargs)
            manifest_path = get_output_dir(cfg) / MANIFEST_FILENAME

            class CrashingModel(FakeModel):
                def generate(self, prompts):
                    running = json.loads(manifest_path.read_text())["stages"]["base"]
                    self.seen_state = running
                    raise RuntimeError("simulated generation crash")

            model = CrashingModel()
            with self.assertRaisesRegex(RuntimeError, "simulated generation crash"):
                run_stage(cfg, model, **kwargs)

            self.assertFalse(model.seen_state["completed"])
            self.assertEqual("running", model.seen_state["state"])
            saved = json.loads(manifest_path.read_text())["stages"]["base"]
            self.assertFalse(saved["completed"])
            self.assertEqual("running", saved["state"])

    def test_formatted_qids_and_schema_must_be_exact(self):
        from pipeline import run_stage

        invalid_results = (
            {
                "other": {
                    "base_answer": "cat",
                    "conf_base": {"seq_ppl": 1.0, "token_min_prob": 0.8},
                }
            },
            {"q0": {"base_answer": "cat"}},
        )
        for invalid in invalid_results:
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as tmp:
                cfg = make_config(Path(tmp), n=1, m=1, k=1).for_stage("base")
                with self.assertRaisesRegex(ValueError, "formatted (qid|schema)"):
                    run_stage(
                        cfg,
                        FakeModel(),
                        dataset_loader=lambda stage_cfg: [fixture_sample()],
                        prompt_builder=lambda mode, sample, stage_cfg, sampler: "base",
                        output_formatter=lambda mode, outputs, qids, n: invalid,
                    )

    def test_formatted_list_counts_must_match_prompt_occurrences(self):
        from pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), n=2, m=1, k=1).for_stage("suba")
            invalid = {
                "q0": {
                    "suba_list": ["only one"],
                    "conf_suba": {
                        "seq_ppl": [1.0],
                        "token_min_prob": [0.8],
                    },
                }
            }
            with self.assertRaisesRegex(ValueError, "formatted count.*q0.*2.*1"):
                run_stage(
                    cfg,
                    FakeModel(),
                    dataset_loader=snapshot_dataset,
                    prompt_builder=lambda mode, sample, stage_cfg, sampler: ["a", "b"],
                    output_formatter=lambda mode, outputs, qids, n: invalid,
                )

    def test_new_manifest_is_validated_before_stage_generation(self):
        from pipeline import run_stage
        from util.artifacts import validate_manifest

        with tempfile.TemporaryDirectory() as tmp, patch(
            "pipeline.validate_manifest", wraps=validate_manifest
        ) as validate:
            cfg = make_config(Path(tmp), n=1, m=1, k=1).for_stage("base")
            run_stage(
                cfg,
                FakeModel(),
                dataset_loader=lambda stage_cfg: [fixture_sample()],
                prompt_builder=lambda mode, sample, stage_cfg, sampler: "base",
                output_formatter=fake_formatter,
            )

            validate.assert_called_once()

    def test_multi_stage_reuses_one_model_and_completes_stages_in_order(self):
        from pipeline import run_multi_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), n=1, m=1, k=1)
            model = FakeModel()
            factory_calls = []
            stage_modes = []
            stage_configs = []

            def model_factory(received_cfg):
                factory_calls.append(received_cfg)
                return model

            def dataset_loader(stage_cfg):
                stage_modes.append(stage_cfg.runner_cfg.mode)
                stage_configs.append(stage_cfg)
                self.assertNotEqual(id(cfg), id(stage_cfg))
                return snapshot_dataset(stage_cfg)

            run_multi_stage(
                cfg,
                model_factory=model_factory,
                dataset_loader=dataset_loader,
                prompt_builder=lambda mode, sample, stage_cfg, sampler: f"{mode}:{sample['qid']}",
                output_formatter=fake_formatter,
            )

            self.assertEqual([cfg], factory_calls)
            self.assertEqual(["subq", "suba", "base", "refined"], stage_modes)
            self.assertEqual(4, len({id(stage_cfg) for stage_cfg in stage_configs}))
            run_dir = get_output_dir(cfg)
            for stage in stage_modes:
                stage_cfg = cfg.for_stage(stage)
                self.assertTrue((run_dir / get_output_filename(stage_cfg)).is_file())
            manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text())
            for stage in stage_modes:
                self.assertTrue(manifest["stages"][stage]["completed"])
                self.assertEqual("completed", manifest["stages"][stage]["state"])
                self.assertTrue(manifest["stages"][stage]["generation_id"])

    def test_refined_stage_writes_stable_sample_records(self):
        from pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), split="dev", n=1, m=1, k=1).for_stage("refined")
            run_stage(
                cfg,
                FakeModel(),
                dataset_loader=snapshot_dataset,
                prompt_builder=lambda mode, sample, stage_cfg, sampler: "refine",
                output_formatter=fake_formatter,
            )

            records = json.loads(
                (get_output_dir(cfg) / "refined_samples.json").read_text()
            )
            manifest = json.loads(
                (get_output_dir(cfg) / MANIFEST_FILENAME).read_text()
            )
            generation_id = manifest["stages"]["refined"]["generation_id"]
            self.assertEqual(
                [
                    {
                        "qid": "q0",
                        "split": "dev",
                        "main_q": "What is shown?",
                        "gt_ans": "cat",
                        "question_type": "open_ended",
                        "base_answer": "dog",
                        "conf_base": 0.25,
                        "refined_answer_list": ["cat"],
                        "conf_refined": {
                            "seq_ppl": [1.25],
                            "token_min_prob": [0.95],
                        },
                        "generation_id": generation_id,
                    }
                ],
                records,
            )

    def test_completion_guard_failure_preserves_previous_canonical_artifact(self):
        from pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), n=1, m=1, k=1).for_stage("base")

            def answer_formatter(answer):
                def formatter(mode, outputs, qids, n):
                    formatted = fake_formatter(mode, outputs, qids, n)
                    formatted["q0"]["base_answer"] = answer
                    return formatted

                return formatter

            kwargs = {
                "dataset_loader": lambda stage_cfg: [fixture_sample()],
                "prompt_builder": lambda mode, sample, stage_cfg, sampler: "base",
            }
            run_stage(
                cfg,
                FakeModel(),
                output_formatter=answer_formatter("old"),
                **kwargs,
            )
            output_path = get_output_dir(cfg) / get_output_filename(cfg)
            previous = output_path.read_bytes()

            with patch(
                "pipeline.mark_stage_complete",
                side_effect=ValueError("superseded completion"),
            ), self.assertRaisesRegex(ValueError, "superseded completion"):
                run_stage(
                    cfg,
                    FakeModel(),
                    output_formatter=answer_formatter("new"),
                    **kwargs,
                )

            self.assertEqual(previous, output_path.read_bytes())

    def test_refined_second_artifact_write_failure_remains_incomplete(self):
        from evaluation.c2r import load_run
        from pipeline import run_stage, write_json_atomic as real_write

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(
                Path(tmp), split="dev", n=1, m=1, k=1
            ).for_stage("refined")
            run_dir = get_output_dir(cfg)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "refined_samples.json").write_text("[]")

            def fail_samples(path, value):
                if Path(path).name == "refined_samples.json":
                    raise RuntimeError("simulated refined samples write failure")
                return real_write(path, value)

            with patch("pipeline.write_json_atomic", side_effect=fail_samples):
                with self.assertRaisesRegex(RuntimeError, "samples write failure"):
                    run_stage(
                        cfg,
                        FakeModel(),
                        dataset_loader=snapshot_dataset,
                        prompt_builder=(
                            lambda mode, sample, stage_cfg, sampler: "refine"
                        ),
                        output_formatter=fake_formatter,
                    )

            status = json.loads(
                (run_dir / MANIFEST_FILENAME).read_text()
            )["stages"]["refined"]
            self.assertFalse(status["completed"])
            self.assertEqual("running", status["state"])
            with self.assertRaisesRegex(ValueError, "not completed"):
                load_run(run_dir)

    def test_upstream_rerun_invalidates_stale_refined_evaluation(self):
        from evaluation.c2r import load_run
        from pipeline import run_multi_stage, run_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(
                Path(tmp), split="dev", n=1, m=1, k=1
            )
            cfg.dataset_cfg.dataset_name = "MMMU"
            (Path(tmp) / "annotations.json").write_text(
                json.dumps([{"question_id": "q0"}]),
                encoding="utf-8",
            )
            run_multi_stage(
                cfg,
                model_factory=lambda received_cfg: FakeModel(),
                dataset_loader=snapshot_dataset,
                prompt_builder=(
                    lambda mode, sample, stage_cfg, sampler: f"{mode}:{sample['qid']}"
                ),
                output_formatter=fake_formatter,
            )
            run_dir = get_output_dir(cfg)
            stale_records = (run_dir / "refined_samples.json").read_bytes()
            self.assertEqual(["q0"], load_run(run_dir)["qids"])

            class CrashingModel(FakeModel):
                def generate(self, prompts):
                    raise RuntimeError("simulated upstream crash")

            with self.assertRaisesRegex(RuntimeError, "upstream crash"):
                run_stage(
                    cfg.for_stage("subq"),
                    CrashingModel(),
                    dataset_loader=snapshot_dataset,
                    prompt_builder=(
                        lambda mode, sample, stage_cfg, sampler: "subq"
                    ),
                    output_formatter=fake_formatter,
                )

            self.assertEqual(
                stale_records,
                (run_dir / "refined_samples.json").read_bytes(),
            )
            with self.assertRaisesRegex(ValueError, "not completed"):
                load_run(run_dir)

    def test_single_stage_uses_the_same_run_stage_boundary(self):
        from pipeline import run

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), n=1, m=1, k=1)
            cfg.runner_cfg.mode = "base"
            model = FakeModel()
            calls = []

            def stage_runner(stage_cfg, received_model, **kwargs):
                calls.append((stage_cfg, received_model))
                return {"q0": {"base_answer": "cat"}}

            result = run(
                cfg,
                model_factory=lambda received_cfg: model,
                stage_runner=stage_runner,
            )

            self.assertEqual({"q0": {"base_answer": "cat"}}, result)
            self.assertEqual(1, len(calls))
            self.assertEqual("base", calls[0][0].runner_cfg.mode)
            self.assertIsNot(cfg, calls[0][0])
            self.assertIs(model, calls[0][1])


if __name__ == "__main__":
    unittest.main()
