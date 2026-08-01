import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_artifacts import make_config
from util.artifacts import MANIFEST_FILENAME
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
        "subq": {"subq_list": ["What animal?"], "conf_subq": {"token_min_prob": 0.5}},
        "suba": {"suba_list": ["A cat."], "conf_suba": {"token_min_prob": [0.75]}},
        "base": {"base_answer": "dog", "conf_base": {"token_min_prob": 0.25}},
        "refined": {
            "refined_answer_list": ["cat"],
            "conf_refined": {"token_min_prob": [0.95]},
        },
    }
    return {"q0": values[mode]}


class PipelineTests(unittest.TestCase):
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
                return [fixture_sample()]

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
            self.assertEqual(
                {stage: {"completed": True} for stage in stage_modes},
                manifest["stages"],
            )

    def test_refined_stage_writes_stable_sample_records(self):
        from pipeline import run_stage

        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), split="dev", n=1, m=1, k=1).for_stage("refined")
            run_stage(
                cfg,
                FakeModel(),
                dataset_loader=lambda stage_cfg: [fixture_sample()],
                prompt_builder=lambda mode, sample, stage_cfg, sampler: "refine",
                output_formatter=fake_formatter,
            )

            records = json.loads(
                (get_output_dir(cfg) / "refined_samples.json").read_text()
            )
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
                        "conf_refined": {"token_min_prob": [0.95]},
                    }
                ],
                records,
            )

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
