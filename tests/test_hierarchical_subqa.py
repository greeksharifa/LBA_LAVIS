import math
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


class HierarchyConfigTests(unittest.TestCase):
    def test_default_config_exposes_flat_compatibility_and_new_global_defaults(self):
        from config.configs import Config
        from subqa.schema import normalize_hierarchy_config

        cfg = Config(
            Namespace(
                cfg_path="config/default.yaml",
                options=[
                    "model.model_name=qwen2.5-vl-7b",
                    "dataset.dataset_name=MMMU",
                ],
            )
        )

        hierarchy = normalize_hierarchy_config(cfg.runner_cfg)
        self.assertEqual((4, 2, 4), (cfg.runner_cfg.N, cfg.runner_cfg.M, cfg.runner_cfg.K))
        self.assertEqual(1, hierarchy.depth)
        self.assertEqual((4,), hierarchy.branching)
        self.assertEqual((2, 3), (hierarchy.suba_m, hierarchy.suba_k))
        self.assertEqual(2, hierarchy.schema_version)
        self.assertFalse(hierarchy.enabled)

    def test_depth_two_primary_config_is_canonical(self):
        from subqa.schema import normalize_hierarchy_config

        hierarchy = normalize_hierarchy_config(
            self._runner(subqa_depth=2, branching_by_depth=[4, 3])
        )

        self.assertEqual((4, 3), hierarchy.branching)
        self.assertEqual(16, hierarchy.node_count)
        self.assertTrue(hierarchy.enabled)

    def test_invalid_hierarchy_configurations_are_rejected(self):
        from subqa.schema import normalize_hierarchy_config

        cases = (
            ({"subqa_depth": 2, "branching_by_depth": None}, "branching_by_depth"),
            ({"subqa_depth": 2, "branching_by_depth": [4]}, "length"),
            ({"subqa_depth": 2, "branching_by_depth": [3, 3]}, "runner.N"),
            ({"subqa_depth": True}, "subqa_depth"),
            ({"subqa_depth": 2, "branching_by_depth": [4, 0]}, "branching"),
            (
                {
                    "subqa_depth": 3,
                    "branching_by_depth": [4, 3, 3],
                    "subqa_max_nodes": 51,
                },
                "subqa_max_nodes",
            ),
            (
                {"subqa_depth": 2, "branching_by_depth": [4, 1]},
                "suba_M",
            ),
            (
                {
                    "subqa_depth": 2,
                    "branching_by_depth": [4, 3],
                    "suba_K": 4,
                },
                "suba_K",
            ),
            ({"subqa_generation_batch_size": 0}, "batch"),
            ({"subqa_repair_attempts": -1}, "repair"),
            ({"condition_on_direct_suba": 1}, "condition_on_direct_suba"),
            ({"subqa_schema_version": 1}, "schema"),
            ({"suba_confidence_type": "seq_conf"}, "confidence"),
            (
                {
                    "subqa_depth": 2,
                    "branching_by_depth": [4, 3],
                    "confidence_type": "seq_conf",
                },
                "confidence_type",
            ),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    normalize_hierarchy_config(self._runner(**overrides))

    def test_pipeline_validates_hierarchy_before_model_factory(self):
        from pipeline import run

        cfg = SimpleNamespace(
            runner_cfg=self._runner(
                mode="multi_stage",
                subqa_depth=2,
                branching_by_depth=None,
            )
        )
        called = []

        with self.assertRaisesRegex(ValueError, "branching_by_depth"):
            run(cfg, model_factory=lambda received: called.append(received))

        self.assertEqual([], called)

    @staticmethod
    def _runner(**overrides):
        values = {
            "mode": "multi_stage",
            "N": 4,
            "M": 2,
            "K": 4,
            "subqa_depth": 1,
            "branching_by_depth": None,
            "suba_M": 2,
            "suba_K": 3,
            "suba_confidence_type": "token_min_prob",
            "condition_on_direct_suba": True,
            "subqa_max_nodes": 64,
            "subqa_repair_attempts": 1,
            "subqa_generation_batch_size": 64,
            "subqa_schema_version": 2,
        }
        values.update(overrides)
        return OmegaConf.create(values)


class GenerationProtocolTests(unittest.TestCase):
    def test_generation_result_is_frozen_and_model_independent(self):
        from dataclasses import FrozenInstanceError

        from model.protocol import GenerationResult

        result = GenerationResult(
            text="answer",
            confidence={"seq_ppl": 2.0, "token_min_prob": 0.7},
        )

        self.assertEqual("answer", result.text)
        self.assertEqual(0.7, result.confidence["token_min_prob"])
        with self.assertRaises(FrozenInstanceError):
            result.text = "changed"

    def test_vllm_adapter_extracts_text_and_confidence(self):
        from model.protocol import generation_result_from_vllm

        token_ids = [7, 8]
        completion = SimpleNamespace(
            text="answer",
            cumulative_logprob=-math.log(4.0),
            token_ids=token_ids,
            logprobs=[
                {7: SimpleNamespace(logprob=math.log(0.8))},
                {8: SimpleNamespace(logprob=math.log(0.25))},
            ],
        )

        result = generation_result_from_vllm(SimpleNamespace(outputs=[completion]))

        self.assertEqual("answer", result.text)
        self.assertAlmostEqual(2.0, result.confidence["seq_ppl"])
        self.assertAlmostEqual(0.25, result.confidence["token_min_prob"])

    def test_vllm_adapter_handles_empty_generation(self):
        from model.protocol import generation_result_from_vllm

        completion = SimpleNamespace(
            text="",
            cumulative_logprob=0.0,
            token_ids=[],
            logprobs=[],
        )

        result = generation_result_from_vllm(SimpleNamespace(outputs=[completion]))

        self.assertEqual(
            {"seq_ppl": 0.0, "token_min_prob": 0.0},
            result.confidence,
        )

    def test_common_confidence_uses_first_normalized_maximum(self):
        from util.confidence import normalize_confidence, select_first_maximum

        self.assertEqual(0.8, normalize_confidence(0.8, "token_min_prob"))
        self.assertEqual(0.5, normalize_confidence(2.0, "seq_ppl"))
        self.assertEqual(
            ("first", 0.5, 0),
            select_first_maximum(
                ["first", "second"], [2.0, 2.0], "seq_ppl"
            ),
        )


class QwenTemplateTests(unittest.TestCase):
    @staticmethod
    def _model(modality):
        from model.models import Qwen2_5VL

        model = Qwen2_5VL.__new__(Qwen2_5VL)
        model.modality = modality
        return model

    def test_text_template_omits_vision_fields_and_tokens(self):
        rendered = self._model("text").apply_chat_template(
            "question", vision=None, mm_uuids="q0"
        )

        self.assertIsInstance(rendered, str)
        self.assertIn("question", rendered)
        self.assertNotIn("vision", rendered)
        self.assertNotIn("image_pad", rendered)
        self.assertNotIn("video_pad", rendered)

    def test_image_and_video_templates_preserve_payload_and_uuid_order(self):
        for modality, token in (("image", "<|image_pad|>"), ("video", "<|video_pad|>")):
            with self.subTest(modality=modality):
                vision = [object(), object()]
                rendered = self._model(modality).apply_chat_template(
                    "question", vision=vision, mm_uuids="q0"
                )
                self.assertIs(vision, rendered["multi_modal_data"][modality])
                self.assertEqual(
                    ["q0_0", "q0_1"], rendered["multi_modal_uuids"][modality]
                )
                self.assertEqual(2, rendered["prompt"].count(token))

    def test_unknown_or_missing_multimodal_payload_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "unsupported modality.*audio"):
            self._model("audio").apply_chat_template("question")
        for modality in ("image", "video"):
            with self.subTest(modality=modality):
                with self.assertRaisesRegex(ValueError, f"{modality}.*vision"):
                    self._model(modality).apply_chat_template("question", vision=None)


if __name__ == "__main__":
    unittest.main()
