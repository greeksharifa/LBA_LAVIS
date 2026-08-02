import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model.protocol import GenerationResult
from tests.test_artifacts import make_config
from util.artifacts import MANIFEST_FILENAME
from util.path import get_output_dir


def hierarchy_config(
    root,
    *,
    mode="subq",
    batch_size=2,
    repair_attempts=1,
    branching=(2, 2),
    suba_m=2,
    suba_k=1,
):
    cfg = make_config(root, n=branching[0], m=1, k=1).for_stage(mode)
    cfg.runner_cfg.subqa_depth = 2
    cfg.runner_cfg.subqa_depth = len(branching)
    cfg.runner_cfg.branching_by_depth = list(branching)
    cfg.runner_cfg.suba_M = suba_m
    cfg.runner_cfg.suba_K = suba_k
    cfg.runner_cfg.suba_confidence_type = "token_min_prob"
    cfg.runner_cfg.condition_on_direct_suba = True
    cfg.runner_cfg.subqa_max_nodes = 16
    cfg.runner_cfg.subqa_repair_attempts = repair_attempts
    cfg.runner_cfg.subqa_generation_batch_size = batch_size
    cfg.runner_cfg.subqa_schema_version = 2
    return cfg


def sample(qid, *, data_type="text", vision=None):
    value = {
        "qid": qid,
        "main_q": f"main question {qid}",
        "question_type": "open_ended",
        "data_type": data_type,
    }
    if vision is not None:
        value["vision"] = vision
    return value


class CompleteDecompositionModel:
    def __init__(self):
        self.template_calls = []
        self.batch_sizes = []

    def apply_chat_template(self, prompt, vision=None, mm_uuids=None):
        rendered = {"prompt": prompt, "qid": mm_uuids, "vision": vision}
        self.template_calls.append(rendered)
        return rendered

    def generate_results(self, prompts):
        self.batch_sizes.append(len(prompts))
        results = []
        for prompt in prompts:
            qid = prompt["qid"]
            text = prompt["prompt"]
            parent_prefix = "Parent question: "
            if parent_prefix in text:
                parent = text.split(parent_prefix, 1)[1].splitlines()[0]
                output = f"1. {parent} child 0?\n2. {parent} child 1?"
            else:
                output = f"1. {qid} root 0?\n2. {qid} root 1?"
            results.append(
                GenerationResult(
                    output,
                    {"seq_ppl": 2.0, "token_min_prob": 0.5},
                )
            )
        return results


class HierarchicalSubQOrchestrationTests(unittest.TestCase):
    def test_generation_requires_only_configured_confidence_metric_union(self):
        from subqa.orchestrator import generate_subq_artifacts
        from subqa.schema import normalize_hierarchy_config

        class SingleMetricModel(CompleteDecompositionModel):
            def generate_results(self, prompts):
                results = super().generate_results(prompts)
                return [
                    GenerationResult(
                        result.text,
                        {"token_min_prob": result.confidence["token_min_prob"]},
                    )
                    for result in results
                ]

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(Path(tmp))
            hierarchy = normalize_hierarchy_config(cfg.runner_cfg)
            artifact = generate_subq_artifacts(
                [sample("q0")], SingleMetricModel(), hierarchy
            )["q0"]
            self.assertEqual(
                {"token_min_prob": 0.5}, artifact["conf_subq"]
            )

            cfg.runner_cfg.suba_confidence_type = "seq_ppl"
            hierarchy = normalize_hierarchy_config(cfg.runner_cfg)
            with self.assertRaisesRegex(ValueError, "seq_ppl"):
                generate_subq_artifacts(
                    [sample("q0")], SingleMetricModel(), hierarchy
                )

    def test_multi_qid_frontiers_are_chunked_in_qid_then_node_order(self):
        from subqa.orchestrator import generate_subq_artifacts
        from subqa.schema import normalize_hierarchy_config, validate_subq_tree

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(Path(tmp), batch_size=2)
            model = CompleteDecompositionModel()

            artifacts = generate_subq_artifacts(
                [sample("q0"), sample("q1")],
                model,
                normalize_hierarchy_config(cfg.runner_cfg),
            )

        self.assertEqual([2, 2, 2], model.batch_sizes)
        self.assertEqual(
            ["q0", "q1", "q0", "q0", "q1", "q1"],
            [call["qid"] for call in model.template_calls],
        )
        self.assertEqual(["q0", "q1"], list(artifacts))
        for qid, artifact in artifacts.items():
            self.assertEqual([f"{qid} root 0?", f"{qid} root 1?"], artifact["subq_list"])
            self.assertEqual(
                ["0", "0.0", "0.1", "1", "1.0", "1.1"],
                [node["id"] for node in artifact["subq_tree"]["nodes"]],
            )
            validate_subq_tree(artifact["subq_tree"], branching=(2, 2))

    def test_root_shortfall_is_repaired_without_generic_padding(self):
        from subqa.orchestrator import generate_subq_artifacts
        from subqa.schema import normalize_hierarchy_config

        class RepairingModel(CompleteDecompositionModel):
            def generate_results(self, prompts):
                self.batch_sizes.append(len(prompts))
                results = []
                for prompt in prompts:
                    text = prompt["prompt"]
                    if "Complete a partial question decomposition" in text:
                        output = "1. repaired root 1?"
                    elif "Parent question:" not in text:
                        output = "1. original root 0?"
                    else:
                        parent = text.split("Parent question: ", 1)[1].splitlines()[0]
                        output = f"1. {parent} child 0?\n2. {parent} child 1?"
                    results.append(
                        GenerationResult(
                            output,
                            {"seq_ppl": 3.0, "token_min_prob": 0.4},
                        )
                    )
                return results

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(Path(tmp))
            artifact = generate_subq_artifacts(
                [sample("q0")],
                RepairingModel(),
                normalize_hierarchy_config(cfg.runner_cfg),
            )["q0"]

        self.assertEqual(
            ["original root 0?", "repaired root 1?"], artifact["subq_list"]
        )
        self.assertEqual(0.4, artifact["conf_subq"]["token_min_prob"])
        self.assertNotIn("Can you describe", json.dumps(artifact))

    def test_root_repair_exhaustion_fails_the_stage(self):
        from subqa.orchestrator import (
            HierarchyStageIncompleteError,
            generate_subq_artifacts,
        )
        from subqa.schema import normalize_hierarchy_config

        class AlwaysShort(CompleteDecompositionModel):
            def generate_results(self, prompts):
                self.batch_sizes.append(len(prompts))
                return [
                    GenerationResult(
                        "1. only one?",
                        {"seq_ppl": 2.0, "token_min_prob": 0.5},
                    )
                    for _ in prompts
                ]

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(Path(tmp))
            with self.assertRaisesRegex(
                HierarchyStageIncompleteError, "root.*q0.*expected 2.*got 1"
            ):
                generate_subq_artifacts(
                    [sample("q0")],
                    AlwaysShort(),
                    normalize_hierarchy_config(cfg.runner_cfg),
                )

    def test_descendant_shortfall_is_local_and_records_partial_status(self):
        from subqa.orchestrator import generate_subq_artifacts
        from subqa.schema import normalize_hierarchy_config, validate_subq_tree

        class PartialDescendant(CompleteDecompositionModel):
            def generate_results(self, prompts):
                self.batch_sizes.append(len(prompts))
                results = []
                for prompt in prompts:
                    qid = prompt["qid"]
                    text = prompt["prompt"]
                    if "Parent question: q0 root 0?" in text:
                        output = "1. surviving child?"
                    elif "Complete a partial question decomposition" in text:
                        output = "1. surviving child?"
                    elif "Parent question:" in text:
                        parent = text.split("Parent question: ", 1)[1].splitlines()[0]
                        output = f"1. {parent} child 0?\n2. {parent} child 1?"
                    else:
                        output = f"1. {qid} root 0?\n2. {qid} root 1?"
                    results.append(
                        GenerationResult(
                            output,
                            {"seq_ppl": 2.0, "token_min_prob": 0.5},
                        )
                    )
                return results

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(Path(tmp))
            artifacts = generate_subq_artifacts(
                [sample("q0"), sample("q1")],
                PartialDescendant(),
                normalize_hierarchy_config(cfg.runner_cfg),
            )

        q0_nodes = {node["id"]: node for node in artifacts["q0"]["subq_tree"]["nodes"]}
        self.assertEqual("partial", q0_nodes["0"]["expansion_status"])
        self.assertEqual(["0.0"], q0_nodes["0"]["child_ids"])
        self.assertIn("expected 2", q0_nodes["0"]["failure_reason"])
        self.assertEqual("expanded", q0_nodes["1"]["expansion_status"])
        self.assertEqual(6, len(artifacts["q1"]["subq_tree"]["nodes"]))
        validate_subq_tree(artifacts["q0"]["subq_tree"], branching=(2, 2))

    def test_output_count_mismatch_and_malformed_results_are_system_errors(self):
        from subqa.orchestrator import generate_subq_artifacts
        from subqa.schema import normalize_hierarchy_config

        class BadModel(CompleteDecompositionModel):
            def __init__(self, result):
                super().__init__()
                self.result = result

            def generate_results(self, prompts):
                return self.result(prompts)

        cases = (
            lambda prompts: [],
            lambda prompts: [object() for _ in prompts],
            lambda prompts: [
                GenerationResult("1. a?\n2. b?", {"seq_ppl": 2.0})
                for _ in prompts
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(Path(tmp))
            hierarchy = normalize_hierarchy_config(cfg.runner_cfg)
            for result in cases:
                with self.subTest(result=result):
                    with self.assertRaises(ValueError):
                        generate_subq_artifacts(
                            [sample("q0")], BadModel(result), hierarchy
                        )

    def test_pipeline_failure_preserves_canonical_artifact_and_leaves_manifest_running(self):
        from pipeline import run_stage

        class AlwaysShort(CompleteDecompositionModel):
            def generate_results(self, prompts):
                return [
                    GenerationResult(
                        "1. only one?",
                        {"seq_ppl": 2.0, "token_min_prob": 0.5},
                    )
                    for _ in prompts
                ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = hierarchy_config(root)
            run_dir = get_output_dir(cfg)
            run_dir.mkdir(parents=True)
            output_path = run_dir / "subq_outputs.json"
            output_path.write_text('{"old": true}\n')

            with self.assertRaisesRegex(RuntimeError, "root"):
                run_stage(
                    cfg,
                    AlwaysShort(),
                    dataset_loader=lambda received: [sample("q0")],
                )

            self.assertEqual('{"old": true}\n', output_path.read_text())
            manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text())
            self.assertEqual("running", manifest["stages"]["subq"]["state"])
            self.assertFalse(manifest["stages"]["subq"]["completed"])


def answered_sample(qid, branching=(2, 3)):
    from subqa.schema import build_complete_subq_tree, project_depth_one_questions

    value = sample(qid)
    tree = build_complete_subq_tree(
        branching, question_factory=lambda node_id: f"{qid} question {node_id}?"
    )
    value.update(
        {
            "subq_tree": tree,
            "subq_list": project_depth_one_questions(
                tree, expected_count=branching[0]
            ),
            "conf_subq_list": 0.5,
        }
    )
    return value


class AnsweringModel:
    def __init__(self, *, blank_refined=False, blank_targets=()):
        self.prompts = []
        self.batch_sizes = []
        self.blank_refined = blank_refined
        self.blank_targets = tuple(blank_targets)
        self.refined_index = 0

    def apply_chat_template(self, prompt, vision=None, mm_uuids=None):
        rendered = {"prompt": prompt, "qid": mm_uuids, "vision": vision}
        self.prompts.append(rendered)
        return rendered

    def generate_results(self, prompts):
        self.batch_sizes.append(len(prompts))
        results = []
        refined_confidences = (0.1, 0.9, 0.5)
        for rendered in prompts:
            prompt = rendered["prompt"]
            target = prompt.split("Target question: ", 1)[1].splitlines()[0]
            if "Refine an answer" in prompt:
                answer = "" if self.blank_refined else f"refined {target}"
                confidence = refined_confidences[self.refined_index % 3]
                self.refined_index += 1
            else:
                answer = "" if target in self.blank_targets else f"direct {target}"
                confidence = 0.2
            results.append(
                GenerationResult(
                    answer,
                    {"seq_ppl": 1.0 / max(confidence, 1e-6), "token_min_prob": confidence},
                )
            )
        return results


class HierarchicalSubAOrchestrationTests(unittest.TestCase):
    def test_generated_suba_is_self_validated_before_return(self):
        from subqa.orchestrator import generate_suba_artifacts
        from subqa.schema import normalize_hierarchy_config

        def select_wrong_candidate(candidates, confidence_type):
            return candidates[0], candidates[0]["confidence"][confidence_type], 0

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(
                Path(tmp), mode="suba", branching=(2, 3), suba_k=3
            )
            with patch(
                "subqa.orchestrator.select_best_candidate",
                side_effect=select_wrong_candidate,
            ), self.assertRaisesRegex(ValueError, "first maximum"):
                generate_suba_artifacts(
                    [answered_sample("q0")],
                    AnsweringModel(),
                    normalize_hierarchy_config(cfg.runner_cfg),
                )

    def test_single_configured_metric_is_preserved_in_flat_projection(self):
        from subqa.orchestrator import generate_suba_artifacts
        from subqa.schema import normalize_hierarchy_config

        class SingleMetricAnsweringModel(AnsweringModel):
            def generate_results(self, prompts):
                results = super().generate_results(prompts)
                return [
                    GenerationResult(
                        result.text,
                        {"token_min_prob": result.confidence["token_min_prob"]},
                    )
                    for result in results
                ]

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(
                Path(tmp), mode="suba", branching=(2, 3), suba_k=3
            )
            artifact = generate_suba_artifacts(
                [answered_sample("q0")],
                SingleMetricAnsweringModel(),
                normalize_hierarchy_config(cfg.runner_cfg),
            )["q0"]

        self.assertEqual({"token_min_prob"}, set(artifact["conf_suba"]))

    def test_direct_answers_precede_deepest_first_candidates_and_project_depth_one(self):
        from subqa.orchestrator import generate_suba_artifacts
        from subqa.schema import normalize_hierarchy_config

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(
                Path(tmp), mode="suba", branching=(2, 3), suba_k=3
            )
            model = AnsweringModel()
            artifact = generate_suba_artifacts(
                [answered_sample("q0")],
                model,
                normalize_hierarchy_config(cfg.runner_cfg),
            )["q0"]

        prompts = [call["prompt"] for call in model.prompts]
        self.assertTrue(all("Answer the target" in prompt for prompt in prompts[:8]))
        self.assertTrue(all("Refine an answer" in prompt for prompt in prompts[8:]))
        answers = artifact["answers_by_node"]
        self.assertEqual(
            ["0", "0.0", "0.1", "0.2", "1", "1.0", "1.1", "1.2"],
            list(answers),
        )
        for leaf_id in ("0.0", "0.1", "0.2", "1.0", "1.1", "1.2"):
            self.assertEqual("leaf_direct", answers[leaf_id]["selected"]["source"])
        self.assertEqual(
            [
                ["0.0", "0.1"],
                ["0.0", "0.2"],
                ["0.1", "0.2"],
            ],
            [candidate["support_node_ids"] for candidate in answers["0"]["candidates"]],
        )
        self.assertEqual("confidence", answers["0"]["selected"]["source"])
        self.assertEqual(1, answers["0"]["selected"]["candidate_index"])
        self.assertEqual(
            [answers["0"]["selected"]["answer"], answers["1"]["selected"]["answer"]],
            artifact["suba_list"],
        )
        self.assertEqual([0.9, 0.9], artifact["conf_suba"]["token_min_prob"])

    def test_all_invalid_candidates_use_direct_fallback_and_can_omit_direct_context(self):
        from subqa.orchestrator import generate_suba_artifacts
        from subqa.schema import normalize_hierarchy_config

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(
                Path(tmp), mode="suba", branching=(2, 3), suba_k=3
            )
            cfg.runner_cfg.condition_on_direct_suba = False
            model = AnsweringModel(blank_refined=True)
            artifact = generate_suba_artifacts(
                [answered_sample("q0")],
                model,
                normalize_hierarchy_config(cfg.runner_cfg),
            )["q0"]

        refined_prompts = [
            call["prompt"] for call in model.prompts if "Refine an answer" in call["prompt"]
        ]
        self.assertTrue(refined_prompts)
        self.assertTrue(all("Direct draft:" not in prompt for prompt in refined_prompts))
        for node_id in ("0", "1"):
            node_answer = artifact["answers_by_node"][node_id]
            self.assertTrue(all(c["status"] == "invalid" for c in node_answer["candidates"]))
            self.assertEqual("direct_fallback", node_answer["selected"]["source"])
            self.assertEqual(node_answer["direct"]["answer"], node_answer["selected"]["answer"])

    def test_invalid_descendant_is_recorded_and_excluded_from_parent_evidence(self):
        from subqa.orchestrator import generate_suba_artifacts
        from subqa.schema import normalize_hierarchy_config

        invalid_target = "q0 question 0.0?"
        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(
                Path(tmp),
                mode="suba",
                branching=(2, 3),
                suba_k=3,
                repair_attempts=0,
            )
            artifact = generate_suba_artifacts(
                [answered_sample("q0")],
                AnsweringModel(blank_targets=(invalid_target,)),
                normalize_hierarchy_config(cfg.runner_cfg),
            )["q0"]

        answers = artifact["answers_by_node"]
        self.assertEqual("invalid", answers["0.0"]["direct"]["status"])
        self.assertEqual("invalid", answers["0.0"]["selected"]["status"])
        self.assertEqual(
            [["0.1", "0.2"]],
            [candidate["support_node_ids"] for candidate in answers["0"]["candidates"]],
        )
        self.assertEqual("confidence", answers["0"]["selected"]["source"])

    def test_depth_one_direct_answer_repair_is_required(self):
        from subqa.orchestrator import (
            HierarchyStageIncompleteError,
            generate_suba_artifacts,
        )
        from subqa.schema import normalize_hierarchy_config

        class RepairDepthOne(AnsweringModel):
            def __init__(self, repair_succeeds):
                super().__init__()
                self.repair_succeeds = repair_succeeds

            def generate_results(self, prompts):
                results = []
                for rendered in prompts:
                    prompt = rendered["prompt"]
                    target = prompt.split("Target question: ", 1)[1].splitlines()[0]
                    is_depth_one = target in {"q0 question 0?", "q0 question 1?"}
                    is_repair = "previous response was blank" in prompt
                    answer = (
                        f"repaired {target}"
                        if is_depth_one and is_repair and self.repair_succeeds
                        else ("" if is_depth_one else f"direct {target}")
                    )
                    results.append(
                        GenerationResult(
                            answer,
                            {"seq_ppl": 2.0, "token_min_prob": 0.5},
                        )
                    )
                return results

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(Path(tmp), mode="suba", branching=(2, 3), suba_k=3)
            hierarchy = normalize_hierarchy_config(cfg.runner_cfg)
            artifact = generate_suba_artifacts(
                [answered_sample("q0")], RepairDepthOne(True), hierarchy
            )["q0"]
            self.assertTrue(artifact["answers_by_node"]["0"]["direct"]["answer"].startswith("repaired"))

            with self.assertRaisesRegex(
                HierarchyStageIncompleteError, "depth-1 direct answer.*q0.*node 0"
            ):
                generate_suba_artifacts(
                    [answered_sample("q0")], RepairDepthOne(False), hierarchy
                )

    def test_depth_three_refines_deepest_internal_nodes_first(self):
        from subqa.orchestrator import generate_suba_artifacts
        from subqa.schema import normalize_hierarchy_config

        branching = (1, 2, 2)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(
                Path(tmp), mode="suba", branching=branching, suba_m=2, suba_k=1
            )
            model = AnsweringModel()
            artifact = generate_suba_artifacts(
                [answered_sample("q0", branching=branching)],
                model,
                normalize_hierarchy_config(cfg.runner_cfg),
            )["q0"]

        refined_targets = [
            prompt["prompt"].split("Target question: ", 1)[1].splitlines()[0]
            for prompt in model.prompts
            if "Refine an answer" in prompt["prompt"]
        ]
        self.assertEqual(
            ["q0 question 0.0?", "q0 question 0.1?", "q0 question 0?"],
            refined_targets,
        )
        root_candidate_prompt = next(
            prompt["prompt"]
            for prompt in model.prompts
            if "Target question: q0 question 0?" in prompt["prompt"]
            and "Refine an answer" in prompt["prompt"]
        )
        self.assertIn("refined q0 question 0.0?", root_candidate_prompt)
        self.assertIn("refined q0 question 0.1?", root_candidate_prompt)
        self.assertEqual(1, len(artifact["suba_list"]))


class HierarchyContractMatrixTests(unittest.TestCase):
    def test_text_image_video_and_question_types_share_one_protocol(self):
        from prompt.prompts import get_refined_prompt
        from subqa.orchestrator import generate_suba_artifacts, generate_subq_artifacts
        from subqa.schema import normalize_hierarchy_config
        from util.utils import IndexSampler

        cases = (
            ("text", "multiple_choice", None),
            ("text", "open_ended", None),
            ("image", "multiple-choice", [object()]),
            ("image", "open_ended", [object(), object()]),
            ("video", "multiple_choice", [(object(), {"fps": 1})]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(
                Path(tmp),
                branching=(1, 2),
                suba_m=2,
                suba_k=1,
                batch_size=4,
            )
            cfg.dataset_cfg.prompt = {"add_base_prompt": True}
            hierarchy = normalize_hierarchy_config(cfg.runner_cfg)
            for index, (modality, question_type, vision) in enumerate(cases):
                with self.subTest(modality=modality, question_type=question_type):
                    qid = f"matrix-{index}"
                    current = sample(qid, data_type=modality, vision=vision)
                    current["question_type"] = question_type
                    current["candidate_list"] = ["CHOICE_LEAK_SENTINEL"]
                    current["gt_ans"] = "GT_LEAK_SENTINEL"
                    subq_model = CompleteDecompositionModel()
                    subq = generate_subq_artifacts(
                        [current], subq_model, hierarchy
                    )[qid]
                    suba_input = dict(current, **subq)
                    suba_model = AnsweringModel()
                    suba = generate_suba_artifacts(
                        [suba_input], suba_model, hierarchy
                    )[qid]

                    hierarchy_prompts = [
                        call["prompt"]
                        for call in subq_model.template_calls + suba_model.prompts
                    ]
                    self.assertTrue(hierarchy_prompts)
                    self.assertTrue(
                        all("CHOICE_LEAK_SENTINEL" not in prompt for prompt in hierarchy_prompts)
                    )
                    self.assertTrue(
                        all("GT_LEAK_SENTINEL" not in prompt for prompt in hierarchy_prompts)
                    )
                    for call in subq_model.template_calls + suba_model.prompts:
                        self.assertIs(vision, call["vision"])
                    self.assertEqual(1, len(subq["subq_list"]))
                    self.assertEqual(1, len(suba["suba_list"]))

                    refined_sample = dict(
                        current,
                        **subq,
                        **suba,
                        base_answer="base",
                    )
                    refined_prompts = get_refined_prompt(
                        refined_sample, cfg, IndexSampler(1, 1, 1)
                    )
                    self.assertEqual(1, len(refined_prompts))
                    self.assertIn(subq["subq_list"][0], refined_prompts[0])
                    self.assertIn(suba["suba_list"][0], refined_prompts[0])
                    self.assertNotIn("question 0.0", refined_prompts[0])

    def test_arbitrary_registry_model_runs_hierarchy_pipeline_without_name_branch(self):
        from model import MODEL_REGISTRY
        from pipeline import run

        instances = []

        class ArbitraryProtocolModel(CompleteDecompositionModel):
            def __init__(self, cfg):
                super().__init__()
                self.cfg = cfg
                instances.append(self)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = hierarchy_config(Path(tmp))
            cfg.model_cfg.model_name = "arbitrary-protocol-model"
            MODEL_REGISTRY["arbitrary-protocol-model"] = ArbitraryProtocolModel
            try:
                result = run(
                    cfg,
                    dataset_loader=lambda received: [sample("q0")],
                )
            finally:
                del MODEL_REGISTRY["arbitrary-protocol-model"]

        self.assertEqual(1, len(instances))
        self.assertEqual(["q0"], list(result))
        self.assertEqual(6, len(result["q0"]["subq_tree"]["nodes"]))

    def test_full_fake_multi_stage_uses_only_depth_one_projection_for_refined(self):
        from pipeline import run_multi_stage
        from tests.test_pipeline import SnapshotDataset, fake_vllm_output
        from util.path import get_output_filename

        class HybridModel(CompleteDecompositionModel):
            def __init__(self):
                super().__init__()
                self.cfg = None
                self.stage_prompts = []

            def apply_chat_template(self, prompt, vision=None, mm_uuids=None):
                stage = str(self.cfg.runner_cfg.mode)
                self.stage_prompts.append((stage, prompt))
                return {"prompt": prompt, "qid": mm_uuids, "vision": vision}

            def generate_results(self, prompts):
                results = []
                for rendered in prompts:
                    prompt = rendered["prompt"]
                    qid = rendered["qid"]
                    if "Decompose the target" in prompt:
                        if "Parent question:" in prompt:
                            parent = prompt.split("Parent question: ", 1)[1].splitlines()[0]
                            text = f"1. {parent} child 0?\n2. {parent} child 1?"
                        else:
                            text = f"1. {qid} root 0?"
                    elif "Refine an answer" in prompt:
                        target = prompt.split("Target question: ", 1)[1].splitlines()[0]
                        text = f"refined {target}"
                    else:
                        target = prompt.split("Target question: ", 1)[1].splitlines()[0]
                        text = f"direct {target}"
                    results.append(
                        GenerationResult(
                            text,
                            {"seq_ppl": 2.0, "token_min_prob": 0.5},
                        )
                    )
                return results

            def generate(self, prompts):
                stage = str(self.cfg.runner_cfg.mode)
                text = "base answer" if stage == "base" else "refined answer"
                return [fake_vllm_output(text) for _ in prompts]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = hierarchy_config(
                root,
                branching=(1, 2),
                suba_m=2,
                suba_k=1,
                batch_size=4,
            )
            cfg.runner_cfg.mode = "multi_stage"
            cfg.dataset_cfg.prompt = {"add_base_prompt": True}
            base_sample = sample("q0")
            base_sample.update(
                {
                    "candidate_list": ["first", "second"],
                    "gt_ans": "a",
                }
            )
            model = HybridModel()

            def dataset_loader(stage_cfg):
                mode = str(stage_cfg.runner_cfg.mode)
                run_dir = get_output_dir(stage_cfg)
                manifest_path = run_dir / MANIFEST_FILENAME
                current = dict(base_sample)
                dependencies = {}
                if manifest_path.is_file():
                    manifest = json.loads(manifest_path.read_text())
                else:
                    manifest = None
                if mode in ("suba", "refined"):
                    subq = json.loads(
                        (run_dir / get_output_filename(stage_cfg.for_stage("subq"))).read_text()
                    )["q0"]
                    current.update(subq)
                    dependencies["subq"] = manifest["stages"]["subq"]["generation_id"]
                if mode == "refined":
                    suba = json.loads(
                        (run_dir / get_output_filename(stage_cfg.for_stage("suba"))).read_text()
                    )["q0"]
                    base = json.loads(
                        (run_dir / get_output_filename(stage_cfg.for_stage("base"))).read_text()
                    )["q0"]
                    current.update(suba)
                    current.update(base)
                    dependencies["suba"] = manifest["stages"]["suba"]["generation_id"]
                    dependencies["base"] = manifest["stages"]["base"]["generation_id"]
                return SnapshotDataset([current], dependencies)

            results = run_multi_stage(
                cfg,
                model_factory=lambda received: model,
                dataset_loader=dataset_loader,
            )

            manifest = json.loads(
                (get_output_dir(cfg) / MANIFEST_FILENAME).read_text()
            )

        self.assertEqual(["subq", "suba", "base", "refined"], list(results))
        self.assertTrue(all(manifest["stages"][stage]["completed"] for stage in results))
        self.assertEqual(3, len(results["subq"]["q0"]["subq_tree"]["nodes"]))
        self.assertEqual(1, len(results["suba"]["q0"]["suba_list"]))
        refined_prompts = [prompt for stage, prompt in model.stage_prompts if stage == "refined"]
        self.assertTrue(refined_prompts)
        self.assertTrue(all("q0 root 0?" in prompt for prompt in refined_prompts))
        self.assertTrue(all("child 0?" not in prompt for prompt in refined_prompts))


if __name__ == "__main__":
    unittest.main()
