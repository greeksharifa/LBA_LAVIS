import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

from tests.test_artifacts import SamplingDataset, make_config
from subqa.schema import build_complete_subq_tree, project_depth_one_questions
from util.artifacts import MANIFEST_FILENAME, create_manifest, write_manifest
from util.path import get_output_dir, get_sub_qas_path


def write_annotations(root: Path, qids=("q0",)):
    annotations = [
        {
            "qid": qid,
            "main_q": f"question {qid}",
            "gt_ans": "answer",
            "question_type": "open_ended",
        }
        for qid in qids
    ]
    (root / "annotations.json").write_text(json.dumps(annotations))


def dependency_payload(stage, qids=("q0",)):
    value_key = "base_answer" if stage == "base" else f"{stage}_list"
    value = "answer" if stage == "base" else [f"{stage} value"]
    return {
        qid: {
            value_key: value,
            f"conf_{stage}": {"token_min_prob": 0.5},
        }
        for qid in qids
    }


def write_dependency(path: Path, stage, qids=("q0",)):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dependency_payload(stage, qids)))


def write_completed_manifest(cfg, qids=("q0",), stages=("subq", "suba", "base")):
    manifest = create_manifest(cfg, qids)
    for stage in stages:
        manifest["stages"][stage].update(
            {
                "completed": True,
                "state": "completed",
                "generation_id": f"{stage}-generation",
            }
        )
    write_manifest(get_output_dir(cfg) / MANIFEST_FILENAME, manifest)


def make_hierarchy_config(root: Path, mode: str):
    cfg = make_config(root, n=4, m=2, k=4)
    cfg.runner_cfg.subqa_depth = 2
    cfg.runner_cfg.branching_by_depth = [4, 3]
    cfg.runner_cfg.suba_M = 2
    cfg.runner_cfg.suba_K = 3
    cfg.runner_cfg.suba_confidence_type = "token_min_prob"
    cfg.runner_cfg.condition_on_direct_suba = True
    cfg.runner_cfg.subqa_max_nodes = 64
    cfg.runner_cfg.subqa_repair_attempts = 1
    cfg.runner_cfg.subqa_generation_batch_size = 64
    cfg.runner_cfg.subqa_schema_version = 2
    return cfg.for_stage(mode)


def hierarchy_subq_payload(qids=("q0",)):
    tree = build_complete_subq_tree(
        (4, 3), question_factory=lambda node_id: f"question {node_id}"
    )
    for node in tree["nodes"]:
        if node["depth"] < tree["max_depth"]:
            node["expansion_confidence"] = {
                "token_min_prob": 0.5,
                "seq_ppl": 2.0,
            }
    return {
        qid: {
            "subq_list": project_depth_one_questions(tree, expected_count=4),
            "conf_subq": {"token_min_prob": 0.5},
            "subq_tree": tree,
        }
        for qid in qids
    }


def hierarchy_suba_payload(tree, qids=("q0",)):
    answers_by_node = {}
    for node in tree["nodes"]:
        answer = f"answer {node['id']}"
        confidence = {"token_min_prob": 0.5, "seq_ppl": 2.0}
        direct = {
            "answer": answer,
            "confidence": confidence,
            "status": "valid",
            "failure_reason": None,
        }
        answers_by_node[node["id"]] = {
            "direct": direct,
            "candidates": [],
            "selected": {
                **direct,
                "source": (
                    "leaf_direct"
                    if node["depth"] == tree["max_depth"]
                    else "direct_fallback"
                ),
                "support_node_ids": [],
            },
        }
    for node in reversed(tree["nodes"]):
        if node["depth"] == tree["max_depth"]:
            continue
        entry = answers_by_node[node["id"]]
        supports = list(combinations(node["child_ids"], 2))[:3]
        entry["candidates"] = [
            {
                "answer": entry["direct"]["answer"],
                "confidence": dict(entry["direct"]["confidence"]),
                "status": entry["direct"]["status"],
                "failure_reason": entry["direct"]["failure_reason"],
                "support_node_ids": list(support_node_ids),
            }
            for support_node_ids in supports
        ]
        entry["selected"] = {
            **entry["candidates"][0],
            "confidence": dict(entry["candidates"][0]["confidence"]),
            "source": "confidence",
            "candidate_index": 0,
            "normalized_confidence": 0.5,
        }
    depth_one = [answers_by_node[str(index)]["selected"] for index in range(4)]
    return {
        qid: {
            "suba_list": [entry["answer"] for entry in depth_one],
            "conf_suba": {
                metric: [entry["confidence"][metric] for entry in depth_one]
                for metric in ("token_min_prob", "seq_ppl")
            },
            "answers_by_node": answers_by_node,
        }
        for qid in qids
    }


def write_payload(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


class DatasetDependencyTests(unittest.TestCase):
    def test_base_and_subq_need_no_prior_artifacts(self):
        for mode in ("base", "subq"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_annotations(root)
                cfg = make_config(root).for_stage(mode)

                sample = SamplingDataset(cfg)[0]

                self.assertEqual("q0", sample["qid"])
                self.assertNotIn("subq_list", sample)
                self.assertNotIn("suba_list", sample)
                self.assertNotIn("base_answer", sample)

    def test_suba_missing_subq_raises_with_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_config(root).for_stage("suba")
            subq_path, _ = get_sub_qas_path(cfg)
            write_completed_manifest(cfg, stages=("subq",))

            with self.assertRaises(FileNotFoundError) as raised:
                SamplingDataset(cfg)

            self.assertIn(str(subq_path), str(raised.exception))

    def test_refined_requires_each_dependency_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_config(root).for_stage("refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg)

            for expected_missing, available in (
                (subq_path, ()),
                (suba_path, ((subq_path, "subq"),)),
                (
                    base_path,
                    ((subq_path, "subq"), (suba_path, "suba")),
                ),
            ):
                with self.subTest(path=expected_missing):
                    for path, stage in available:
                        write_dependency(path, stage)
                    if expected_missing.exists():
                        expected_missing.unlink()

                    with self.assertRaises(FileNotFoundError) as raised:
                        SamplingDataset(cfg)
                    self.assertIn(str(expected_missing), str(raised.exception))

    def test_refined_rejects_missing_selected_qid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root, qids=("q0", "q1"))
            cfg = make_config(root).for_stage("refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg, qids=("q0", "q1"))
            write_dependency(subq_path, "subq", qids=("q0", "q1"))
            write_dependency(suba_path, "suba", qids=("q0",))
            write_dependency(base_path, "base", qids=("q0", "q1"))

            with self.assertRaises(KeyError) as raised:
                SamplingDataset(cfg)

            message = str(raised.exception)
            self.assertIn(str(suba_path), message)
            self.assertIn("q1", message)

    def test_refined_rejects_missing_required_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_config(root).for_stage("refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg)
            write_dependency(subq_path, "subq")
            write_dependency(suba_path, "suba")
            invalid_base = dependency_payload("base")
            del invalid_base["q0"]["conf_base"]
            base_path.parent.mkdir(parents=True, exist_ok=True)
            base_path.write_text(json.dumps(invalid_base))

            with self.assertRaises(KeyError) as raised:
                SamplingDataset(cfg)

            message = str(raised.exception)
            self.assertIn(str(base_path), message)
            self.assertIn("q0", message)
            self.assertIn("conf_base", message)

    def test_dependency_validation_uses_post_sampling_qids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root, qids=("q0", "q1", "q2"))
            cfg = make_config(root, num_data=1).for_stage("refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg, qids=("q0",))
            write_dependency(subq_path, "subq", qids=("q0",))
            write_dependency(suba_path, "suba", qids=("q0",))
            write_dependency(base_path, "base", qids=("q0",))

            dataset = SamplingDataset(cfg)
            sample = dataset[0]

            self.assertEqual(["q0"], [ann["qid"] for ann in dataset.annotation])
            self.assertEqual(
                {
                    "subq": "subq-generation",
                    "suba": "suba-generation",
                    "base": "base-generation",
                },
                getattr(dataset, "dependency_generations", None),
            )
            self.assertEqual(["subq value"], sample["subq_list"])
            self.assertEqual(["suba value"], sample["suba_list"])
            self.assertEqual("answer", sample["base_answer"])

    def test_suba_rejects_stale_subq_json_without_completed_manifest(self):
        scenarios = ("missing", "running", "incomplete")
        for scenario in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                write_annotations(root)
                cfg = make_config(root).for_stage("suba")
                subq_path, _ = get_sub_qas_path(cfg)
                write_dependency(subq_path, "subq")
                manifest_path = get_output_dir(cfg) / MANIFEST_FILENAME

                if scenario != "missing":
                    manifest = create_manifest(cfg, ["q0"])
                    if scenario == "running":
                        manifest["stages"]["subq"].update(
                            {
                                "completed": False,
                                "state": "running",
                                "generation_id": "failed-rerun",
                            }
                        )
                    write_manifest(manifest_path, manifest)

                with self.assertRaises((FileNotFoundError, ValueError)) as raised:
                    SamplingDataset(cfg)

                message = str(raised.exception)
                self.assertIn("subq", message)
                self.assertIn(str(manifest_path), message)

    def test_hierarchy_suba_receives_validated_subq_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "suba")
            subq_path, _ = get_sub_qas_path(cfg)
            write_completed_manifest(cfg, stages=("subq",))
            payload = hierarchy_subq_payload()
            write_payload(subq_path, payload)

            sample = SamplingDataset(cfg)[0]

            self.assertEqual(payload["q0"]["subq_tree"], sample["subq_tree"])
            self.assertEqual(payload["q0"]["subq_list"], sample["subq_list"])

    def test_hierarchy_suba_rejects_missing_tree_with_path_and_qid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "suba")
            subq_path, _ = get_sub_qas_path(cfg)
            write_completed_manifest(cfg, stages=("subq",))
            write_dependency(subq_path, "subq")

            with self.assertRaises((KeyError, ValueError)) as raised:
                SamplingDataset(cfg)

            message = str(raised.exception)
            self.assertIn(str(subq_path), message)
            self.assertIn("q0", message)
            self.assertIn("subq_tree", message)

    def test_hierarchy_suba_rejects_tree_that_violates_canonical_branching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "suba")
            subq_path, _ = get_sub_qas_path(cfg)
            write_completed_manifest(cfg, stages=("subq",))
            payload = hierarchy_subq_payload()
            payload["q0"]["subq_tree"] = build_complete_subq_tree((4, 2))
            write_payload(subq_path, payload)

            with self.assertRaises(ValueError) as raised:
                SamplingDataset(cfg)

            message = str(raised.exception)
            self.assertIn(str(subq_path), message)
            self.assertIn("q0", message)

    def test_hierarchy_subq_rejects_flat_question_projection_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "suba")
            subq_path, _ = get_sub_qas_path(cfg)
            write_completed_manifest(cfg, stages=("subq",))
            payload = hierarchy_subq_payload()
            payload["q0"]["subq_list"][0] = "not node zero"
            write_payload(subq_path, payload)

            with self.assertRaises(ValueError) as raised:
                SamplingDataset(cfg)

            message = str(raised.exception)
            self.assertIn(str(subq_path), message)
            self.assertIn("q0", message)
            self.assertIn("projection", message)

    def test_hierarchy_suba_rejects_missing_internal_expansion_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "suba")
            subq_path, _ = get_sub_qas_path(cfg)
            write_completed_manifest(cfg, stages=("subq",))
            payload = hierarchy_subq_payload()
            payload["q0"]["subq_tree"]["nodes"][0][
                "expansion_confidence"
            ] = None
            write_payload(subq_path, payload)

            with self.assertRaisesRegex(ValueError, "expansion_confidence"):
                SamplingDataset(cfg)

    def test_hierarchy_refined_validates_but_does_not_expose_tree_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg)
            subq_payload = hierarchy_subq_payload()
            tree = subq_payload["q0"]["subq_tree"]
            write_payload(subq_path, subq_payload)
            write_payload(suba_path, hierarchy_suba_payload(tree))
            write_dependency(base_path, "base")

            sample = SamplingDataset(cfg)[0]

            self.assertEqual(subq_payload["q0"]["subq_list"], sample["subq_list"])
            self.assertEqual(
                [f"answer {index}" for index in range(4)], sample["suba_list"]
            )
            self.assertEqual("answer", sample["base_answer"])
            self.assertNotIn("subq_tree", sample)
            self.assertNotIn("answers_by_node", sample)

    def test_hierarchy_refined_rejects_malformed_answers_by_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg)
            subq_payload = hierarchy_subq_payload()
            tree = subq_payload["q0"]["subq_tree"]
            suba_payload = hierarchy_suba_payload(tree)
            del suba_payload["q0"]["answers_by_node"]["0"]["selected"]
            write_payload(subq_path, subq_payload)
            write_payload(suba_path, suba_payload)
            write_dependency(base_path, "base")

            with self.assertRaises((KeyError, ValueError)) as raised:
                SamplingDataset(cfg)

            message = str(raised.exception)
            self.assertIn(str(suba_path), message)
            self.assertIn("q0", message)
            self.assertIn("answers_by_node", message)

    def test_hierarchy_refined_rejects_flat_answer_projection_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg)
            subq_payload = hierarchy_subq_payload()
            tree = subq_payload["q0"]["subq_tree"]
            suba_payload = hierarchy_suba_payload(tree)
            suba_payload["q0"]["suba_list"][0] = "not selected node zero"
            write_payload(subq_path, subq_payload)
            write_payload(suba_path, suba_payload)
            write_dependency(base_path, "base")

            with self.assertRaises(ValueError) as raised:
                SamplingDataset(cfg)

            message = str(raised.exception)
            self.assertIn(str(suba_path), message)
            self.assertIn("q0", message)
            self.assertIn("projection", message)

    def test_hierarchy_refined_rejects_non_finite_answer_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg)
            subq_payload = hierarchy_subq_payload()
            tree = subq_payload["q0"]["subq_tree"]
            suba_payload = hierarchy_suba_payload(tree)
            answers = suba_payload["q0"]["answers_by_node"]
            answers["0.0"]["direct"]["confidence"]["seq_ppl"] = float("inf")
            answers["0.0"]["selected"]["confidence"]["seq_ppl"] = float("inf")
            write_payload(subq_path, subq_payload)
            write_payload(suba_path, suba_payload)
            write_dependency(base_path, "base")

            with self.assertRaisesRegex(ValueError, "seq_ppl"):
                SamplingDataset(cfg)

    def test_hierarchy_refined_rejects_candidate_supported_by_invalid_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg)
            subq_payload = hierarchy_subq_payload()
            tree = subq_payload["q0"]["subq_tree"]
            suba_payload = hierarchy_suba_payload(tree)
            answers = suba_payload["q0"]["answers_by_node"]
            answers["0.0"]["direct"].update(
                {
                    "answer": None,
                    "status": "invalid",
                    "failure_reason": "blank answer",
                }
            )
            answers["0.0"]["selected"].update(
                {
                    "answer": None,
                    "status": "invalid",
                    "failure_reason": "blank answer",
                }
            )
            answers["0"]["candidates"] = [
                {
                    "answer": "candidate",
                    "confidence": {
                        "token_min_prob": 0.8,
                        "seq_ppl": 1.25,
                    },
                    "status": "valid",
                    "failure_reason": None,
                    "support_node_ids": ["0.0", "0.1"],
                }
            ]
            answers["0"]["selected"] = {
                **answers["0"]["candidates"][0],
                "confidence": dict(
                    answers["0"]["candidates"][0]["confidence"]
                ),
                "source": "confidence",
                "candidate_index": 0,
                "normalized_confidence": 0.8,
            }
            write_payload(subq_path, subq_payload)
            write_payload(suba_path, suba_payload)
            write_dependency(base_path, "base")

            with self.assertRaisesRegex(ValueError, "valid child"):
                SamplingDataset(cfg)

    def test_hierarchy_refined_rejects_truncated_candidate_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg)
            subq_payload = hierarchy_subq_payload()
            tree = subq_payload["q0"]["subq_tree"]
            suba_payload = hierarchy_suba_payload(tree)
            entry = suba_payload["q0"]["answers_by_node"]["0"]
            entry["candidates"] = []
            entry["selected"] = {
                **entry["direct"],
                "source": "direct_fallback",
                "support_node_ids": [],
            }
            write_payload(subq_path, subq_payload)
            write_payload(suba_path, suba_payload)
            write_dependency(base_path, "base")

            with self.assertRaisesRegex(ValueError, "candidate count"):
                SamplingDataset(cfg)

    def test_hierarchy_refined_rejects_non_maximum_selected_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_hierarchy_config(root, "refined")
            subq_path, suba_path = get_sub_qas_path(cfg)
            base_path = get_output_dir(cfg) / "base_outputs.json"
            write_completed_manifest(cfg)
            subq_payload = hierarchy_subq_payload()
            tree = subq_payload["q0"]["subq_tree"]
            suba_payload = hierarchy_suba_payload(tree)
            entry = suba_payload["q0"]["answers_by_node"]["0"]
            scores = (0.1, 0.9, 0.5)
            for candidate, score in zip(entry["candidates"], scores):
                candidate["confidence"]["token_min_prob"] = score
            entry["selected"] = {
                **entry["candidates"][0],
                "source": "confidence",
                "candidate_index": 0,
                "normalized_confidence": 0.1,
            }
            write_payload(subq_path, subq_payload)
            write_payload(suba_path, suba_payload)
            write_dependency(base_path, "base")

            with self.assertRaisesRegex(ValueError, "first maximum"):
                SamplingDataset(cfg)

    def test_depth_one_suba_accepts_legacy_subq_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_annotations(root)
            cfg = make_config(root).for_stage("suba")
            subq_path, _ = get_sub_qas_path(cfg)
            write_completed_manifest(cfg, stages=("subq",))
            write_dependency(subq_path, "subq")

            sample = SamplingDataset(cfg)[0]

            self.assertEqual(["subq value"], sample["subq_list"])
            self.assertNotIn("subq_tree", sample)


if __name__ == "__main__":
    unittest.main()
