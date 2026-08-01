import json
import tempfile
import unittest
from pathlib import Path

from tests.test_artifacts import SamplingDataset, make_config
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
            write_dependency(subq_path, "subq", qids=("q0",))
            write_dependency(suba_path, "suba", qids=("q0",))
            write_dependency(base_path, "base", qids=("q0",))

            dataset = SamplingDataset(cfg)
            sample = dataset[0]

            self.assertEqual(["q0"], [ann["qid"] for ann in dataset.annotation])
            self.assertEqual(["subq value"], sample["subq_list"])
            self.assertEqual(["suba value"], sample["suba_list"])
            self.assertEqual("answer", sample["base_answer"])


if __name__ == "__main__":
    unittest.main()
