import fcntl
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from config.configs import Config
from dataset.base_dataset import BaseDataset
from util.artifacts import (
    core_run_config,
    create_manifest,
    mark_stage_complete,
    mark_stage_started,
    validate_manifest,
    write_manifest,
)
from util.path import get_output_dir


def mark_stage_in_process(path, stage, generation_id, start, ready, finished):
    ready.set()
    start.wait()
    mark_stage_complete(Path(path), stage, generation_id)
    finished.set()


def make_config(root: Path, *, split="dev", n=5, m=2, k=8, num_data=-1):
    cfg = Config.__new__(Config)
    cfg.args = None
    cfg.config = OmegaConf.create(
        {
            "runner": {
                "mode": "multi_stage",
                "output_dir": str(root / "output"),
                "subqa_dir": str(root / "subqa"),
                "subqa_mode": "self",
                "confidence_type": "token_min_prob",
                "few_shot": "",
                "N": n,
                "M": m,
                "K": k,
            },
            "dataset": {
                "dataset_name": "FixtureDataset",
                "root_dir": str(root),
                "vis_root": ".",
                "split": split,
                "ann_paths": {split: ["annotations.json"]},
                "num_data": num_data,
                "question_type": "open_ended",
                "data_type": "text",
                "vqa_acc": False,
            },
            "model": {
                "model_name": "fixture-model",
                "model_id": "fixture/model-id",
            },
        }
    )
    return cfg


class SamplingDataset(BaseDataset):
    def load_annotation(self, ann_paths):
        with ann_paths[0].open("r") as handle:
            self.annotation.extend(json.load(handle))

    def __getitem__(self, index):
        ann = self.annotation[index]
        result = {
            "qid": ann["qid"],
            "main_q": ann["main_q"],
            "gt_ans": ann["gt_ans"],
            "question_type": ann["question_type"],
        }
        return self.load_additional_attr(ann, result)


class ArtifactTests(unittest.TestCase):
    def _start_stage(self, path, stage, generation_id, parent_generations):
        try:
            return mark_stage_started(
                path,
                stage,
                generation_id,
                parent_generations=parent_generations,
            )
        except TypeError as error:
            self.fail(f"mark_stage_started lacks parent lineage API: {error}")

    def _complete_stage(self, path, stage, generation_id, artifact_writer=None):
        try:
            return mark_stage_complete(
                path,
                stage,
                generation_id,
                artifact_writer=artifact_writer,
            )
        except TypeError as error:
            self.fail(f"mark_stage_complete lacks guarded writer API: {error}")

    @staticmethod
    def _completed_manifest(cfg, qids=("q0",)):
        manifest = create_manifest(cfg, qids)
        for stage in ("subq", "suba", "base", "refined"):
            manifest["stages"][stage].update(
                {
                    "completed": True,
                    "state": "completed",
                    "generation_id": f"{stage}-old",
                }
            )
        return manifest

    def test_core_run_config_records_active_annotation_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_config(root, split="dev", num_data=17)

            provenance = core_run_config(cfg)

            self.assertEqual(["annotations.json"], provenance["annotation_paths"])
            self.assertEqual(
                [str((root / "annotations.json").resolve())],
                provenance["annotation_paths_resolved"],
            )
            self.assertEqual(17, provenance["num_data"])

    def test_core_run_config_selects_split_specific_num_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), split="dev")
            cfg.dataset_cfg.num_data = {"dev": 13, "validation": 29}

            provenance = core_run_config(cfg)

            self.assertEqual(13, provenance["num_data"])

    def test_run_directory_separates_split_and_run_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dev = make_config(root, split="dev")
            test = make_config(root, split="test")
            other_n = make_config(root, n=6)
            other_m = make_config(root, m=3)
            other_k = make_config(root, k=9)

            paths = {
                get_output_dir(dev),
                get_output_dir(test),
                get_output_dir(other_n),
                get_output_dir(other_m),
                get_output_dir(other_k),
            }

            self.assertEqual(5, len(paths))
            self.assertEqual(
                root
                / "output"
                / "FixtureDataset"
                / "fixture-model"
                / "dev"
                / "N=5_M=2_K=8",
                get_output_dir(dev),
            )

    def test_for_stage_returns_independent_deep_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp))

            stage_cfg = cfg.for_stage("subq")
            stage_cfg.runner_cfg.N = 99
            stage_cfg.dataset_cfg.ann_paths.dev.append("other.json")

            self.assertIsInstance(stage_cfg, Config)
            self.assertEqual("subq", stage_cfg.runner_cfg.mode)
            self.assertEqual("multi_stage", cfg.runner_cfg.mode)
            self.assertEqual(5, cfg.runner_cfg.N)
            self.assertEqual(["annotations.json"], list(cfg.dataset_cfg.ann_paths.dev))

    def test_manifest_rejects_core_config_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = create_manifest(make_config(root), ["q0", "q1"])
            mismatched = make_config(root, split="test")

            with self.assertRaisesRegex(ValueError, "manifest config mismatch.*split"):
                validate_manifest(manifest, mismatched, ["q0", "q1"])

    def test_manifest_rejects_exact_qid_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp))
            manifest = create_manifest(cfg, ["q0", "q1"])

            with self.assertRaisesRegex(ValueError, "manifest qid mismatch"):
                validate_manifest(manifest, cfg, ["q0", "q2"])

    def test_manifest_write_and_stage_completion_are_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp))
            path = get_output_dir(cfg) / "run_manifest.json"
            write_manifest(path, create_manifest(cfg, ["q0"]))

            started = mark_stage_started(path, "subq", "subq-generation")
            mark_stage_complete(
                path,
                "subq",
                started["stages"]["subq"]["generation_id"],
            )

            with path.open("r") as handle:
                saved = json.load(handle)
            self.assertTrue(saved["stages"]["subq"]["completed"])
            self.assertEqual([], list(path.parent.glob(f".{path.name}.*.tmp")))

    def test_concurrent_stage_completions_preserve_both_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp))
            path = get_output_dir(cfg) / "run_manifest.json"
            write_manifest(path, create_manifest(cfg, ["q0"]))
            generations = {
                "subq": mark_stage_started(path, "subq", "subq-generation")[
                    "stages"
                ]["subq"]["generation_id"],
                "base": mark_stage_started(path, "base", "base-generation")[
                    "stages"
                ]["base"]["generation_id"],
            }
            lock_path = path.with_name(f".{path.name}.lock")
            context = multiprocessing.get_context("fork")
            start = context.Event()
            ready = [context.Event(), context.Event()]
            finished = [context.Event(), context.Event()]
            processes = [
                context.Process(
                    target=mark_stage_in_process,
                    args=(
                        path,
                        stage,
                        generations[stage],
                        start,
                        ready[index],
                        finished[index],
                    ),
                )
                for index, stage in enumerate(("subq", "base"))
            ]

            with lock_path.open("a+") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    for process in processes:
                        process.start()
                    for event in ready:
                        self.assertTrue(event.wait(5))
                    start.set()
                    completed_while_locked = [event.wait(0.5) for event in finished]
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                    for process in processes:
                        process.join(5)
                        if process.is_alive():
                            process.terminate()
                            process.join(5)

            self.assertEqual([False, False], completed_while_locked)
            self.assertEqual([0, 0], [process.exitcode for process in processes])
            saved = json.loads(path.read_text())
            self.assertTrue(saved["stages"]["subq"]["completed"])
            self.assertTrue(saved["stages"]["base"]["completed"])

    def test_stage_start_invalidates_exact_dependents_and_preserves_base(self):
        cases = (
            ("subq", {}, {"suba", "refined"}, {"base"}),
            ("suba", {"subq": "subq-old"}, {"refined"}, {"subq", "base"}),
            ("base", {}, {"refined"}, {"subq", "suba"}),
            (
                "refined",
                {
                    "subq": "subq-old",
                    "suba": "suba-old",
                    "base": "base-old",
                },
                set(),
                {"subq", "suba", "base"},
            ),
        )
        for stage, parents, invalidated, preserved in cases:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                cfg = make_config(Path(tmp))
                path = get_output_dir(cfg) / "run_manifest.json"
                write_manifest(path, self._completed_manifest(cfg))

                manifest = self._start_stage(
                    path,
                    stage,
                    f"{stage}-new",
                    parents,
                )

                self.assertEqual("running", manifest["stages"][stage]["state"])
                self.assertEqual(parents, manifest["stages"][stage]["parent_generations"])
                for dependent in invalidated:
                    status = manifest["stages"][dependent]
                    self.assertFalse(status["completed"])
                    self.assertEqual("invalidated", status["state"])
                    self.assertNotIn("generation_id", status)
                for independent in preserved:
                    self.assertTrue(manifest["stages"][independent]["completed"])
                    self.assertEqual(
                        f"{independent}-old",
                        manifest["stages"][independent]["generation_id"],
                    )

    def test_dependent_start_requires_exact_current_parent_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp))
            path = get_output_dir(cfg) / "run_manifest.json"
            write_manifest(path, self._completed_manifest(cfg))

            for parents in (
                None,
                {},
                {"subq": ""},
                {"subq": "subq-old", "base": "base-old"},
            ):
                with self.subTest(parents=parents):
                    kwargs = {} if parents is None else {"parent_generations": parents}
                    try:
                        with self.assertRaisesRegex(ValueError, "parent generations"):
                            mark_stage_started(path, "suba", "suba-new", **kwargs)
                    except TypeError as error:
                        self.fail(
                            f"mark_stage_started lacks parent lineage API: {error}"
                        )

            snapshot = {"subq": "subq-old"}
            mark_stage_started(path, "subq", "subq-new")
            with self.assertRaisesRegex(ValueError, "parent generation.*subq"):
                self._start_stage(path, "suba", "suba-new", snapshot)

    def test_invalidated_downstream_cannot_complete_old_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp))
            path = get_output_dir(cfg) / "run_manifest.json"
            write_manifest(path, self._completed_manifest(cfg))
            self._start_stage(
                path,
                "suba",
                "suba-running",
                {"subq": "subq-old"},
            )

            mark_stage_started(path, "subq", "subq-new")

            with self.assertRaisesRegex(ValueError, "generation changed"):
                mark_stage_complete(path, "suba", "suba-running")

    def test_completion_requires_generation_and_guards_artifact_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_config(root)
            path = get_output_dir(cfg) / "run_manifest.json"
            artifact = root / "artifact.json"
            write_manifest(path, create_manifest(cfg, ["q0"]))

            with self.assertRaisesRegex(ValueError, "generation_id"):
                mark_stage_complete(path, "base")

            mark_stage_started(path, "base", "base-old")
            mark_stage_started(path, "base", "base-new")
            self._complete_stage(
                path,
                "base",
                "base-new",
                artifact_writer=lambda: artifact.write_text("new"),
            )
            with self.assertRaisesRegex(ValueError, "generation changed"):
                self._complete_stage(
                    path,
                    "base",
                    "base-old",
                    artifact_writer=lambda: artifact.write_text("old"),
                )
            self.assertEqual("new", artifact.read_text())

    def test_writer_failure_leaves_stage_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp))
            path = get_output_dir(cfg) / "run_manifest.json"
            write_manifest(path, create_manifest(cfg, ["q0"]))
            mark_stage_started(path, "base", "base-generation")

            def fail_write():
                raise RuntimeError("simulated artifact write failure")

            with self.assertRaisesRegex(RuntimeError, "artifact write failure"):
                self._complete_stage(
                    path,
                    "base",
                    "base-generation",
                    artifact_writer=fail_write,
                )

            status = json.loads(path.read_text())["stages"]["base"]
            self.assertFalse(status["completed"])
            self.assertEqual("running", status["state"])

    def test_manifest_rejects_malformed_container_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp))
            valid = create_manifest(cfg, ["q0"])
            malformed_manifests = (
                [],
                {**valid, "config": []},
                {**valid, "qids": None},
                {**valid, "qids": "q0"},
            )

            for manifest in malformed_manifests:
                with self.subTest(manifest=manifest):
                    with self.assertRaises(ValueError):
                        validate_manifest(manifest, cfg, ["q0"])

    def test_num_data_manifest_uses_post_sampling_qids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = [
                {
                    "qid": f"q{index}",
                    "main_q": f"question {index}",
                    "gt_ans": "answer",
                    "question_type": "open_ended",
                }
                for index in range(3)
            ]
            (root / "annotations.json").write_text(json.dumps(annotations))
            cfg = make_config(root, num_data=1)
            stage_cfg = cfg.for_stage("base")

            dataset = SamplingDataset(stage_cfg)
            selected_qids = [ann["qid"] for ann in dataset.annotation]
            manifest = create_manifest(stage_cfg, selected_qids)

            self.assertEqual(["q0"], manifest["qids"])
            validate_manifest(manifest, stage_cfg, selected_qids)


if __name__ == "__main__":
    unittest.main()
