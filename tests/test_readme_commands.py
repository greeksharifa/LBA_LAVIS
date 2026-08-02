import re
import shlex
import unittest
from argparse import Namespace
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README = REPOSITORY_ROOT / "README.md"
REQUIREMENTS = REPOSITORY_ROOT / "requirements.txt"
GPU_WRAPPER = "/home/ywjang/.codex/bin/run_gpu.sh"
PYTHON = "/home/ywjang/miniconda3/envs/qwen2vl/bin/python"
RUN_ROOT = "output/MMMU/qwen2.5-vl-7b"
RUN_SIGNATURE = "N=4_M=2_K=4"


def shell_blocks(markdown):
    """Return shell-family fenced blocks without treating other fences as commands."""
    blocks = []
    current = None
    for line in markdown.splitlines():
        if current is None:
            if re.fullmatch(r"\s*```(?:bash|sh|shell)\s*", line, re.IGNORECASE):
                current = []
        elif re.fullmatch(r"\s*```\s*", line):
            blocks.append("\n".join(current))
            current = None
        else:
            current.append(line)
    if current is not None:
        raise AssertionError("unterminated shell code block in README")
    return blocks


def shell_commands(markdown):
    """Join backslash continuations and return tokenized non-comment commands."""
    commands = []
    for block in shell_blocks(markdown):
        pending = ""
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            pending = f"{pending} {line}".strip()
            if pending.endswith("\\"):
                pending = pending[:-1].rstrip()
                continue
            commands.append(shlex.split(pending))
            pending = ""
        if pending:
            commands.append(shlex.split(pending))
    return commands


def option_value(command, name):
    prefix = f"{name}="
    values = [token[len(prefix) :] for token in command if token.startswith(prefix)]
    if len(values) != 1:
        raise AssertionError(f"expected exactly one {prefix} option in {command!r}")
    return values[0]


class ReadmeCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = README.read_text(encoding="utf-8")
        cls.commands = shell_commands(cls.readme)
        cls.run_commands = [command for command in cls.commands if "main.py" in command]

    def test_gpu_commands_use_physical_gpu_wrapper_and_matching_tp(self):
        for command in self.commands:
            self.assertFalse(
                any(token.startswith("CUDA_VISIBLE_DEVICES=") for token in command),
                f"CUDA_VISIBLE_DEVICES must not be a top-level README command: {command!r}",
            )

        self.assertTrue(self.run_commands, "README must contain runnable main.py examples")
        for command in self.run_commands:
            self.assertEqual(GPU_WRAPPER, command[0], command)
            self.assertEqual(1, command.count("--"), command)
            separator = command.index("--")
            self.assertEqual("env", command[separator + 1], command)

            self.assertEqual(1, command.count(PYTHON), command)
            python_index = command.index(PYTHON)
            self.assertGreater(python_index, separator + 1, command)
            for assignment in (
                "HF_HOME=/home/ywjang/.cache/huggingface",
                "VLLM_USE_V1=0",
                "VLLM_WORKER_MULTIPROC_METHOD=spawn",
            ):
                self.assertEqual(1, command.count(assignment), command)
                self.assertGreater(command.index(assignment), separator + 1, command)
                self.assertLess(command.index(assignment), python_index, command)

            gpu_ids = command[1].split(",")
            self.assertTrue(set(gpu_ids) <= {"5", "6", "7", "8"}, command)
            self.assertEqual(len(gpu_ids), len(set(gpu_ids)), command)
            self.assertEqual(len(gpu_ids), int(option_value(command, "model.tensor_parallel_size")))

    def test_readme_documents_verified_runtime_and_pinned_requirements(self):
        for expected in (
            PYTHON,
            "torch 2.6.0+cu124",
            "vLLM 0.8.2",
            "transformers 4.55.2",
            "VLLM_USE_V1=0",
            "VLLM_WORKER_MULTIPROC_METHOD=spawn",
            "enforce_eager=true",
            "environment-specific",
            "model/vllm_config.py",
            "setdefault",
        ):
            self.assertIn(expected, self.readme)

        requirements = {
            line.strip()
            for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("torch==2.6.0", requirements)
        self.assertIn("vllm==0.8.2", requirements)
        self.assertIn("transformers==4.55.2", requirements)
        self.assertIn("omegaconf", requirements)

    def test_readme_documents_zero_cpu_swap_for_greedy_qwen_mmmu(self):
        for expected in (
            "Qwen2.5-VL MMMU",
            "greedy",
            "sampling_n=1",
            "swap_space=0",
            "TP=4",
            "64 GiB",
            "RAM exhaustion",
        ):
            self.assertIn(expected, self.readme)

    def test_every_documented_main_command_resolves_through_cpu_only_config(self):
        from config.configs import Config

        for command in self.run_commands:
            with self.subTest(command=command):
                main_index = command.index("main.py")
                self.assertEqual("--options", command[main_index + 1], command)
                options = command[main_index + 2 :]
                cfg = Config(
                    Namespace(cfg_path="config/default.yaml", options=options)
                )

                self.assertEqual(option_value(command, "runner.mode"), cfg.runner_cfg.mode)
                self.assertEqual(4, cfg.runner_cfg.N)
                self.assertEqual(2, cfg.runner_cfg.M)
                self.assertEqual(4, cfg.runner_cfg.K)
                self.assertEqual(
                    option_value(command, "runner.output_dir"),
                    cfg.runner_cfg.output_dir,
                )

                self.assertEqual("qwen2.5-vl-7b", cfg.model_cfg.model_name)
                self.assertEqual("Qwen/Qwen2.5-VL-7B-Instruct", cfg.model_cfg.model_id)
                self.assertEqual(
                    int(option_value(command, "model.tensor_parallel_size")),
                    cfg.model_cfg.tensor_parallel_size,
                )
                self.assertIs(True, cfg.model_cfg.enforce_eager)

                self.assertEqual("MMMU", cfg.dataset_cfg.dataset_name)
                self.assertEqual(
                    option_value(command, "dataset.split"),
                    cfg.dataset_cfg.split,
                )
                self.assertEqual(
                    int(option_value(command, "dataset.num_data")),
                    cfg.dataset_cfg.num_data,
                )
                self.assertTrue(cfg.dataset_cfg.ann_paths[cfg.dataset_cfg.split])

    def test_readme_documents_all_single_stage_and_multi_stage_modes(self):
        modes = {option_value(command, "runner.mode") for command in self.run_commands}
        self.assertTrue({"subq", "suba", "base", "refined", "multi_stage"} <= modes)

    def test_smoke_command_is_isolated_and_has_exact_profile(self):
        smoke_commands = [
            command
            for command in self.run_commands
            if "runner.output_dir=output/smoke-hierarchical" in command
        ]
        self.assertEqual(1, len(smoke_commands))
        command = smoke_commands[0]
        self.assertEqual("6", command[1])
        self.assertEqual("1", option_value(command, "model.tensor_parallel_size"))
        self.assertEqual("multi_stage", option_value(command, "runner.mode"))
        self.assertEqual("qwen2.5-vl-7b", option_value(command, "model.model_name"))
        self.assertEqual("MMMU", option_value(command, "dataset.dataset_name"))
        self.assertEqual("val", option_value(command, "dataset.split"))
        self.assertEqual("1", option_value(command, "dataset.num_data"))
        self.assertEqual("4", option_value(command, "runner.N"))
        self.assertEqual("2", option_value(command, "runner.M"))
        self.assertEqual("4", option_value(command, "runner.K"))
        self.assertEqual("2", option_value(command, "runner.subqa_depth"))
        self.assertEqual("[4,3]", option_value(command, "runner.branching_by_depth"))
        self.assertEqual("2", option_value(command, "runner.suba_M"))
        self.assertEqual("3", option_value(command, "runner.suba_K"))
        self.assertEqual(
            "token_min_prob", option_value(command, "runner.suba_confidence_type")
        )
        self.assertEqual(
            "true", option_value(command, "runner.condition_on_direct_suba")
        )

    def test_smoke_and_full_commands_use_cached_huggingface_offline_mode(self):
        offline_commands = [
            command
            for command in self.run_commands
            if "runner.mode=multi_stage" in command
            and (
                "runner.output_dir=output/smoke-hierarchical" in command
                or "dataset.num_data=-1" in command
            )
        ]
        self.assertEqual(3, len(offline_commands))
        for command in offline_commands:
            env_index = command.index("env")
            python_index = command.index(PYTHON)
            for assignment in ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
                self.assertEqual(1, command.count(assignment), command)
                self.assertGreater(command.index(assignment), env_index, command)
                self.assertLess(command.index(assignment), python_index, command)

        self.assertIn("already cached", self.readme)
        self.assertIn("first model download", self.readme)

    def test_full_dev_and_validation_commands_have_exact_profiles(self):
        full_commands = [
            command
            for command in self.run_commands
            if "runner.mode=multi_stage" in command
            and "dataset.num_data=-1" in command
        ]
        self.assertEqual(2, len(full_commands))
        self.assertEqual({"val", "test"}, {option_value(c, "dataset.split") for c in full_commands})
        for command in full_commands:
            self.assertEqual("5,6,7,8", command[1])
            self.assertEqual("4", option_value(command, "model.tensor_parallel_size"))
            self.assertEqual("qwen2.5-vl-7b", option_value(command, "model.model_name"))
            self.assertEqual("MMMU", option_value(command, "dataset.dataset_name"))
            self.assertEqual("4", option_value(command, "runner.N"))
            self.assertEqual("2", option_value(command, "runner.M"))
            self.assertEqual("4", option_value(command, "runner.K"))
            self.assertNotIn("runner.output_dir=output/smoke-hierarchical", command)

    def test_evaluation_command_uses_exact_dev_and_validation_run_paths(self):
        evaluation_commands = [
            command for command in self.commands if "scripts/evaluate_c2r.py" in command
        ]
        self.assertEqual(1, len(evaluation_commands))
        command = evaluation_commands[0]
        self.assertEqual(PYTHON, command[0])
        self.assertEqual(
            f"{RUN_ROOT}/val/{RUN_SIGNATURE}",
            command[command.index("--dev-run") + 1],
        )
        self.assertEqual(
            f"{RUN_ROOT}/test/{RUN_SIGNATURE}",
            command[command.index("--validation-run") + 1],
        )

    def test_readme_explains_artifact_and_evaluation_guards(self):
        for expected in (
            "run namespace",
            "run_manifest.json",
            "refined_samples.json",
            "generation_id",
            "provenance",
            "dev",
            "validation",
            "/home/ywjang/C2R",
            "historical",
            "identical configured `num_data`",
            "`-1` on both full splits",
            "D=<depth>_H=<hash>",
            "depth-1",
            "hierarchical Sub-QA",
        ):
            self.assertIn(expected, self.readme)


if __name__ == "__main__":
    unittest.main()
