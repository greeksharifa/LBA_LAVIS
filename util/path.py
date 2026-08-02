import hashlib
import json
from config.configs import Config
from pathlib import Path
from typing import Tuple

from subqa.schema import normalize_hierarchy_config


def get_output_filename(cfg: Config) -> str:
    mode = cfg.runner_cfg.mode

    if mode == "subq" or mode == "suba":
        filename = mode
    elif mode == "CoT":
        filename = f"CoT_{cfg.runner_cfg.CoT_path}_paths"
    elif mode == "llm_judge":
        filename = f"llm_judge_{cfg.runner_cfg.llm_judge_mode}"
    else:
        filename = mode
    return f"{filename}_outputs.json"

def get_output_dir(cfg: Config) -> Path:
    runner_cfg = cfg.runner_cfg
    dataset_cfg = cfg.dataset_cfg
    model_cfg = cfg.model_cfg

    output_dir = (
        Path(runner_cfg.output_dir)
        / dataset_cfg.dataset_name
        / model_cfg.model_name
        / dataset_cfg.split
        / f"N={runner_cfg.N}_M={runner_cfg.M}_K={runner_cfg.K}"
    )
    hierarchy = normalize_hierarchy_config(runner_cfg)
    if hierarchy.enabled:
        canonical_json = json.dumps(
            hierarchy.to_manifest_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:12]
        output_dir /= f"D={hierarchy.depth}_H={digest}"
    return output_dir


def get_sub_qas_path(cfg: Config) -> Tuple[Path, Path]:
    output_dir = get_output_dir(cfg)
    return output_dir / "subq_outputs.json", output_dir / "suba_outputs.json"
