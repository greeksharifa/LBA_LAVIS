from config.configs import Config
from pathlib import Path
from typing import Tuple

def get_output_dir(cfg: Config) -> Path:
    runner_cfg = cfg.runner_cfg
    dataset_cfg = cfg.dataset_cfg
    model_cfg = cfg.model_cfg

    if runner_cfg.mode == "subq" or runner_cfg.mode == "suba":
        output_dir = Path(runner_cfg.subqa_dir) / runner_cfg.subqa_mode
    else:
        if runner_cfg.mode == "CoT":
            output_dir = Path(runner_cfg.output_dir) / f"CoT_{runner_cfg.CoT_path}_paths"
        elif runner_cfg.mode == "llm_judge":
            output_dir = Path(runner_cfg.output_dir) / f"llm_judge_{runner_cfg.llm_judge_mode}"
        else:
            output_dir = Path(runner_cfg.output_dir) / runner_cfg.mode
    
    output_dir = output_dir / dataset_cfg.dataset_name / model_cfg.model_name
    return output_dir


def get_sub_qas_path(cfg: Config) -> Tuple[Path, Path]:
    runner_cfg = cfg.runner_cfg
    dataset_cfg = cfg.dataset_cfg
    model_cfg = cfg.model_cfg

    sub_qas_path = Path("subqa") / runner_cfg.subqa_mode / dataset_cfg.dataset_name / model_cfg.model_name
    return sub_qas_path / "sub_qs.json", sub_qas_path / "sub_as.json"