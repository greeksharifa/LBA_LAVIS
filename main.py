"""Command-line entry point for single-stage and multi-stage inference."""

import logging

from config.configs import Config
from pipeline import run
from util.logger import get_logger, setup_logger
from util.path import get_output_dir
from util.utils import parse_args, setup_seeds


def main():
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)

    mode = str(cfg.runner_cfg.mode)
    path_cfg = cfg.for_stage("subq" if mode == "multi_stage" else mode)
    output_dir = get_output_dir(path_cfg)
    level = getattr(logging, str(cfg.runner_cfg.logging_level).upper())
    setup_logger(output_dir, level=level)
    logger = get_logger()
    cfg.set_logger(logger)
    cfg.pretty_print()
    logger.info("Output directory: %s", output_dir)

    run(cfg)


if __name__ == "__main__":
    main()
