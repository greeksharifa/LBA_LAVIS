import logging
import sys
import json
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from config.configs import Config
from dataset import load_dataset
# from model import get_framework
from util.logger import setup_logger, get_logger
from util.path import get_output_dir
from util.utils import setup_seeds, parse_args, IndexSampler, transpose_list#, print_sample
# from utils.misc import MetricLogger
# from visualize import visualize, visualize_base, record_num_tokens

def main():
    # config
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)

    runner_cfg = cfg.runner_cfg
    dataset_cfg = cfg.dataset_cfg
    model_cfg = cfg.model_cfg

    # output directory
    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    # temporary directory
    # temp_dir

    # logger
    level = getattr(logging, cfg.runner_cfg.logging_level)
    setup_logger(output_dir, level=level)
    logger = get_logger()
    cfg.set_logger(logger)
    cfg.pretty_print()

    logger.info(f"Output directory: {output_dir}")

    
    # dataset
    dataset = load_dataset(cfg)#, output_dir)
    dataloader = DataLoader(dataset, batch_size=cfg.model_cfg.batch_size, shuffle=False, collate_fn=dataset.collater)
    N, M, K = cfg.runner_cfg.N, cfg.runner_cfg.M, cfg.runner_cfg.K
    index_sampler = IndexSampler(N, M, K)



if __name__ == "__main__":
    main()
