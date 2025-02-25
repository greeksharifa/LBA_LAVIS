import logging
import sys
import json
import math
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from config.configs import Config
from dataset import load_dataset
from model import get_model, C2RFramework
from util.logger import setup_logger, get_logger
from util.path import get_output_dir
from util.utils import setup_seeds, parse_args, IndexSampler, transpose_list#, print_sample
from prompt.prompts import get_subq_prompt
from prompt.postprocess import format_vllm_outputs
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


    # dataset
    dataset = load_dataset(cfg)#, output_dir)
    # dataloader = DataLoader(dataset, batch_size=cfg.model_cfg.batch_size, shuffle=False, collate_fn=dataset.collater)
    N, M, K = cfg.runner_cfg.N, cfg.runner_cfg.M, cfg.runner_cfg.K
    index_sampler = IndexSampler(N, M, K)


    logger.info(f"Output directory: {output_dir}")
    # run
    if runner_cfg.mode != "visualize":
        # model = get_model(cfg)
        model = C2RFramework(cfg)
        vllm_prompts = []
        qids = []
        for data_iter_idx, sample in enumerate(dataset): # dataloader
            qids.append(sample["qid"])
            # def _run():
            # bsz = len(sample["qid"])
            # candidate_list = sample["candidate_list"] if "candidate_list" in sample else [None] * bsz
            candidate_list = sample["candidate_list"] if "candidate_list" in sample else None
            if runner_cfg.mode == "blind":
                # sample["vision"] = [None] * bsz
                sample["vision"] = None

            # generate prompt to vllm
            if runner_cfg.mode == "subq":
                main_q = sample["main_q"]
                prompt = get_subq_prompt(runner_cfg.subqa_mode, main_q, dataset_cfg.data_type, N)
                vllm_prompts.append(prompt)
            elif runner_cfg.mode == "suba":
                subq_list = sample["subq_list"]
            else:
                raise NotImplementedError(f"Mode {runner_cfg.mode} not implemented")

            # if args.ignore_error:
            #     try:
            #         _run()
            #     except Exception as e:
            #         logger.error(f"Error processing qid: {sample['qid']}")
            #         logger.error(e)
            #         continue
            # else:
                _run()     

        outputs = model.generate(vllm_prompts)

        if runner_cfg.mode == "subq":
            sub_qs = format_vllm_outputs("subq", outputs, qids, N)
    else: # visualize
        raise NotImplementedError(f"Mode {runner_cfg.mode} not implemented")

    json.dump(sub_qs, open(output_dir / "sub_qs.json", "w"), indent=4)
    logger.info(f"Saved sub_qs to {output_dir / 'sub_qs.json'}")

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
