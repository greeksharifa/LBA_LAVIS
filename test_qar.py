"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import argparse
import os
import json
import random
import easydict
from time import sleep

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common import dist_utils
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


class Colors:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    
    RESET = '\033[0m'
    
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    BRIGHT_END = '\033[0m'


def _get_model(finetuned, prompt_type):
    model_arch = "blip2_vicuna_instruct_qar" if prompt_type == "reasoner" else "blip2_vicuna_instruct"
    args = easydict.EasyDict({
        "cfg_path": "instructBLIP_FT_vicuna7b_qar.yaml",
        "options": [
            "model.load_finetuned=True",
            f"model.finetuned={finetuned}",
            f"model.arch={model_arch}"
        ]
    })
    cfg = Config(args)
    
    if prompt_type == "reasoner":
        init_distributed_mode(cfg.run_cfg)

        # set after init_distributed_mode() to only log on master.
        setup_logger()
        
        setup_seeds(cfg)
        
        cfg.pretty_print()
        
    
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    
    print(f"Loading finetuned model from {finetuned}")
    
    return model, (task, datasets, cfg)
    


def main():
    _prompt_file_path = "prompts.json"
    prompts = json.load(open(_prompt_file_path, "r"))
    # logging.warning(Colors.YELLOW + "prompts: \n" + json.dumps(prompts, indent=4) + Colors.RESET)

    
    # logging.info('in captioning_base branch')
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    
    best_model_dir = "lavis/ywjang_output/best_models/"
    
    finetuned = {
        # "Questioner": f"{best_model_dir}20230810150_Questioner_MultipleSubQ_epoch10/checkpoint_best.pth",
        "Questioner": f"{best_model_dir}20230812121_Questioner_MultipleSubQ_epoch50/checkpoint_best.pth",
        
        # "Answerer": f"{best_model_dir}20230810124_Answerer/checkpoint_best.pth",
        "Answerer": "/home/ywjang/models/instruct_blip_vicuna7b_trimmed.pth",
        
        # "Reasoner": "/home/ywjang/models/instruct_blip_vicuna7b_trimmed.pth",
        # "Reasoner": f"{best_model_dir}20230812163_Reasoner_epoch50/checkpoint_best.pth",
        "Reasoner": f"{best_model_dir}20230810074_Reasoner/checkpoint_best.pth",
        # "Reasoner": f"{best_model_dir}20230812113_Reasoner_simpleprompt_epoch10/checkpoint_best.pth",
    }
    
    Reasoner, (task, datasets, cfg) = _get_model(
        finetuned=finetuned['Reasoner'],
        prompt_type="reasoner"
    )
    
    Reasoner.set_prompts()
    print(Colors.RED + "set_prompts done." + Colors.RESET)
    print(Colors.BRIGHT_GREEN + "finetuned: " + json.dumps(finetuned, indent=4) + Colors.RESET)
    print(Colors.BRIGHT_YELLOW + "prompts: \n" + json.dumps(prompts, indent=4) + Colors.RESET)

    Questioner, _ = _get_model(
        finetuned=finetuned['Questioner'],
        prompt_type="questioner"
    )
    Answerer, _ = _get_model(
        finetuned=finetuned['Answerer'],
        prompt_type="answerer"
    )
    
    Reasoner.set_questioner_and_answerer(Questioner, Answerer, finetuned)
    logging.info(Colors.YELLOW + "set_questioner_and_answerer done." + Colors.RESET)

    # model = model.to("cuda:0")
    logging.info(f"Reasoner device: {Reasoner.device}")

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=Reasoner, datasets=datasets
    )
    runner.evaluate(skip_reload=True)
    
    # set after init_distributed_mode() to only log on master.
    # setup_logger(runner.output_dir)
    # if dist_utils.is_main_process():
    # while True:
    #     print(Colors.BRIGHT_MAGENTA + "new prompt: enter 'new'. quit: 'quit'. " + Colors.RESET)
    #     inputs = input()
    #     if inputs == "quit":
    #         break
    #     elif inputs == "new":
    #         new_prompt = dict(prompts["Reasoner"])
    #
    #         for prompt_type in ["init_prompt", "pair_prompt", "final_prompt"]:
    #             print(Colors.BRIGHT_MAGENTA + f"if you want to pass, just enter 'pass'. Enter {prompt_type}: " + Colors.RESET)
    #             _prompt = input()
    #             if _prompt != "pass":
    #                 new_prompt.update({prompt_type: _prompt})
    #
    #         print(Colors.BRIGHT_YELLOW + "new_prompt: " + json.dumps(new_prompt, indent=4) + Colors.RESET)
    #         print(Colors.BRIGHT_MAGENTA + "Are you sure? Enter 'yes' or 'no'." + Colors.RESET)
    #         _yes_or_no = input()
    #         if _yes_or_no == "yes":
    #             print(Colors.YELLOW + "Evaluation Start." + Colors.RESET)
    #             Reasoner.set_new_reasoner_prompt(new_prompt)
    #             runner.evaluate(skip_reload=True)
    #         else:
    #             print(Colors.YELLOW + "Evaluation Canceled." + Colors.RESET)
    #             pass
    
    # print()
    # while True:
    #     N = 5
    #     for i in range(N):
    #         print(f"\r{N-i} / {N}", end="")
    #         sleep(1)
    #     if Reasoner.update_reasoner_prompt():
    #         print()
    #         runner.evaluate(skip_reload=True)
    
    print(Colors.BRIGHT_GREEN + "finetuned: " + json.dumps(finetuned, indent=4) + Colors.RESET)
    print(Colors.BRIGHT_YELLOW + "prompts: \n" + json.dumps(prompts, indent=4) + Colors.RESET)
    # print(Colors.BRIGHT_YELLOW + "prompts: \n" + json.dumps(prompts["Reasoner"], indent=4) + Colors.RESET)
    print(Colors.YELLOW + "Evaluation finished." + Colors.RESET)


if __name__ == "__main__":
    main()
