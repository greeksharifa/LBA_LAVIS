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
from prompt.prompts import get_subq_prompt, get_suba_prompt, get_base_prompt
from prompt.postprocess import format_vllm_outputs
from prompt.chat_template import apply_chat_template
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
        model = get_model(cfg)  # model = C2RFramework(cfg)

        # get tokenizer from VLLM for chat template
        # tokenizer = model.llm.get_tokenizer()
        
        vllm_prompts = []
        qids = []
        for data_iter_idx, sample in enumerate(dataset): # dataloader
            candidate_list = sample["candidate_list"] if "candidate_list" in sample else None
            vision = sample["vision"] if "vision" in sample and runner_cfg.mode != "blind" else None

            # generate prompt to vllm
            if runner_cfg.mode == "subq":
                text_prompt = get_subq_prompt(sample, cfg)
            elif runner_cfg.mode == "suba":
                text_prompt = get_suba_prompt(sample, cfg)
            elif runner_cfg.mode == "refined":
                raise NotImplementedError(f"Mode {runner_cfg.mode} not implemented")
                text_prompt = get_refined_prompt(sample, cfg)
            else: # base
                text_prompt = get_base_prompt(sample, cfg)

            # apply chat template to prompts and vision
            if isinstance(text_prompt, str):
                vllm_prompts.append(model.apply_chat_template(text_prompt, vision=vision, mm_uuids=sample["qid"]))
                qids.append(sample["qid"])
            elif isinstance(text_prompt, list):
                for prompt in text_prompt:
                    vllm_prompts.append(model.apply_chat_template(prompt, vision=vision, mm_uuids=sample["qid"]))
                    qids.append(sample["qid"])
            else:
                raise ValueError(f"Invalid text prompt: {text_prompt}")
            
            '''
                apply chat template to prompts
                text_prompts = apply_chat_template(text_prompts, tokenizer, model_cfg.model_type)
                prompts = model.apply_chat_template(text_prompts, visions=sample["vision"] if "vision" in sample else [None] * len(text_prompts))
                vllm_prompts.extend(prompts)

                add vision to prompts (for multimodal models)
                if "vision" in sample and sample["vision"] is not None:
                    """
                        {
                            'multi_modal_data': {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1770x1180 at 0x7F4F3FE5CC80>},
                            'multi_modal_uuids': {'image': 'uuid_0'},
                            'prompt': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>What is the content of this image?<|im_end|>\n<|im_start|>assistant\n'}
                        }
                    """
                    for text_prompt in text_prompts:
                        vllm_prompts.append({
                            'multi_modal_data': {dataset_cfg.data_type: sample["vision"]},
                            # 'multi_modal_uuids': {'image': 'uuid_0'},
                            'prompt': text_prompt
                        })
                    pass
            '''

        logger.info(f"len(vllm_prompts): {len(vllm_prompts)}")
        pprint(vllm_prompts[0], width=250)
        outputs = model.generate(vllm_prompts)

        outputs = format_vllm_outputs(runner_cfg.mode, outputs, qids, N)
    else: # visualize
        raise NotImplementedError(f"Mode {runner_cfg.mode} not implemented")

    json.dump(outputs, open(output_dir / f"{runner_cfg.mode}_outputs.json", "w"), indent=4)
    logger.info(f"Saved {runner_cfg.mode} outputs to {output_dir / f'{runner_cfg.mode}_outputs.json'}")

    import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()
