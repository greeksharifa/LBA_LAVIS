import logging
import sys
import json
import math
from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from config.configs import Config
from dataset import load_dataset
from model import get_model, C2RFramework
from util.logger import setup_logger, get_logger
from util.path import get_output_dir, get_output_filename
from util.utils import setup_seeds, parse_args, IndexSampler, transpose_list#, print_sample
from prompt.prompts import get_subq_prompt, get_suba_prompt, get_base_prompt, get_refined_prompt
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

    samples = OrderedDict()
    # run
    if runner_cfg.mode != "visualize":
        model = get_model(cfg)  # model = C2RFramework(cfg)
        # tokenizer = model.llm.get_tokenizer()
        
        vllm_prompts = []
        qids = []
        for sample in tqdm(dataset): # dataloader
            samples[sample["qid"]] = sample
            candidate_list = sample["candidate_list"] if "candidate_list" in sample else None
            vision = sample["vision"] if "vision" in sample and runner_cfg.mode != "blind" else None

            # generate prompt to vllm
            if runner_cfg.mode == "subq":
                text_prompt = get_subq_prompt(sample, cfg)
            elif runner_cfg.mode == "suba":
                text_prompt = get_suba_prompt(sample, cfg)
            elif runner_cfg.mode == "refined":
                # raise NotImplementedError(f"Mode {runner_cfg.mode} not implemented")
                text_prompt = get_refined_prompt(sample, cfg, index_sampler)
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
            

        logger.info(f"len(vllm_prompts): {len(vllm_prompts)}")
        pprint(vllm_prompts[0], width=250)
        outputs = model.generate(vllm_prompts)

        outputs = format_vllm_outputs(runner_cfg.mode, outputs, qids, N)

        # samples와 outputs 통합 (qid 기준으로 merge)
        for qid, output_data in outputs.items():
            if qid in samples:
                samples[qid].update(output_data)

        filename = get_output_filename(cfg)

        json.dump(outputs, open(output_dir / filename, "w"), indent=4)
        logger.info(f"Saved {runner_cfg.mode} outputs to {output_dir / filename}")

        # import pdb; pdb.set_trace()

    '''========================================== visualize =============================================='''
    # =========================================== visualize ==============================================
    # if not any(runner_cfg.mode in mode for mode in ["subq", "suba", "refined", "base", "CoT", "llm_judge"]):
    if runner_cfg.mode == "refined":
        import pdb; pdb.set_trace()
        pass  # TODO: implement visualization for refined mode
        '''
            # pprint(samples["validation_Accounting_1"], width=350)
            {'base_answer': 'A. $6',
            'candidate_list': ['$6', '$7', '$8', '$9'],
            'conf_base': 0.995443458693748,
            'conf_refined': {'seq_ppl': [1.3394814791153304, 1.2675376521409543, 1.4199761679233758, 1.3337169031494736], 'token_min_prob': [0.5621765025686553, 0.6224593298742985, 0.6791786964925157, 0.5621765025686553]},
            'conf_suba_list': [0.8518304265903065, 0.4537173649616403, 0.49940782022797425, 0.4959059498075811, 0.4145557221948152],
            'conf_subq_list': 0.4997720632523633,
            'data_type': 'image',
            'gt_ans': 'b',
            'main_q': '<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?',
            'qid': 'validation_Accounting_1',
            'question_type': 'multiple-choice',
            'refined_answer_list': ['A', 'A', 'A', 'C'],
            'suba_list': ['The fixed manufacturing overhead cost per unit remains $6 regardless of production volume, so at 25,000 units, it is still $6 per unit.',
                        'Fixed manufacturing overhead cost remains constant in total regardless of production volume changes.',
                        'The total fixed manufacturing overhead cost at 25,000 units is $150,000 ($6 per unit × 25,000 units).',
                        'The total fixed manufacturing overhead cost at 30,000 units is $180,000 ($6 per unit × 30,000 units).',
                        'The per unit manufacturing overhead cost at 30,000 units is $8, calculated as $6 (fixed) + $2 (variable).'],
            'subq_list': ['What are the fixed manufacturing overhead costs per unit at 25,000 units of production?',
                        'How does the fixed manufacturing overhead cost behave when production increases from 25,000 to 30,000 units?',
                        'What is the total fixed manufacturing overhead cost at 25,000 units of production?',
                        'What is the total fixed manufacturing overhead cost at 30,000 units of production?',
                        'What is the per unit manufacturing overhead cost at 30,000 units of production, considering both fixed and variable components?'],
            'vision': [<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=733x237 at 0x7F96C75374A0>],
            'vpath': ['/data/MMMU/mmmu_images/validation/validation_Accounting_1_1.png']}
        '''
    else:
        raise NotImplementedError(f"Visualization for {runner_cfg.mode} not implemented")



if __name__ == "__main__":
    main()
