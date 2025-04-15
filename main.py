import logging
import sys
import json
import math
import seaborn as sns
import matplotlib.pyplot as plt

from collections import OrderedDict
from pathlib import Path
from pprint import pprint
from pandas.core.window import ewm
from tqdm import tqdm
        

import numpy as np
import torch
from torch.utils.data import DataLoader

from config.configs import Config
from dataset import load_dataset
from model import get_model, C2RFramework
from util.logger import setup_logger, get_logger
from util.path import get_output_dir, get_output_filename
from util.utils import setup_seeds, parse_args, IndexSampler, transpose_list, data_print#, print_sample
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
    if runner_cfg.visualize_only:
        # load samples_list
        samples_list = json.load(open(output_dir / f"{runner_cfg.mode}_samples_list.json", "r"))
    else:
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
        logger.info(f"vllm_prompts[0]: {data_print(vllm_prompts[0])}")
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
        samples_list = list(samples.values())
        samples_list.sort(key=lambda x: x["conf_base"], reverse=False)

    '''========================================== visualize =============================================='''
    # =========================================== visualize ==============================================
    # if not any(runner_cfg.mode in mode for mode in ["subq", "suba", "refined", "base", "CoT", "llm_judge"]):
    if runner_cfg.mode == "refined":
        # calculate accuracy of base answers
        base_acc = 0.0
        for sample in samples_list:
            base_score = dataset.get_score(sample["base_answer"], sample["gt_ans"], sample["question_type"], sample["main_q"])
            sample["base_score"] = base_score
            base_acc += base_score

            max_idx = np.argmax(sample["conf_refined"][runner_cfg.confidence_type])
            conf_refined_max = sample["conf_refined"][runner_cfg.confidence_type][max_idx]
            refined_answer_max = sample["refined_answer_list"][max_idx]
            refined_score = dataset.get_score(refined_answer_max, sample["gt_ans"], sample["question_type"], sample["main_q"])
            sample["refined_score"] = refined_score
            sample["conf_refined_max"] = conf_refined_max
            sample["refined_answer_max"] = refined_answer_max

            if "vision" in sample:
                del sample["vision"]

            
        base_acc /= len(samples_list)
        logger.info(f"Base accuracy: {base_acc:.4f}")

        # save samples_list
        json.dump(samples_list, open(output_dir / f"{runner_cfg.mode}_samples_list.json", "w"), indent=4)
        logger.info(f"Saved samples_list to {output_dir / f'{runner_cfg.mode}_samples_list.json'}")

        # t1_cands:  0.0, 0.1, 0.2, ..., 1.0
        # t2_cands: -1.0, -0.9, -0.8, ..., 1.0
        t1_cands = np.arange(0, 1, 0.1)  # [0.1 * x for x in range(11)]
        t2_cands = np.arange(-1, 1, 0.1) # [0.1 * x for x in range(-10, 11)]
        max_acc_matrix = np.zeros((len(t1_cands), len(t2_cands)))

        for t2_idx, t2_cand in enumerate(t2_cands):
            for t1_idx, t1_cand in enumerate(t1_cands):
                acc = 0.0
                for sample in samples_list:
                    conf_base = sample["conf_base"]
                    conf_refined_max = sample["conf_refined_max"]
                    if conf_base >= t1_cand:
                        acc += sample["base_score"]
                    elif conf_refined_max >= conf_base + t2_cand:
                        acc += sample["refined_score"]
                    else:
                        acc += sample["base_score"]

                max_acc_matrix[t1_idx, t2_idx] = acc / len(samples_list)

        refined_acc = 0.0
        t1, t2 = -1, -1
        for t1_idx in range(len(t1_cands)):
            for t2_idx in range(len(t2_cands)):
                if max_acc_matrix[t1_idx, t2_idx] > refined_acc:
                    refined_acc = max_acc_matrix[t1_idx, t2_idx]
                    t1, t2 = t1_cands[t1_idx], t2_cands[t2_idx]
        
        logger.info(f"Refined accuracy: {refined_acc:.4f} at t1: {t1}, t2: {t2}")

        # plot max_acc_matrix as heatmap seaborn
        plt.figure(figsize=(15, 15)) # plt.figure(figsize=(len(t1_cands), len(t2_cands)))
        # fontsize
        plt.rcParams.update({'font.size': 14})
        sns.heatmap(max_acc_matrix.T, annot=True, fmt=".4f", cmap="YlGnBu", cbar=True)
        plt.xlabel("t2")
        plt.ylabel("t1")
        plt.xticks(np.arange(0.5, len(t1_cands) + 0.5, 1), np.round(t1_cands, 1))
        plt.yticks(np.arange(0.5, len(t2_cands) + 0.5, 1), np.round(t2_cands, 1))
        plt.title(f"Max Accuracy Matrix ({runner_cfg.confidence_type})")
        plt.savefig(output_dir / f"max_acc_matrix_{runner_cfg.confidence_type}.png")
        logger.info(f"Saved max_acc_matrix to {output_dir / f'max_acc_matrix_{runner_cfg.confidence_type}.png'}")
        plt.close()

        # import pdb; pdb.set_trace()
                
        # refined_acc = 0.0
        # for t2_idx, t2_cand in enumerate(t2_cands):
        #     # for t1_cand in t1_cands:
        #     t1_idx = 0
        #     acc = base_acc
        #     for sample in samples_list:
        #         conf_base = sample["conf_base"]
        #         conf_refined_max = sample["conf_refined_max"]
        #         if conf_base >= t1_cands[t1_idx]:
        #             max_acc_matrix[t1_idx, t2_idx] = acc
        #             t1_idx += 1

        #         if conf_refined_max >= conf_base + t2_cand:
        #             acc += sample["refined_score"] - sample["base_score"]

        
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
    elif any(runner_cfg.mode in mode for mode in ["subq", "suba", "base"]):
        pass
    else:
        raise NotImplementedError(f"Visualization for {runner_cfg.mode} not implemented")



if __name__ == "__main__":
    main()
