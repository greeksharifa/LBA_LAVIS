import argparse
import os
import json
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime
from pprint import pprint
from collections import OrderedDict
import PIL
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from configs.config import Config
# from dataset.VQA_Introspect import VQAIntrospectDataset
from dataset.base_dataset import load_dataset, get_text_input, get_sevila_input, get_train_examplar
from models.model import Decomposer, Recomposer

from utils.misc import SmoothedValue, MetricLogger
from utils.visualize import visualize, sample_print


def setup_seeds(config):
    seed = config.runner_cfg.seed # + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='LBA method')
    parser.add_argument("--cfg-path", default='configs/runner.yaml', help="path to configuration file.")
    # verbose
    parser.add_argument('--verbose', action='store_true', help='verbose')
    parser.add_argument('--visualize_high_confidence', default=True, type=bool, help='select high confidence')
    
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)
    if cfg.runner_cfg.visualize:
        args.cfg_path = os.path.join(cfg.runner_cfg.output_dir, 'config.yaml')
        cfg = Config(args)
    os.environ['HF_HOME'] = cfg.runner_cfg.HF_HOME
    print('cfg:\n', cfg._convert_node_to_json(cfg.config), sep='')

    if not cfg.runner_cfg.visualize:
        output_dir = os.path.join(cfg.runner_cfg.output_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(output_dir)
        OmegaConf.save(config=cfg.config, f=os.path.join(output_dir, "config.yaml"))
    else:
        print(type(cfg.runner_cfg.output_dir), cfg.runner_cfg.output_dir)
        # output_dir = os.path.join('output/', cfg.runner_cfg.output_dir)
        output_dir = cfg.runner_cfg.output_dir
    
    s = datetime.now()
    
    if cfg.runner_cfg.sub_mode == "subqa":
        n_supple = cfg.runner_cfg.num_sub_qa_generate
    elif cfg.runner_cfg.sub_mode == "frame_sampling":
        n_supple = cfg.runner_cfg.num_frame_sampling
    else:
        n_supple = 0
    
    dataset = load_dataset(cfg.datasets_cfg, n_supple=n_supple)
    dataloader = DataLoader(dataset, batch_size=cfg.runner_cfg.batch_size,
                            shuffle=False, collate_fn=dataset.collater)
    print('dataset loading time : ', datetime.now()-s)
    
    if not cfg.runner_cfg.visualize:
        s = datetime.now()
        recomposer = Recomposer(cfg, device="cuda:0")
        
        if cfg.runner_cfg.decomposer_name == "self":
            decomposer = recomposer # Recomposer(cfg, device="cuda:1") # 
        else:
            decomposer = Decomposer(cfg, device="cuda:1")
            
        if cfg.runner_cfg.recomposer_name == "sevila":
            answerer = Recomposer(cfg, device=f"cuda:{torch.cuda.device_count() - 1}", answerer=True)
        else:
            answerer = recomposer
        print('model loading time : ', datetime.now()-s)

        s = datetime.now()
        total_base_match, total_cnt = 0., 0

        metric_logger = MetricLogger(delimiter="  ")
        print_freq = max(1, int(len(dataloader) / cfg.runner_cfg.print_freq))
        # print('print_freq:', print_freq)

        try:
            examplar = get_train_examplar(cfg.datasets_cfg)
        except:
            examplar = ""
        print('examplar:', examplar)
        results = []
        wrong2right, right2wrong = 0, 0
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header='')):
            if args.verbose and data_iter_step == 0:
                print('batch:')
                for k, v in batch.items():
                    if hasattr(v, "shape"):
                        print(f'{k}: {v.shape}')
                    elif isinstance(v, list) and hasattr(v[0], "shape"):
                        print(f'{k}: {len(v)} {v[0].shape}')
                    elif isinstance(v, list) and isinstance(v[0], list) and hasattr(v[0][0], "shape"):
                        print(f'{k}: {len(v)} {len(v[0])} {v[0][0].shape}')
                    elif isinstance(v, list) and isinstance(v[0], list) and isinstance(v[0][0], list) and hasattr(v[0][0][0], "shape"):
                        print(f'{k}: {len(v)} {len(v[0])} {len(v[0][0])} {v[0][0][0].shape}')
                    elif k != "candidate_list":
                        print(f'{k}: {v}')
                # pprint(batch, width=300)

            bsz = len(batch['vision'])
            vision = batch['vision']
            
            """##############################  Baseline Inference   ##############################"""    
            if cfg.runner_cfg.recomposer_name == "sevila":
                # return list of dict, not list of str
                text_inputs = get_sevila_input("default", batch=batch)
            elif cfg.datasets_cfg.data_type == "videos":
                text_inputs = get_text_input("default_video", main_questions=batch['text_input'], candidate_lists=batch['candidate_list'])
            else:                          # "images"
                text_inputs = get_text_input("default_image", main_questions=batch['text_input'])
            text_outputs_base, confidences_base = recomposer(vision, text_inputs)
            print(f'{data_iter_step:5d}/{len(dataloader)} \t base: ', text_outputs_base[0], ' | ', confidences_base[0])

            gt_answers = batch['gt_ans']  # vqa: list[bsz, 10], videoqa: list[bsz]
            if cfg.runner_cfg.recomposer_name != "sevila":
                gt_answers = [dataset.answer_mapping(ans) for ans in gt_answers]
                    
            acc_base = dataset.get_accuracy(text_outputs_base, gt_answers)

            total_base_match += sum(acc_base)
            total_cnt += bsz
            metric_logger.update(n=bsz, base_acc=sum(acc_base)/bsz)
            metric_logger.update(n=bsz, total_base_acc=total_base_match/total_cnt)
            
            print("batch['text_input'][0]:", batch['text_input'][0])
            
            """############################## Decompose & Recompose ##############################"""
            sub_questions_list, sub_answers_list, text_outputs_lba_list, confidences_lba_list = [], [], [], []
            
            if cfg.runner_cfg.sub_mode == "subqa":
                for i in range(cfg.runner_cfg.num_sub_qa_generate):
                    # generating sub_questions
                    if cfg.runner_cfg.vision_supple:# and i >= 1:
                        vision = []
                        for b in range(bsz):
                            vision.append(batch['vision_supple'][b][i])
                            # vision.append(batch['vision_supple'][b][i-1])
                    text_inputs = get_text_input("decomposer", main_questions=batch['text_input'])
                    if cfg.runner_cfg.decomposer_name == "self":  # Image+Text, BLIP-2
                        sub_questions, _ = decomposer(vision, text_inputs, generate_sub_q=True)
                    else:                               # Only Text, flan-t5
                        sub_questions = decomposer(text_inputs)
                    sub_questions_list.append(sub_questions)
                    
                    
                    # generating sub_answers
                    text_inputs = get_text_input("sub_answer", sub_questions=sub_questions)
                    sub_answers, _ = answerer(vision, text_inputs)
                    sub_answers_list.append(sub_answers)
                    
                    # generating recomposed_answers
                    if cfg.runner_cfg.recomposer_name == "sevila":
                        text_inputs = get_sevila_input("recomposer", 
                                                    batch=batch, 
                                                    sub_questions=sub_questions, 
                                                    sub_answers=sub_answers)
                    elif cfg.datasets_cfg.data_type == "videos":
                        text_inputs = get_text_input("recomposer_video", 
                                                    main_questions=batch['text_input'], 
                                                    sub_questions=sub_questions, 
                                                    sub_answers=sub_answers,
                                                    candidate_lists=batch['candidate_list'],
                                                    examplar=examplar,
                                                    train_recomposer_examplar=cfg.runner_cfg.train_recomposer_examplar)
                    else:                          # "images"
                        text_inputs = get_text_input("recomposer_image", 
                                                    main_questions=batch['text_input'], 
                                                    sub_questions=sub_questions, 
                                                    sub_answers=sub_answers)
                    
                    if cfg.runner_cfg.debug:
                        print('sub_questions text_inputs:', text_inputs)
                        print('sub_answers text_inputs:', text_inputs)
                        print('recomposer_video text_inputs:', text_inputs)
                    text_outputs_lba, confidences_lba = recomposer(vision, text_inputs)
                    text_outputs_lba_list.append(text_outputs_lba)
                    confidences_lba_list.append(confidences_lba)
                    
                    if args.verbose:
                        # print(f'sub_QA: {sub_questions} -> {sub_answers}. LBA: {text_outputs_lba} | {[f"{float(x):.6f}" for x in confidences_lba]}')
                        print(f'sub_QA: {sub_questions[0]} -> {sub_answers[0]}. LBA: {text_outputs_lba[0]} | {confidences_lba[0]:.6f}')
            
            elif cfg.runner_cfg.sub_mode == "description":
                descriptions_list = []
                text_inputs = ["Describe the video in detail. What is happening in the video?" for _ in range(bsz)]
                descriptions, _ = recomposer(vision, text_inputs, generate_sub_q=True)
                descriptions_list.append(descriptions)
                
                # generating recomposed_answers
                if cfg.runner_cfg.recomposer_name == "sevila":
                    text_inputs = get_sevila_input("recomposer", 
                                                batch=batch, 
                                                sub_questions=sub_questions, 
                                                sub_answers=sub_answers)
                elif cfg.datasets_cfg.data_type == "videos":
                    text_inputs = get_text_input("recomposer_video_description", 
                                                main_questions=batch['text_input'], 
                                                descriptions=descriptions,
                                                candidate_lists=batch['candidate_list'],
                                                examplar=examplar,
                                                train_recomposer_examplar=cfg.runner_cfg.train_recomposer_examplar)
                else:                          # "images"
                    raise NotImplementedError("description mode is not implemented for images")
                
                if cfg.runner_cfg.debug:
                    print('sub_questions text_inputs:', text_inputs)
                    print('sub_answers text_inputs:', text_inputs)
                    print('recomposer_video text_inputs:', text_inputs)
                text_outputs_lba, confidences_lba = recomposer(vision, text_inputs)
                # sub_questions_list = descriptions_list
                # sub_answers_list = 
                text_outputs_lba_list.append(text_outputs_lba)
                confidences_lba_list.append(confidences_lba)
                
                if args.verbose:
                    # print(f'sub_QA: {sub_questions} -> {sub_answers}. LBA: {text_outputs_lba} | {[f"{float(x):.6f}" for x in confidences_lba]}')
                    print(f'Description: {descriptions[0]}. LBA: {text_outputs_lba[0]} | {confidences_lba[0]:.6f}')
            
            elif cfg.runner_cfg.sub_mode == "frame_sampling":
                for i in range(cfg.runner_cfg.num_sub_qa_generate):
                    
                    if cfg.runner_cfg.vision_supple:# and i >= 1:
                        vision = []
                        for b in range(bsz):
                            vision.append(batch['vision_supple'][b][i])
                            # vision.append(batch['vision_supple'][b][i-1])
                            
                    # sub_question generate 및 select text_outputs_lba by argmax confidence 재활용
                    text_inputs = get_text_input("sub_answer", sub_question=batch['text_input'])
                    text_outputs_lba, confidences_lba = answerer(vision, text_inputs)
                    text_outputs_lba_list.append(text_outputs_lba)
                    confidences_lba_list.append(confidences_lba)
                    
            def _convert_nested_list(lists):
                return [
                    [inner_list[i] for inner_list in lists] 
                    for i in range(len(lists[0]))
                ]
                
            # sub_questions_list = _convert_nested_list(sub_questions_list)
            # sub_answers_list = _convert_nested_list(sub_answers_list)
            text_outputs_lba_list = _convert_nested_list(text_outputs_lba_list)
            confidences_lba_list = _convert_nested_list(confidences_lba_list)
            
            final_text_outputs_lba, final_confidences_lba = [], []
            indices = []
            
            # TODO: select num_sub_qa_select
            for text_output_lba_list, confidence_lba_list in zip(text_outputs_lba_list, confidences_lba_list):
                # select highest confidence_lba among sub_qa
                index = confidence_lba_list.index(max(confidence_lba_list))
                final_text_outputs_lba.append(text_output_lba_list[index])
                final_confidences_lba.append(round(confidence_lba_list[index], 6))
                indices.append(index)
            
            if args.verbose:
                print(f'indices: {indices[0]}, LBA: {final_text_outputs_lba[0]} | {final_confidences_lba[0]}')
            
            
            """##############################      Save result      ##############################"""
            for i in range(bsz):
                result = OrderedDict({
                    "question_id": batch['question_id'][i],
                    "text_input": text_inputs['qa_input'] if cfg.runner_cfg.recomposer_name == "sevila" else text_inputs[i],
                    "main_question": batch['text_input'][i],
                })
                if cfg.runner_cfg.sub_mode == "subqa":
                    result.update({
                        "sub_question": sub_questions[i],
                        "sub_answer": sub_answers[i],
                    })
                elif cfg.runner_cfg.sub_mode == "description":
                    result.update({
                        "description": descriptions[i],
                    })
                result.update({
                    "gt_ans": gt_answers[i],
                    "confidence_base": confidences_base[i],
                    "confidence_lba": final_confidences_lba[i], #confidences_lba[i],
                    "text_output_base": text_outputs_base[i],
                    "text_output_lba": final_text_outputs_lba[i], #text_outputs_lba[i],
                })
                if args.verbose:
                    w2r, r2w = sample_print(text_outputs_base[i], final_text_outputs_lba[i], gt_answers[i], dataset.get_accuracy, i)
                    wrong2right += w2r
                    right2wrong += r2w
                    
                if 'type' in batch:
                    result['type'] = batch['type'][i]
                    
                results.append(result)
            
            if args.verbose:
                print()
                print(f'accumulated wrong2right: {wrong2right}, right2wrong: {right2wrong}')

        result_path = os.path.join(output_dir, 'results_base.json')
        json.dump(results, open(result_path, 'w'), indent=4)
        print(f'results saved at {result_path}')

        print('inference time : ', datetime.now()-s)
        s = datetime.now()
    else:
        result_path = os.path.join(output_dir, 'results_base.json')
        results = json.load(open(result_path, 'r'))
        print('load results from:', result_path)
        
        total_base_match, total_cnt = 0., 0
        
        for result in results:
            acc_base = dataset.get_accuracy(result['text_output_base'], result['gt_ans'])
            total_base_match += acc_base
            total_cnt += 1
        
        print(f'loaded config path is {args.cfg_path}')#os.path.join(output_dir, "config.yaml")}')
            
        
    """##############################     Report metrics     ##############################"""
    visualize(results, dataset, cfg, output_dir, total_base_match)
    

if __name__ == '__main__':
    main()
