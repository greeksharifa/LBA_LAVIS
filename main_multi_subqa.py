import argparse
import os
import json
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime
from pprint import pprint
from collections import OrderedDict
from itertools import combinations
import PIL
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from transformers import InstructBlipVideoProcessor

from configs.config import Config
from dataset.base_dataset import load_dataset, get_text_input, get_train_examplar
from models.model import Decomposer, Recomposer


from utils.misc import SmoothedValue, MetricLogger
from utils.visualize import visualize, sample_print
from utils.llava_answer_eval import map_prediction_to_answer_v2


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
        output_dir = cfg.runner_cfg.output_dir
    
    s = datetime.now()
    
    if cfg.runner_cfg.sub_mode == "frame_sampling":
        n_supple = cfg.runner_cfg.num_frame_sampling
    elif cfg.runner_cfg.vision_supple:
        n_supple = cfg.runner_cfg.num_sub_qa_generate
    else:
        n_supple = 0
    
    if cfg.runner_cfg.recomposer_name == "flipped_vqa":
        import pickle
        from flipped_vqa.get_model import get_flipped_vqa_model
        flipped_vqa_args = pickle.load(open(cfg.runner_cfg.flipped_vqa_args_pkl_path, 'rb'))
        flipped_vqa_model, dataloader = get_flipped_vqa_model(flipped_vqa_args, device="cuda:0")
    
    else:        
        xl_or_xxl = "xxl" if "xxl" in cfg.runner_cfg.recomposer_name else "xl"
        print('xl_or_xxl:', xl_or_xxl)
        dataset = load_dataset(cfg.datasets_cfg, n_supple=n_supple, xl_or_xxl=xl_or_xxl)
        dataloader = DataLoader(dataset, batch_size=cfg.runner_cfg.batch_size,
                                shuffle=False, collate_fn=dataset.collater)
    
    if cfg.runner_cfg.sub_mode == "multi_subqa_highest":
        single_subqa_results = json.load(open(f'{cfg.runner_cfg.single_subqa_output_dir}/results_base.json'))
        single_subqa_results = {r["question_id"]: r for r in single_subqa_results}
        
    if cfg.runner_cfg.sub_mode == "multi_subqa_combo":
        # Get all combinations for sizes 2, 3, 4, and 5
        if cfg.runner_cfg.get("all_combinations", False):
            all_combinations = []
            for size in range(2, 6):
                all_combinations.extend(list(combinations(list(range(5)), size)))
        else:
            all_combinations = list(combinations(list(range(5)), cfg.runner_cfg.num_sub_qa_select))
        print("List of all combinations:")
        for combo in all_combinations:
            print(combo)
    
    print('dataset loading time : ', datetime.now()-s)
    
    if not cfg.runner_cfg.visualize:
        s = datetime.now()
        device_recomposer = "cuda:0"
        device_decomposer = f"cuda:{torch.cuda.device_count() - 1}"
        # recomposer
        recomposer = Recomposer(cfg, device="cuda:0", model_type="recomposer")
        # decomposer
        if cfg.runner_cfg.decomposer_name == "self":
            decomposer = recomposer # Recomposer(cfg, device="cuda:1") # 
        elif "blip" in cfg.runner_cfg.decomposer_name:
            decomposer = Recomposer(cfg, device="cuda:1", model_type="decomposer")
        else:
            decomposer = Decomposer(cfg, device="cuda:1")
        # answerer
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
        if cfg.runner_cfg.train_recomposer_examplar:
            print('examplar:', examplar)
            
        results = []
        wrong2right, right2wrong = 0, 0
        wrong, right = 0, 0
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header='')):
            if args.verbose and data_iter_step == 0:
                print('batch:')
                for k, v in batch.items():
                    if hasattr(v, "shape"):
                        print(f'{k}: {v.shape}')
                    elif isinstance(v, list) and hasattr(v[0], "shape"):
                        print(f'{k}: {len(v)} {v[0].shape}')
                    elif isinstance(v, list) and v[0] == []:
                        print(f'{k}: {len(v)}, v[0]==[]')
                    elif isinstance(v, list) and isinstance(v[0], list) and hasattr(v[0][0], "shape"):
                        print(f'{k}: {len(v)} {len(v[0])} {v[0][0].shape}')
                    elif isinstance(v, list) and isinstance(v[0], list) and isinstance(v[0][0], list) and hasattr(v[0][0][0], "shape"):
                        print(f'{k}: {len(v)} {len(v[0])} {len(v[0][0])} {v[0][0][0].shape}')
                    elif k != "candidate_list":
                        print(f'{k}: {v}')

            bsz = len(batch['vision'])
            vision = batch['vision']
            
            """##############################  Baseline Inference   ##############################"""    
            if cfg.datasets_cfg.data_type == "videos":
                text_inputs = get_text_input("default_video", 
                                                main_questions=batch['text_input'], 
                                                candidate_lists=batch['candidate_list'],
                                                add_examplar="blip2" not in cfg.runner_cfg.recomposer_name,
                                                video_llava="Video-LLaVA" in cfg.runner_cfg.recomposer_name,
                                                )
            else:                          # "images"
                text_inputs = get_text_input("default_image", main_questions=batch['text_input'])
            text_outputs_base, confidences_base = recomposer(vision, text_inputs)
            
            if args.verbose:
                print(f'{data_iter_step:5d}/{len(dataloader)} \t base: ', text_outputs_base[0], ' | ', confidences_base[0])

            gt_answers = batch['gt_ans']  # vqa: list[bsz, 10], videoqa: list[bsz]
            gt_answers = [dataset.answer_mapping(ans) for ans in gt_answers]
            
            acc_base = dataset.get_accuracy(text_outputs_base, gt_answers)

            total_base_match += sum(acc_base)
            total_cnt += bsz
            metric_logger.update(n=bsz, base_acc=sum(acc_base)/bsz)
            metric_logger.update(n=bsz, total_base_acc=total_base_match/total_cnt)
            
            if args.verbose:
                print("batch['text_input'][0]:", batch['text_input'][0])
            
            """############################## Decompose & Recompose ##############################"""
            sub_questions_list, sub_answers_list, text_outputs_lba_list, confidences_lba_list = [], [], [], []
            
            if cfg.runner_cfg.sub_mode == "subqa" or cfg.runner_cfg.sub_mode == "Ktype":
                for i in range(cfg.runner_cfg.num_sub_qa_generate):
                    # generating sub_questions
                    if cfg.runner_cfg.use_pre_generated_sub_q:
                        # using pre-generated sub_questions
                        sub_questions = [[] for _ in range(bsz)]
                        # sub_questions = []
                        for b in range(bsz):
                            try:
                                for j in range(cfg.runner_cfg.num_sub_qa_select):
                                    sub_questions[b].append(batch['sub_question_list'][b][(i+j) % cfg.runner_cfg.num_sub_qa_generate])
                                # sub_questions.append(batch['sub_question_list'][b][i])
                            except:
                                import pdb; pdb.set_trace()
                    else:
                        if cfg.runner_cfg.vision_supple:# and i >= 1:
                            vision = []
                            for b in range(bsz):
                                vision.append(batch['vision_supple'][b][i])
                                # vision.append(batch['vision_supple'][b][i-1])
                        text_inputs = get_text_input("decomposer", main_questions=batch['text_input'])
                        
                        if type(decomposer) == Decomposer:
                            sub_questions = decomposer(text_inputs)
                        else:
                            beam_search = i==0
                            sub_questions, _ = decomposer(vision, text_inputs, generate_sub_q=True, beam_search=beam_search)
                    sub_questions_list.append(sub_questions)
                    
                    # generating sub_answers
                    if cfg.runner_cfg.use_pre_generated_sub_a:
                        sub_answers = [[] for _ in range(bsz)]
                        # sub_answers = []
                        for b in range(bsz):
                            for j in range(cfg.runner_cfg.num_sub_qa_select):
                                sub_answers[b].append(batch['sub_answer_list'][b][(i+j) % cfg.runner_cfg.num_sub_qa_generate])
                            # sub_answers.append(batch['sub_answer_list'][b][i])
                    else:
                        text_inputs = get_text_input("sub_answer", sub_questions=sub_questions)
                        sub_answers, _ = answerer(vision, text_inputs)
                    sub_answers_list.append(sub_answers)
                    
                    # generating recomposed_answers
                    if cfg.datasets_cfg.data_type == "videos":
                        text_inputs = get_text_input("recomposer_video", 
                                                    main_questions=batch['text_input'], 
                                                    sub_questions=sub_questions, 
                                                    sub_answers=sub_answers,
                                                    candidate_lists=batch['candidate_list'],
                                                    examplar=examplar,
                                                    train_recomposer_examplar=cfg.runner_cfg.train_recomposer_examplar,
                                                    video_llava="Video-LLaVA" in cfg.runner_cfg.recomposer_name,
                                                    )
                    else:                          # "images"
                        text_inputs = get_text_input("recomposer_image", 
                                                    main_questions=batch['text_input'], 
                                                    sub_questions=sub_questions, 
                                                    sub_answers=sub_answers)
                    text_outputs_lba, confidences_lba = recomposer(vision, text_inputs)
                    
                    if cfg.runner_cfg.debug:
                        t_inputs = text_inputs[0]
                        print('sub_questions text_inputs:', t_inputs)
                        print('sub_answers text_inputs:', t_inputs)
                        print('recomposer_video text_inputs:', t_inputs)
                        
                    text_outputs_lba_list.append(text_outputs_lba)
                    confidences_lba_list.append(confidences_lba)
                    
            elif cfg.runner_cfg.sub_mode == "multi_subqa_highest": # always use pre-generated sub_q
                batch_rank_list = []
                for b in range(bsz):
                    question_id = batch['question_id'][b]
                    ss_result = single_subqa_results[question_id]
                    # ss_text_outputs_lba_list = ss_result['text_outputs_lba_list']
                    ss_confidences_lba_list = ss_result['confidences_lba_list']

                    indexed_confidences = list(enumerate(ss_confidences_lba_list))
                    sorted_confidences = sorted(indexed_confidences, key=lambda x: x[1], reverse=True)
                    rank_list = [original_index for original_index, _ in sorted_confidences]
                    batch_rank_list.append(rank_list)
                
                # get num_sub_qa_select sub_qas with higher confidences
                sub_questions = [[] for _ in range(bsz)]
                sub_answers = [[] for _ in range(bsz)]
                for b in range(bsz):
                    for j in range(cfg.runner_cfg.num_sub_qa_select):
                        idx = batch_rank_list[b][j]
                        sub_questions[b].append(batch['sub_question_list'][b][idx])
                        sub_answers[b].append(batch['sub_answer_list'][b][idx])
                
                # generating recomposed_answers
                if cfg.datasets_cfg.data_type == "videos":
                    text_inputs = get_text_input("recomposer_video", 
                                                main_questions=batch['text_input'], 
                                                sub_questions=sub_questions, 
                                                sub_answers=sub_answers,
                                                candidate_lists=batch['candidate_list'],
                                                examplar=examplar,
                                                train_recomposer_examplar=cfg.runner_cfg.train_recomposer_examplar,
                                                video_llava="Video-LLaVA" in cfg.runner_cfg.recomposer_name,
                                                )
                else:                          # "images"
                    text_inputs = get_text_input("recomposer_image", 
                                                main_questions=batch['text_input'], 
                                                sub_questions=sub_questions, 
                                                sub_answers=sub_answers)
                text_outputs_lba, confidences_lba = recomposer(vision, text_inputs)
                
                if cfg.runner_cfg.debug:
                    t_inputs = text_inputs[0]
                    print('sub_questions text_inputs:', t_inputs)
                    print('sub_answers text_inputs:', t_inputs)
                    print('recomposer_video text_inputs:', t_inputs)
                    
                text_outputs_lba_list.append(text_outputs_lba)
                confidences_lba_list.append(confidences_lba)
                
            elif cfg.runner_cfg.sub_mode == "multi_subqa_combo": # always use pre-generated sub_q
                for _combinations in all_combinations:
                    sub_questions = [[] for _ in range(bsz)]
                    sub_answers = [[] for _ in range(bsz)]
                    for b in range(bsz):
                        for idx in _combinations:
                            sub_questions[b].append(batch['sub_question_list'][b][idx])
                            sub_answers[b].append(batch['sub_answer_list'][b][idx])
                    sub_questions_list.append(sub_questions)
                    sub_answers_list.append(sub_answers)
                    
                    # generating recomposed_answers
                    if cfg.datasets_cfg.data_type == "videos":
                        text_inputs = get_text_input("recomposer_video", 
                                                    main_questions=batch['text_input'], 
                                                    sub_questions=sub_questions, 
                                                    sub_answers=sub_answers,
                                                    candidate_lists=batch['candidate_list'],
                                                    examplar=examplar,
                                                    train_recomposer_examplar=cfg.runner_cfg.train_recomposer_examplar,
                                                    video_llava="Video-LLaVA" in cfg.runner_cfg.recomposer_name,
                                                    )
                    else:                          # "images"
                        text_inputs = get_text_input("recomposer_image", 
                                                    main_questions=batch['text_input'], 
                                                    sub_questions=sub_questions, 
                                                    sub_answers=sub_answers)
                    text_outputs_lba, confidences_lba = recomposer(vision, text_inputs)
                    
                    if cfg.runner_cfg.debug:
                        t_inputs = text_inputs[0]
                        print('sub_questions text_inputs:', t_inputs)
                        print('sub_answers text_inputs:', t_inputs)
                        print('recomposer_video text_inputs:', t_inputs)
                        
                    text_outputs_lba_list.append(text_outputs_lba)
                    confidences_lba_list.append(confidences_lba)
                   
            elif cfg.runner_cfg.sub_mode == "description":
                descriptions_list = []
                text_inputs = ["Describe the video in detail. What is happening in the video?" for _ in range(bsz)]
                descriptions, _ = recomposer(vision, text_inputs, generate_sub_q=True)
                descriptions_list.append(descriptions)
                
                # generating recomposed_answers
                if cfg.datasets_cfg.data_type == "videos":
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
                    
            else:
                raise NotImplementedError(f"{cfg.runner_cfg.sub_mode} is not implemented.")
            
            if args.verbose:
                # print(f'sub_QA: {sub_questions} -> {sub_answers}. LBA: {text_outputs_lba} | {[f"{float(x):.6f}" for x in confidences_lba]}')
                print(f'text_inputs:\n{text_inputs[0]}')
                print(f'sub_QA: {sub_questions[0]} -> {sub_answers[0]}. LBA: {text_outputs_lba[0]} | {confidences_lba[0]:.6f}')
            
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
                final_confidences_lba.append(round(confidence_lba_list[index], 6)) # 10?
                indices.append(index)
            
            if args.verbose:
                print(f'indices: {indices[0]}, LBA: {final_text_outputs_lba[0]} | {final_confidences_lba[0]}')
            
            """##############################      Save result      ##############################"""
            for i in range(bsz):
                result = OrderedDict({
                    "question_id": batch['question_id'][i],
                    "text_input": text_inputs[i],
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
                    "text_outputs_lba_list": text_outputs_lba_list[i],
                    "confidences_lba_list": confidences_lba_list[i],
                })
                if args.verbose:
                    w2r, r2w, w, r = sample_print(text_outputs_base[i], final_text_outputs_lba[i], gt_answers[i], dataset.get_accuracy, i)
                    wrong2right += w2r
                    right2wrong += r2w
                    wrong += w
                    right += r
                    
                if 'type' in batch:
                    result['type'] = batch['type'][i]
                    
                results.append(result)
            
            if args.verbose:
                print(f'\nwrong: {wrong} wrong2right: {wrong2right}, right2wrong: {right2wrong} right: {right} total_cnt: {total_cnt}')

        result_path = os.path.join(output_dir, 'results_base.json')
        json.dump(results, open(result_path, 'w'), indent=4)
        print(f'results saved at {result_path}')

        print('inference time : ', datetime.now()-s)
        s = datetime.now()
    else:
        if cfg.runner_cfg.get("sevila_visualize", False):
            # python main_multi_subqa.py --options runner.visualize=True runner.sevila_visualize=True runner.output_dir="output/visualize_sevila/" runner.sub_mode="subqa" datasets.root_dir="/data1/" runner.select_high_confidence=True runner.max_conf_gap=None datasets.dataset_name="VLEP" runner.num_sub_qa_generate=1 runner.visualize_xl=True
            total_base_match, total_cnt = 0., 0
            _results = {}
            dataset_name = cfg.datasets_cfg.dataset_name.lower()
            if cfg.runner_cfg.get("sub_mode", "subqa") == "Ktype":
                subqa_type = "sub_qas_val_xl_Ktype"
            elif cfg.runner_cfg.get("visualize_xl", False):
                subqa_type = "sub_qas_val_xl"
            elif cfg.runner_cfg.get("visualize_fvu", False):
                subqa_type = "sub_qas_val_fvu"
            else:
                subqa_type = "sub_qas_val"
            results_base = json.load(open(f'SeViLA/lavis/result_{dataset_name}_{subqa_type}/base/result/val_epochbest.json'))
            for r in results_base:
                _results[r['qid']] = {
                    "question_id": r["qid"],
                    "gt_ans": r["target"],
                    "text_output_base": r["prediction"],
                    "confidence_base": r["confidence"],
                    "text_output_lba_list": [],
                    "confidence_lba_list": [],
                }
                total_base_match += dataset.get_accuracy(r['prediction'], r['target'])
                total_cnt += 1
                
            # for i in [0, 3, 4]:
            for i in range(0, cfg.runner_cfg.num_sub_qa_generate):
                results_subqa = json.load(open(f'SeViLA/lavis/result_{dataset_name}_{subqa_type}/{i}/result/val_epochbest.json'))
                for r in results_subqa:
                    _results[r['qid']][f'text_output_lba_list'].append(r["prediction"])
                    _results[r['qid']][f'confidence_lba_list'].append(r["confidence"])
            
            results = []
            for k, v in _results.items():
                max_confidence_lba = max(v['confidence_lba_list'])
                idx_max_confidence_lba = v['confidence_lba_list'].index(max_confidence_lba)
                text_output_lba = v['text_output_lba_list'][idx_max_confidence_lba]
                r = v
                r['text_output_lba'] = text_output_lba
                r['confidence_lba'] = max_confidence_lba
                r['type'] = v["question_id"].split("_")[0]
                results.append(r) 
            
        elif cfg.runner_cfg.get("IGVLM_visualize", False):
            # python main_multi_subqa.py --options runner.visualize=True runner.IGVLM_visualize=True runner.output_dir="output/visualize_IGVLM/" runner.sub_mode="subqa" datasets.root_dir="/data1/" runner.baseline=False runner.select_high_confidence=True runner.max_conf_gap=None datasets.dataset_name="TVQA" runner.num_sub_qa_generate=4 runner.visualize_xl=True
            total_base_match, total_cnt = 0., 0
            _results = {}
            dataset_name = cfg.datasets_cfg.dataset_name
            
            if cfg.runner_cfg.get("sub_mode", "subqa") == "Ktype":
                subqa_type = "sub_qas_val_xl_Ktype"
            elif cfg.runner_cfg.get("visualize_xl_fvu", False):
                subqa_type = "sub_qas_val_xl_fvu"#"sub_qas_val_fewshot_vqaintrospect_unique"
            else:
                subqa_type = "sub_qas_val_xl"
              
            if cfg.runner_cfg.num_sub_qa_select > 1:  
                subqa_type = cfg.runner_cfg.llm_size + f"_select{cfg.runner_cfg.num_sub_qa_select}_" + subqa_type.replace('xl', 'xxl')
            elif cfg.runner_cfg.get("llm_size", ""):
                subqa_type = cfg.runner_cfg.llm_size + "_" + subqa_type.replace('xl', 'xxl')
                
            results_base = pd.read_csv(f'output/IGVLM/result_{dataset_name}_{subqa_type}/base/ffn=6/result.csv', index_col=0)
            
            results_base["predicted_answer"] = results_base.apply(map_prediction_to_answer_v2, axis=1)
            results_base["is_correct"] = results_base["predicted_answer"] == results_base["answer"]
            
            for idx, row in results_base.iterrows():
                pred_base = map_prediction_to_answer_v2(row)
                _results[row['question_id']] = {
                    "question_id": str(row["question_id"]),
                    # "type": row["question_type"],
                    "gt_ans": row["answer"],
                    "text_output_base": pred_base,
                    "confidence_base": row["confidence_score"],
                    "text_output_lba_list": [],
                    "confidence_lba_list": [],
                }
                if "question_type" in row:
                    _results[row['question_id']]["type"] = row["question_type"].split("_")[0]
                total_base_match += pred_base == row["answer"]
                total_cnt += 1
                
                ours_match = pred_base == row["answer"]
                IGVLM_match = row["predicted_answer"] == row["answer"]
                if ours_match != IGVLM_match:
                    import pdb; pdb.set_trace()
                pass
            
            print(f'total_base_match: {total_base_match}, total_cnt: {total_cnt}, accuracy: {total_base_match/total_cnt * 100:.2f}%')
            
            total_accuracy = results_base["is_correct"].mean()
            print(f'IGVLM total_accuracy: {total_accuracy * 100:.2f}%')
                
            # for i in [0, 3, 4]:
            for i in range(0, cfg.runner_cfg.num_sub_qa_generate):
                results_subqa = pd.read_csv(f'output/IGVLM/result_{dataset_name}_{subqa_type}/{i}/ffn=6/result.csv', index_col=0)
                for idx, row in results_subqa.iterrows():
                    _results[row['question_id']][f'text_output_lba_list'].append(map_prediction_to_answer_v2(row))
                    _results[row['question_id']][f'confidence_lba_list'].append(row["confidence_score"])
            
            results = []
            for k, v in _results.items():
                max_confidence_lba = max(v['confidence_lba_list'])
                idx_max_confidence_lba = v['confidence_lba_list'].index(max_confidence_lba)
                text_output_lba = v['text_output_lba_list'][idx_max_confidence_lba]
                r = v
                r['text_output_lba'] = text_output_lba
                r['confidence_lba'] = max_confidence_lba
                # r['type'] = r["type"].split("_")[0] #v["question_id"].split("_")[0]
                results.append(r) 
        else:
            result_path = os.path.join(output_dir, 'results_base.json')
            results = json.load(open(result_path, 'r'))
            print('load results from:', result_path)
            
            total_base_match, total_cnt = 0, 0
            
            for result in results:
                acc_base = dataset.get_accuracy(result['text_output_base'], result['gt_ans'])
                total_base_match += acc_base
                total_cnt += 1
                
                # select highest confidence_lba among sub_qa
                if "confidences_lba_list" in result and "text_outputs_lba_list" in result:
                    # result['confidences_lba_list'] = [result['confidences_lba_list'][i] for i in idxs]
                    # result['text_outputs_lba_list'] = [result['text_outputs_lba_list'][i] for i in idxs]
                    # max_confidence_lba = max(result['confidences_lba_list'])
                    # idx_max_confidence_lba = result['confidences_lba_list'].index(max_confidence_lba)
                    # text_output_lba = result['text_outputs_lba_list'][idx_max_confidence_lba]
                    max_confidence_lba = max(result['confidences_lba_list'][:cfg.runner_cfg.num_sub_qa_generate])
                    idx_max_confidence_lba = result['confidences_lba_list'][:cfg.runner_cfg.num_sub_qa_generate].index(max_confidence_lba)
                    text_output_lba = result['text_outputs_lba_list'][:cfg.runner_cfg.num_sub_qa_generate][idx_max_confidence_lba]
                    
                    result['text_output_lba'] = text_output_lba
                    result['confidence_lba'] = max_confidence_lba
                
            
            print(f'loaded config path is {args.cfg_path}')#os.path.join(output_dir, "config.yaml")}')
            
    try:
        print('recomposer.model.__class__.__name__:', recomposer.model.__class__.__name__)
    except:
        pass
    # import pdb; pdb.set_trace()
    """##############################     Report metrics     ##############################"""
    visualize(results, dataset, cfg, output_dir, total_base_match)
    

if __name__ == '__main__':
    main()
