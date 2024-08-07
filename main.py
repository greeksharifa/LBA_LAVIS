import argparse
import os
import json
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime
from pprint import pprint
from collections import OrderedDict
import PIL

import torch
from torch.utils.data import DataLoader

from configs.config import Config
# from dataset.VQA_Introspect import VQAIntrospectDataset
from dataset.base_dataset import load_dataset, get_text_input, get_sevila_input
from models.model import Decomposer, Recomposer

from utils.misc import SmoothedValue, MetricLogger
from utils.visualize import visualize, sample_print


def parse_args():
    parser = argparse.ArgumentParser(description='LBA method')
    parser.add_argument("--cfg-path", default='configs/runner.yaml', help="path to configuration file.")
    # verbose
    parser.add_argument('--verbose', action='store_true', help='verbose')
    
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
    os.environ['HF_HOME'] = cfg.runner_cfg.HF_HOME
    # args = parse_args()
    output_dir = os.path.join(cfg.runner_cfg.output_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir)
    print('cfg:\n', cfg._convert_node_to_json(cfg.config), sep='')
    OmegaConf.save(config=cfg.config, f=os.path.join(output_dir, "config.yaml"))
    
    s = datetime.now()
    dataset = load_dataset(cfg.datasets_cfg)
    dataloader = DataLoader(dataset, batch_size=cfg.runner_cfg.batch_size,
                            shuffle=False, collate_fn=dataset.collater)
    print('dataset loading time : ', datetime.now()-s)
    
    if not cfg.runner_cfg.visualize:
        s = datetime.now()
        recomposer = Recomposer(cfg, device="cuda:0")
        
        if cfg.runner_cfg.decomposer_name == "self":
            decomposer = recomposer
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

        
        results = []
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header='')):
            # if data_iter_step == 0:
            print('batch:')
            for k, v in batch.items():
                if hasattr(v, "shape"):
                    print(f'{k}: {v.shape}')
                elif isinstance(v, list) and hasattr(v[0], "shape"):
                    print(f'{k}: {len(v)} {v[0].shape}')
                elif isinstance(v, list) and isinstance(v[0], list) and isinstance(v[0][0], PIL.Image.Image):
                    print(f'{k}: {len(v)} {len(v[0])} {v[0][0].size}')
                else:
                    print(f'{k}: {v}')
            # pprint(batch, width=300)

            bsz = len(batch['image'])
            images = batch['image']

            """##############################  Baseline Inference   ##############################"""    
            if cfg.runner_cfg.recomposer_name == "sevila":
                # return list of dict, not list of str
                text_inputs = get_sevila_input("default", batch=batch)
            elif cfg.datasets_cfg.data_type == "videos":
                text_inputs = get_text_input("default_video", main_questions=batch['text_input'], candidate_lists=batch['candidate_list'])
            else:                          # "images"
                text_inputs = get_text_input("default_image", main_questions=batch['text_input'])
            text_outputs_base, confidences_base = recomposer(images, text_inputs)
            print(f'{data_iter_step:5d}/{len(dataloader)} : ', text_outputs_base, confidences_base)

            gt_answers = batch['gt_ans']  # vqa: list[bsz, 10], videoqa: list[bsz]
            if cfg.runner_cfg.recomposer_name != "sevila":
                gt_answers = [dataset.__class__.ANSWER_MAPPING[ans] for ans in gt_answers]
            acc_base = dataset.get_accuracy(text_outputs_base, gt_answers)

            total_base_match += sum(acc_base)
            total_cnt += bsz
            metric_logger.update(n=bsz, base_acc=sum(acc_base)/bsz)
            metric_logger.update(n=bsz, total_base_acc=total_base_match/total_cnt)
            
            """############################## Decompose & Recompose ##############################"""
            
            # generating sub_questions
            text_inputs = get_text_input("decomposer", main_questions=batch['text_input'])
            if cfg.runner_cfg.decomposer_name == "self":  # Image+Text, BLIP-2
                sub_questions, _ = decomposer(images, text_inputs)
            else:                               # Only Text, flan-t5
                sub_questions = decomposer(text_inputs)
            print('sub_questions:', sub_questions)
            
            
            # generating sub_answers
            text_inputs = get_text_input("sub_answer", sub_questions=sub_questions)
            sub_answers, _ = answerer(images, text_inputs)
            '''
            if cfg.runner_cfg.recomposer_name == "sevila":
                # sub_a_batch = batch.copy()
                # sub_a_batch['text_input'] = sub_questions
                # text_inputs = get_sevila_input("default", batch=batch)
                text_inputs = get_text_input("sub_answer", sub_questions=sub_questions)
                sub_answers, _ = answerer(images, text_inputs)
            else:
                text_inputs = get_text_input("sub_answer", sub_questions=sub_questions)
                sub_answers, _ = recomposer(images, text_inputs)
            '''
            print('sub_answers:', sub_answers)
            
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
                                             candidate_lists=batch['candidate_list'])
            else:                          # "images"
                text_inputs = get_text_input("recomposer_image", 
                                             main_questions=batch['text_input'], 
                                             sub_questions=sub_questions, 
                                             sub_answers=sub_answers)
            text_outputs_lba, confidences_lba = recomposer(images, text_inputs)
            
            print('text_outputs_lba:', text_outputs_lba)
            print('confidences_lba:', confidences_lba)
            
            
            """##############################      Save result      ##############################"""
            for i in range(bsz):
                result = OrderedDict({
                            "question_id": batch['question_id'][i],
                            "main_question": batch['text_input'][i],
                            "text_input": text_inputs['qa_input'] if cfg.runner_cfg.recomposer_name == "sevila" else text_inputs[i],
                            "gt_ans": gt_answers[i],
                            "confidence_base": confidences_base[i],
                            "confidence_lba": confidences_lba[i],
                            "text_output_base": text_outputs_base[i],
                            "sub_question": sub_questions[i],
                            "sub_answer": sub_answers[i],
                            "text_output_lba": text_outputs_lba[i],
                        })
                if args.verbose and i == 0:
                    # pprint(result, width=300)
                    sample_print(text_outputs_base[i], text_outputs_lba[i], gt_answers[i], dataset.get_accuracy)
                    
                results.append(result)

        result_path = os.path.join(output_dir, 'results_base.json')
        json.dump(results, open(result_path, 'w'), indent=4)
        print(f'results saved at {result_path}')

        print('inference time : ', datetime.now()-s)
        s = datetime.now()
    else:
        results = json.load(open(os.path.join(output_dir, 'results_base.json'), 'r'))
        
        total_base_match, total_cnt = 0., 0
        
        for result in results:
            acc_base = dataset.get_accuracy(result['text_output_base'], result['gt_ans'])
            total_base_match += acc_base
            total_cnt += 1
            
        
    """##############################     Report metrics     ##############################"""
    visualize(results, dataset, cfg, output_dir, total_base_match)
    

if __name__ == '__main__':
    main()
