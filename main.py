import argparse
import os
import json
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime
from pprint import pprint
from collections import OrderedDict

from torch.utils.data import DataLoader

from configs.config import Config
# from dataset.VQA_Introspect import VQAIntrospectDataset
from dataset.base_dataset import get_text_input, load_dataset
from models.model import Decomposer, Recomposer

from utils.misc import SmoothedValue, MetricLogger
from utils.visualize import visualize
from utils.colors import Colors


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
        print('model loading time : ', datetime.now()-s)

        s = datetime.now()
        total_base_match, total_cnt = 0., 0

        metric_logger = MetricLogger(delimiter="  ")
        print_freq = max(1, int(len(dataloader) / cfg.runner_cfg.print_freq))
        # print('print_freq:', print_freq)

        
        results = []
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header='')):
            # if data_iter_step == 0:
                # pprint(batch, width=300)

            bsz = len(batch['image'])
            images = batch['image']

            """##############################  Baseline Inference   ##############################"""    
            
            if cfg.datasets_cfg.data_type == "videos":
                text_inputs = get_text_input("default_video", main_questions=batch['text_input'], candidate_lists=batch['candidate_list'])
            else:                          # "images"
                text_inputs = get_text_input("default_image", main_questions=batch['text_input'])
            text_outputs_base, confidences_base = recomposer(images, text_inputs)
            # print(f'{data_iter_step:5d}/{len(dataloader)} : ', outputs[0])

            gt_answers = batch['gt_ans']  # list[bsz, 10]
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
            
            # generating sub_answers
            text_inputs = get_text_input("sub_answer", sub_questions=sub_questions)
            sub_answers, _ = recomposer(images, text_inputs)
            
            # generating recomposed_answers
            if cfg.datasets_cfg.data_type == "videos":
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
            
            """##############################      Save result      ##############################"""
            for i in range(bsz):
                result = OrderedDict({
                            "question_id": batch['question_id'][i],
                            "main_question": batch['text_input'][i],
                            "text_input": text_inputs[i],
                            "gt_ans": gt_answers[i],
                            "confidence_base": confidences_base[i],
                            "confidence_lba": confidences_lba[i],
                            "text_output_base": text_outputs_base[i],
                            "sub_question": sub_questions[i],
                            "sub_answer": sub_answers[i],
                            "text_output_lba": text_outputs_lba[i],
                        })
                if args.verbose and i == 0:
                    pprint(result, width=300)
                    base = text_outputs_base[i]
                    lba = text_outputs_lba[i]
                    gt_ans = gt_answers[i]
                    if base != gt_ans and lba == gt_ans:    # wrong -> right
                        color = Colors.BRIGHT_GREEN
                    elif base == gt_ans and lba != gt_ans:  # right -> wrong
                        color = Colors.BRIGHT_RED
                    else:
                        color = Colors.BRIGHT_YELLOW
                    print(Colors.BRIGHT_YELLOW, f'{base} -> {lba}, gt: {gt_ans}', Colors.RESET)
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
