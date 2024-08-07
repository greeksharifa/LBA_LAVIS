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
        model = Recomposer(cfg, device="cuda:0")
        print('model loading time : ', datetime.now()-s)

        s = datetime.now()
        total_base_match, total_cnt = 0., 0

        metric_logger = MetricLogger(delimiter="  ")
        print_freq = max(1, int(len(dataloader) / cfg.runner_cfg.print_freq))
        # print('print_freq:', print_freq)

        results = []
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header='')):
            bsz = len(batch['image'])
            images = batch['image']

            text_inputs = get_sevila_input("default", batch=batch)
            text_outputs_base, confidences_base = model(images, text_inputs)

            gt_answers = batch['gt_ans']  # vqa: list[bsz, 10], videoqa: list[bsz]
            acc_base = dataset.get_accuracy(text_outputs_base, gt_answers)

            total_base_match += sum(acc_base)
            total_cnt += bsz
            metric_logger.update(n=bsz, base_acc=sum(acc_base)/bsz)
            metric_logger.update(n=bsz, total_base_acc=total_base_match/total_cnt)
            
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
        
    metrics = {
        "acc_origin           ": f'{total_base_match / total_cnt * 100:.3f}%',
    }
    print("metrics:", json.dumps(metrics, indent=4))

    

if __name__ == '__main__':
    main()
