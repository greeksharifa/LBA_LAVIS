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

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import InstructBlipVideoImageProcessor, InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration

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
    
    device = "cuda:0"
    os.environ['HF_HOME'] = cfg.runner_cfg.HF_HOME
    # print('cfg:\n', cfg._convert_node_to_json(cfg.config), sep='')
    
    s = datetime.now()
          
    dataset = load_dataset(cfg.datasets_cfg, n_supple=1)
    dataloader = DataLoader(dataset, batch_size=cfg.runner_cfg.batch_size,
                            shuffle=False, collate_fn=dataset.collater)
    print('dataset loading time : ', datetime.now()-s)
    
    s = datetime.now()
    
    # recomposer
    recomposer = Recomposer(cfg, device="cuda:0", model_type="recomposer")
    print('model loading time : ', datetime.now()-s)

    s = datetime.now()
    
    metric_logger = MetricLogger(delimiter="  ")
    print_freq = max(1, int(len(dataloader) / cfg.runner_cfg.print_freq))

    results = {}
    for data_iter_step, batch in tqdm(enumerate(metric_logger.log_every(dataloader, print_freq, header=''))):
        # print(f'\r{data_iter_step:5d}/{len(dataloader)}', end='')
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
        for i in range(5):
            sub_questions = [sub_qs[i] for sub_qs in batch['sub_question_list']]
            text_inputs = get_text_input("sub_answer", sub_questions=sub_questions)
            sub_answers, _ = recomposer(vision, text_inputs)
            
            for b, qid in enumerate(batch['question_id']):
                if qid in results:
                    results[qid].append((sub_questions[b], sub_answers[b]))
                else:
                    results[qid] = [(sub_questions[b], sub_answers[b])]
          
        # if data_iter_step == 0:
        #     import pdb; pdb.set_trace()
            
    """##############################      Save result      ##############################"""
    print('\n')
    output_dir = f'/{cfg.datasets_cfg.root_dir}/{cfg.datasets_cfg.dataset_name}'

    result_path = os.path.join(output_dir, 'sub_qas_val.json')
    json.dump(results, open(result_path, 'w'), indent=4)
    print(f'results saved at {result_path}')

    print('inference time : ', datetime.now()-s)
    s = datetime.now()
    

if __name__ == '__main__':
    main()
