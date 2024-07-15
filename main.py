import argparse
import os
import json
from omegaconf import OmegaConf
from tqdm import tqdm
from datetime import datetime
from pprint import pprint
from collections import OrderedDict
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from configs.config import Config
# from dataset.VQA_Introspect import VQAIntrospectDataset
from dataset.base_dataset import get_text_input, load_dataset
from model import Decomposer, Recomposer

from utils.misc import SmoothedValue, MetricLogger


def parse_args():
    parser = argparse.ArgumentParser(description='LBA method')
    parser.add_argument("--cfg-path", default='configs/runner.yaml', help="path to configuration file.")
    
    '''
    # mode
    parser.add_argument('--visualize', action='store_true', help='visualize the results')
    # match1ok
    parser.add_argument('--match1ok', action='store_true', help='match1ok')

    # model
    parser.add_argument('--recomposer_name', type=str, default='Salesforce/blip2-flan-t5-xl', help='recomposer_name, ex) "Salesforce/blip2-flan-t5-xl"')
    parser.add_argument('--decomposer_name', type=str, default='self', choices=["self", "small", "base", "large", "xl", "xxl"], help='decomposer_name in flan-t5-family, choices=["self", "small", "base", "large", "xl", "xxl"]')

    # dataset
    parser.add_argument('--dataset_path', type=str, default='/data1/VQA-Introspect/VQAIntrospect_valv1.0.json', help='path to the dataset')
    parser.add_argument('--vis_root', type=str, default='/data1/coco/images/', help='Root directory of images (e.g. coco/images/)')
    parser.add_argument('--vqa_acc', action='store_true', help='use vqa acc or exact match')
    # num of data
    parser.add_argument('--num_data', type=int, default=-1, help='number of data to use')

    # output & log
    parser.add_argument('--print_freq', type=int, default=50, help='# of logs per epoch')
    parser.add_argument('--output_dir', type=str, default='output/', help='output directory')
    parser.add_argument('--use_vqa_tool', action='store_true', help='use vqa tool or not')
    parser.add_argument('--num_bin', type=int, default=50, help='number of bins for acc')
    '''

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
    cfg = Config(parse_args())
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
        recomposer = Recomposer(cfg.runner_cfg.recomposer_name, device="cuda:0")
        if cfg.runner_cfg.decomposer_name == "self":
            decomposer = recomposer
        else:
            decomposer = Decomposer(cfg.runner_cfg.decomposer_name, device="cuda:1")
        print('model loading time : ', datetime.now()-s)

        s = datetime.now()
        total_base_match, total_cnt = 0., 0

        metric_logger = MetricLogger(delimiter="  ")
        print_freq = max(1, int(len(dataloader) / cfg.runner_cfg.print_freq))
        print('print_freq:', print_freq)

        
        results = []
        for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header='')):
            # if data_iter_step == 0:
                # pprint(batch, width=300)

            bsz = len(batch['image'])
            images = batch['image']

            """##############################  Baseline Inference   ##############################"""    
            
            text_inputs = get_text_input("default", main_questions=batch['text_input'])
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
            text_inputs = get_text_input("recomposer", 
                                        main_questions=batch['text_input'], 
                                        sub_questions=sub_questions, 
                                        sub_answers=sub_answers)
            text_outputs_lba, confidences_lba = recomposer(images, text_inputs)
            
            """##############################      Save result      ##############################"""
            for i in range(bsz):
                results.append(
                    OrderedDict({
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
                )

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
    results.sort(key=lambda x: x['confidence_lba'])
    
    max_match, cur_match = total_base_match, total_base_match
    match_list = [cur_match]
    max_arg_confidence = -1e10
    confidence_percentile = 0.
    acc_base_list, acc_lba_list = [], []
    N = len(results)
    bins = [[] for _ in range(N // (N // cfg.runner_cfg.num_bin) + 1)]
    
    for i, result in enumerate(results):
        acc_base = dataset.get_accuracy(result['text_output_base'], result['gt_ans'])
        acc_lba = dataset.get_accuracy(result['text_output_lba'], result['gt_ans'], match1ok=cfg.runner_cfg.match1ok)
        acc_base_list.append(acc_base)
        acc_lba_list.append(acc_lba)
        
        bin_key = i // (N // cfg.runner_cfg.num_bin)
        bins[bin_key].append(acc_base)
        
        cur_match += acc_lba - acc_base
        match_list.append(cur_match)
        
        if cur_match > max_match:
            max_match = cur_match
            max_arg_confidence = result['confidence_base']
            confidence_percentile = (i+1) / N * 100
            
    final_acc_list = [match / N for match in match_list]
    
    
    # E_CR, E_IC: Error Correction raio / Error Induction ratio
    e_cr, e_ic = dataset.get_e_cr_e_ic(acc_base_list, acc_lba_list)
    
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
    # plt.subplots(constrained_layout=True)
    plt.figure(figsize=(6,8))
    plt.subplot(2, 1, 1)
    plt.plot([i / N * 100 for i, _ in enumerate(final_acc_list)], final_acc_list, color='b')
    plt.title(f'{dataset.__class__.__name__} | E_CR: {e_cr:.2f}%, E_IC: {e_ic:.2f}%')
    plt.xlabel('Confidence Percentile')
    plt.ylabel('Accuracy')
    plt.xticks([0, 25, 50, 75, 100])
    
    plt.subplot(2, 1, 2)
    acc_bin = [sum(bin) / len(bin) for bin in bins if len(bin) > 0]
    plt.plot([i for i in range(len(acc_bin))], acc_bin, color='r')
    plt.title(f'{dataset.__class__.__name__} | acc for {len(acc_bin)} bins')
    plt.xlabel('bins')
    plt.ylabel('Accuracy')
    plt.xticks([(cfg.runner_cfg.num_bin // 5) * i for i in range(6)])
    fig_path = os.path.join(output_dir, "acc_bin.png")
    plt.savefig(fig_path, dpi=300)
    print(f'saved fig at {fig_path}')
    
    metrics = {
        "acc_origin           ": f'{total_base_match / N * 100:.3f}%',
        "max_acc_by_tau       ": f'{max(final_acc_list) * 100:.3f}%', 
        "max_arg_confidence   ": f'{max_arg_confidence:.6f}',
        "confidence_percentile": f'{confidence_percentile:.3f}%',
        "E_CR                 ": f'{e_cr:.2f}%',
        "E_IC                 ": f'{e_ic:.2f}%',
    }
    print("metrics:", json.dumps(metrics, indent=4))

    with open(os.path.join(output_dir, "evaluate.txt"), "w") as f:
        f.write(json.dumps(metrics, indent=4) + "\n")
        
    
    
    

if __name__ == '__main__':
    main()
