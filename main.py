import argparse
import os
import json
from tqdm import tqdm
from datetime import datetime
from pprint import pprint
from collections import OrderedDict

from torch.utils.data import DataLoader

from dataset.VQA_Introspect import VQAIntrospectDataset
from dataset.base_dataset import get_text_input
from model import Decomposer, Recomposer

from utils.misc import SmoothedValue, MetricLogger


def get_args():
    parser = argparse.ArgumentParser(description='LBA method')

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
    parser.add_argument('--print_freq', type=int, default=4, help='# of logs per epoch')
    parser.add_argument('--output_dir', type=str, default='output/', help='output directory')
    parser.add_argument('--use_vqa_tool', action='store_true', help='use vqa tool or not')

    '''
    parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers (default: find the max)')
    '''

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    s = datetime.now()
    dataset = VQAIntrospectDataset(None, None, args.vis_root, [
                                   args.dataset_path, '/data1/VQA/v2/v2_mscoco_val2014_annotations.json'])
    dataloader = DataLoader(dataset, batch_size=64,
                            shuffle=False, collate_fn=dataset.collater)
    print('dataset loading time : ', datetime.now()-s)
    s = datetime.now()
    recomposer = Recomposer(args.recomposer_name, device="cuda:0")
    if args.decomposer_name == "self":
        decomposer = recomposer
    else:
        decomposer = Decomposer(args.decomposer_name, device="cuda:1")
    print('model loading time : ', datetime.now()-s)

    s = datetime.now()
    acc_base_list = []
    total_base_match, total_cnt = 0., 0

    metric_logger = MetricLogger(delimiter="  ")
    print_freq = max(1, int(len(dataloader) / args.print_freq))
    print('print_freq:', print_freq)

    
    results = []
    for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header='')):
        # if data_iter_step == 0:
            # pprint(batch, width=300)

        bsz = len(batch['image'])
        images = batch['image']
        '''
        inputs = processor(images, questions, return_tensors="pt", padding=True).to("cuda")#, torch.float16)
        out = model.generate(**inputs)
        print(f'{i:5d}/{len(dataloader)} : ', processor.batch_decode(out, skip_special_tokens=True))
        '''
        
        """##############################  Baseline Inference   ##############################"""    
        
        text_inputs = get_text_input("default", main_questions=batch['text_input'])
        text_outputs_base, confidences_base = recomposer(images, text_inputs)
        # print(f'{data_iter_step:5d}/{len(dataloader)} : ', outputs[0])

        gt_answers = batch['gt_ans']  # list[bsz, 10]
        batch_acc_list = dataset.get_accuracy(text_outputs_base, gt_answers)
        acc_base_list.extend(batch_acc_list)

        total_base_match += sum(batch_acc_list)
        total_cnt += bsz
        metric_logger.update(n=bsz, base_acc=sum(batch_acc_list)/bsz)
        metric_logger.update(n=bsz, total_base_acc=total_base_match/total_cnt)
        
        """############################## Decompose & Recompose ##############################"""
        
        # generating sub_questions
        text_inputs = get_text_input("decomposer", main_questions=batch['text_input'])
        if args.decomposer_name == "self":  # Image+Text, BLIP-2
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

    json.dump(results, open(os.path.join(args.output_dir, 'results_base.json'), 'w'), indent=2)

    print('inference time : ', datetime.now()-s)
    s = datetime.now()
    
    """##############################     Report metrics     ##############################"""
    results.sort(key=lambda x: x['confidences_base'])
    acc_lba_list = []
    
    

if __name__ == '__main__':
    main()
