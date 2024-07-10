import argparse
from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader

from dataset.VQA_Introspect import VQAIntrospectDataset
from recomposer import Recomposer

from utils.misc import SmoothedValue, MetricLogger


def get_args():
    parser = argparse.ArgumentParser(description='LBA method')

    # model
    parser.add_argument('--model_name', type=str,
                        default='Salesforce/blip2-flan-t5-xl', help='model name')

    # path
    parser.add_argument('--dataset_path', type=str,
                        default='/data1/VQA-Introspect/VQAIntrospect_valv1.0.json', help='path to the dataset')
    parser.add_argument('--vis_root', type=str, default='/data1/coco/images/',
                        help='Root directory of images (e.g. coco/images/)')

    '''
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')
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
    recomposer = Recomposer(args.model_name)
    print('recomposer loading time : ', datetime.now()-s)

    s = datetime.now()
    acc_list = []
    match, cnt = 0., 0

    metric_logger = MetricLogger(delimiter="  ")
    print_freq = int(len(dataloader) / 50)

    acc_lv = {i+1: 0 for i in range(4)}
    tot_lv = {i+1: 0 for i in range(4)}

    for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, print_freq, header='')):
        # for i, batch in enumerate(dataloader):
        # if data_iter_step >= 5:
        # break
        # from pprint import pprint
        # pprint(batch, width=300)

        bsz = len(batch['image'])
        images = batch['image']
        questions = batch['text_input']

        outputs = recomposer(images, questions)
        # print(f'{data_iter_step:5d}/{len(dataloader)} : ', outputs[0])

        gt_answers = batch['gt_ans']  # list[bsz, 10]
        batch_acc_list = dataset.get_accuracy(outputs[0], gt_answers)
        acc_list.extend(batch_acc_list)

        match += sum(batch_acc_list)
        cnt += bsz

        metric_logger.update(n=bsz, acc=sum(batch_acc_list)/bsz)
        metric_logger.update(n=bsz, total_acc=match/cnt)
        # print('sum(batch_acc_list)/bsz:', sum(batch_acc_list)/bsz)
        # print('match/cnt:', match/cnt)

        # inputs = processor(images, questions, return_tensors="pt", padding=True).to("cuda")#, torch.float16)
        # out = model.generate(**inputs)
        # print(f'{i:5d}/{len(dataloader)} : ', processor.batch_decode(out, skip_special_tokens=True))

    print('inference time : ', datetime.now()-s)


if __name__ == '__main__':
    main()
