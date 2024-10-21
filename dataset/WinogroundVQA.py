import json
import os
from PIL import Image
from pprint import pprint

import torch
from torch.utils.data import DataLoader

try:
    from dataset.base_dataset import BaseDataset
except:
    from base_dataset import BaseDataset



class WinogroundVQADataset(BaseDataset):
    """
    {
        'caption_0': 'an old person kisses a young person',
        'caption_1': 'a young person kisses an old person',
        'collapsed_tag': 'Relation',
        'id': 0,
        'image_0': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1280 at 0x7FC137345610>,
        'image_1': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1920x1280 at 0x7FC1373454F0>,
        'num_main_preds': 1,
        'secondary_tag': '',
        'tag': 'Adjective-Age'
    }
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs): # vqa_acc=True, 
        self.vis_root = vis_root

        if len(ann_paths) == 1:
            ann_path = ann_paths[0]
        elif len(ann_paths) == 2:
            ann_path, sub_qas_path = ann_paths
            if os.path.exists(sub_qas_path):
                self.sub_qas = json.load(open(sub_qas_path, 'r'))
        else:
            raise ValueError(f"Invalid ann_paths: {ann_paths}")
        
        self.annotation = []
        # ann
        from datasets import load_dataset
        samples = load_dataset('facebook/winoground', cache_dir="/data/WinogroundVQA/data/")["test"] # token='your_token_here'

        for sample in samples:
            for i in range(2):
                for j in range(2):
                    img = sample[f'image_{i}']
                    caption = sample[f'caption_{j}']
                    question = f'Does "{caption}" describe the image?'
                    gt_ans = 'yes' if i == j else 'no'
                    
                    ann = {
                        'image': img,
                        'caption': caption,
                        'text_input': question,
                        'gt_ans': gt_ans,
                    }
                    self.annotation.append(ann)
                    
        
        if num_data != -1:
            self.annotation = self.annotation[:num_data]

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._add_instance_ids(key="question_id", prefix="Winoground_")
        
        print("\n" + self.__class__.__name__)
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))

    
    def __getitem__(self, index):
        ann = self.annotation[index]
        question_id = ann["question_id"]
        
        
        sub_qa_list = self.sub_qas[str(question_id)] if hasattr(self, 'sub_qas') else None
        if sub_qa_list is None:
            sub_questions = None
            sub_answers = None
        elif type(sub_qa_list[0]) == list: # include sub_questions and sub_answers
            sub_questions = [sub_qa[0] for sub_qa in sub_qa_list]
            sub_answers = [sub_qa[1] for sub_qa in sub_qa_list]
        else:
            sub_questions = sub_qa_list
            sub_answers = None

        return {
            "vision": ann["image"],
            "text_input": ann["text_input"],
            "question_id": question_id,
            "gt_ans": ann["gt_ans"],
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
        }

def main(ann_paths, split):
    dataset = WinogroundVQADataset(vis_processor=None, text_processor=None, vis_root='dummy_vis_root', 
                             ann_paths=ann_paths, num_data=-1, split=split)
    
    from matplotlib import pyplot as plt
    import torch

    for i in range(len(dataset)):
        print('*' * 100)
        pprint(dataset[i], width=200)
        # ax1 = plt.subplot(2, 1, 1)
        # ax1.title.set_text(dataset[i]['caption_0'])
        # plt.imshow(dataset[i]["image_0"].convert("RGB"))

        # ax2 = plt.subplot(2, 1, 2)
        # ax2.title.set_text(dataset[i]['caption_1'])
        # plt.imshow(dataset[i]["image_1"].convert("RGB"))

        # plt.show()
        # plt.savefig(f'winoground{i}.png')
        break
        
    print('len(dataset):', len(dataset))

if __name__ == '__main__':
    split = 'val' # 'train', 'val', 'test'
    ann_paths = [
        'facebook/winoground'
    ]
    main(ann_paths, split)
    