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



class VQA_radDataset(BaseDataset):
    """
    {
        'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1024x1291 at 0x7F73A41E37C0>, 
        'question': 'is there evidence of an aortic aneurysm?', 
        'answer': 'yes'
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
        samples = load_dataset('flaviagiammarino/vqa-rad', cache_dir="/data/VQA_rad/")["test"] # token='your_token_here'
        
        for sample in samples:
            img = sample['image']
            question = sample['question']
            gt_ans = sample['answer']
            
            ann = {
                'image': img,
                'text_input': question,
                'gt_ans': gt_ans,
            }
            self.annotation.append(ann)

        
        if num_data != -1:
            self.annotation = self.annotation[:num_data]

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        # self.vqa_acc = vqa_acc
        for k, v in kwargs.items():
            setattr(self, k, v)

        # import pdb; pdb.set_trace()
        self._add_instance_ids(key="question_id", prefix="VQA_rad_")
        
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
    dataset = VQA_radDataset(vis_processor=None, text_processor=None, vis_root='dummy_vis_root', 
                             ann_paths=ann_paths, num_data=-1, split=split)
    
    from matplotlib import pyplot as plt
    import torch

    for i in range(len(dataset)):
        print('*' * 100)
        pprint(dataset[i], width=200)
        break
        
    print('len(dataset):', len(dataset))

if __name__ == '__main__':
    split = 'val' # 'train', 'val', 'test'
    ann_paths = [
        'flaviagiammarino/vqa-rad'
    ]
    main(ann_paths, split)
    