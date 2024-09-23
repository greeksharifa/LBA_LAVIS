import json
import os
import pickle
from PIL import Image
from pprint import pprint

import torch
from torch.utils.data import DataLoader

try:
    from dataset.base_dataset import BaseDataset
except:
    from base_dataset import BaseDataset


class ArtVQADataset(BaseDataset):
    """
    len: 5124
    {
        'image': '42809-3baptist.jpg', 
        'question': 'what stand next to each other', 
        'answer': 'people', 
        'need_external_knowledge': False
    }
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_data=num_data, **kwargs)
        
        self._add_instance_ids(key='question_id', prefix='ArtVQA_')
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        question_id = ann["question_id"]

        image_path = os.path.join(self.vis_root, f'{ann["image"]}')
        image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        # text_input = self.text_processor(ann["question"])
        text_input = ann["question"]
        
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
            "vision": image,
            "text_input": text_input,
            "question_id": question_id,
            "gt_ans": ann["answer"], # vqav2 answers list of str(len=10)
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
        }
        

def main(ann_paths, split):
    dataset = ArtVQADataset(vis_processor=None, text_processor=None, vis_root='/data/ArtVQA/SemArt/Images/', 
                             ann_paths=ann_paths, num_data=-1, split=split)
    for i in range(len(dataset)):
        pprint(dataset[i], width=200)
        break
    print('len(dataset):', len(dataset))

if __name__ == '__main__':
    split = 'val' # 'train', 'val', 'test'
    ann_paths = [
        '/data/ArtVQA/val.json'
    ]
    main(ann_paths, split)
    