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


class PathVQADataset(BaseDataset):
    """
    len: 6279
    {
        'image': 'val_0261', 
        'question': 'What have lost their nuclei?', 
        'answer': 'neutrophils'
    }
    """
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=..., num_data=-1, **kwargs):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths, num_data, **kwargs)
        
        # annotations
        ann_path = ann_paths[0]
        with open(ann_path, 'rb') as f:
            self.annotation = pickle.load(f)
        
        if num_data != -1:
            self.annotation = self.annotation[:num_data]
            
        # sub_qa
        if len(ann_paths) == 2:
            sub_qas_path = ann_paths[1]
            with open(sub_qas_path, 'rb') as f:
                self.sub_qas = json.load(open(sub_qas_path, 'r'))
                
        
        self.vis_root = os.path.join(vis_root, kwargs["split"])
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        self._add_instance_ids(key="question_id", prefix="PathVQA_")
        
        
        print("\n" + self.__class__.__name__)
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))
        
        
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        question_id = ann["question_id"]

        image_path = os.path.join(self.vis_root, f'{ann["image"]}.jpg')
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
    dataset = PathVQADataset(vis_processor=None, text_processor=None, vis_root='/data/PathVQA/images/', 
                             ann_paths=ann_paths, num_data=-1, split=split)
    for i in range(len(dataset)):
        pprint(dataset[i], width=200)
        break
    print('len(dataset):', len(dataset))

if __name__ == '__main__':
    split = 'val' # 'train', 'val', 'test'
    ann_paths = [
        f'/data/PathVQA/qas/{split}/{split}_qa.pkl'
    ]
    main(ann_paths, split)
    