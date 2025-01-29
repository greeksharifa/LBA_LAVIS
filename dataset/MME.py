import json
import os
import string
from PIL import Image
import pickle
from pprint import pprint
import ast

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets


try:
    from dataset.base_dataset import BaseDataset
except:
    from base_dataset import BaseDataset


DOMAIN_CAT2SUB_CAT = {
  'Art and Design': ['Art', 'Art_Theory', 'Design', 'Music'],
  'Business': ['Accounting', 'Economics', 'Finance', 'Manage','Marketing'],
  'Science': ['Biology', 'Chemistry', 'Geography', 'Math', 'Physics',],
  'Health and Medicine': ['Basic_Medical_Science', 'Clinical_Medicine', 'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health'],
  'Humanities and Social Science': ['History', 'Literature', 'Sociology', 'Psychology'],
  'Tech and Engineering': ['Agriculture', 'Architecture_and_Engineering', 'Computer_Science', 'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering'],
}
CAT_SHORT2LONG = {
    'acc': 'Accounting',
    'agri': 'Agriculture',
    'arch': 'Architecture_and_Engineering',
    'art': 'Art',
    'art_theory': 'Art_Theory',
    'bas_med': 'Basic_Medical_Science',
    'bio': 'Biology',
    'chem': 'Chemistry',
    'cli_med': 'Clinical_Medicine',
    'cs': 'Computer_Science',
    'design': 'Design',
    'diag_med': 'Diagnostics_and_Laboratory_Medicine',
    'econ': 'Economics',
    'elec': 'Electronics',
    'ep': 'Energy_and_Power',
    'fin': 'Finance',
    'geo': 'Geography',
    'his': 'History',
    'liter': 'Literature',
    'manage': 'Manage',
    'mark': 'Marketing',
    'mate': 'Materials',
    'math': 'Math',
    'mech': 'Mechanical_Engineering',
    'music': 'Music',
    'phar': 'Pharmacy',
    'phys': 'Physics',
    'psy': 'Psychology',
    'pub_health': 'Public_Health',
    'socio': 'Sociology'
}

class MMEDataset(BaseDataset):
    """
    {
        10002.jpg	Does this artwork exist in the form of painting? Please answer yes or no.	Yes
        10002.jpg	Does this artwork exist in the form of glassware? Please answer yes or no.	No
        10049.jpg	Does this artwork exist in the form of painting? Please answer yes or no.	Yes
        10049.jpg	Does this artwork exist in the form of sculpture? Please answer yes or no.	No
        10256.jpg	Does this artwork exist in the form of painting? Please answer yes or no.	Yes
        10256.jpg	Does this artwork exist in the form of sculpture? Please answer yes or no.	No
        ...
    }
    """
    # question 수:
    """
    artwork: ...
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs): # vqa_acc=True, 
        self.vis_root = vis_root
        print('ann_paths : ', ann_paths)
        
        if ann_paths[-1].endswith('.json'): # sub_qas file
            if os.path.exists(ann_paths[-1]):
                self.sub_qas = json.load(open(ann_paths[-1], 'r'))
            ann_paths = ann_paths[:-1]
        
        self.ann_root_dir = os.path.dirname(ann_paths[0]) # /data/MME/eval_tool/Your_Results/
        self.annotation = []
        
        split = kwargs.get('split', 'val')
        print('MME split : ', split)
        
        # for eval_tool
        self.mme_eval_results = {}
        
        for ann_path in ann_paths:
            lines = open(ann_path, 'r').readlines()
            # remove empty lines
            lines = [line.strip() for line in lines if line.strip() != '']
            category = os.path.basename(ann_path).split('.')[0] # ex) scene.txt -> scene
            self.mme_eval_results[category] = lines
            
            for i, line in enumerate(lines):
                image_filename, question, gt_ans = line.strip().split('\t')
                dir_name = os.path.basename(ann_path).split('.')[0]
                if os.path.exists(os.path.join(vis_root, dir_name, image_filename)):
                    image_path = os.path.join(vis_root, dir_name, image_filename)
                else:
                    image_path = os.path.join(vis_root, dir_name, 'images', image_filename)
                image = Image.open(image_path).convert('RGB')
                
                question_id = f'{dir_name}_{i}'
                
                ann = {
                    "image": image,
                    "text_input": question,
                    "question_id": question_id,
                    "gt_ans": gt_ans,
                    "image_path": image_path,
                }
                
                self.annotation.append(ann)
       
       
        if num_data != -1:
            if num_data < len(self.annotation):
                # uniform_sampling
                idxs = np.linspace(0, len(self.annotation)-1, num_data, dtype=int)
                self.annotation = [self.annotation[i] for i in idxs]
                # self.annotation = self.annotation[:num_data]

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        for k, v in kwargs.items():
            setattr(self, k, v)

        # self._add_instance_ids(key="question_id", prefix="MMMU_")
        
        print("\n" + self.__class__.__name__)
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))
        self.cnt = 0

    
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
            "image_path": ann["image_path"],
        }



def main(ann_paths, split):
    dataset = MMEDataset(vis_processor=None, text_processor=None, vis_root='MME/MME_Benchmark/', 
                             ann_paths=ann_paths, num_data=-1, split=split)
    
    from matplotlib import pyplot as plt
    import torch

    for i in range(len(dataset)):
        print('*' * 100)
        pprint(dataset[i], width=200)
        break
        
    print('len(dataset):', len(dataset))


if __name__ == '__main__':
    split = 'dev' # 'dev', 'validation', 'test'
    ann_paths = [
        'MME/eval_tool/Your_Results/scene.txt',
        'MME/eval_tool/Your_Results/code_reasoning.txt',
        'MME/eval_tool/Your_Results/posters.txt',
        'MME/eval_tool/Your_Results/count.txt',
        'MME/eval_tool/Your_Results/artwork.txt',
        'MME/eval_tool/Your_Results/color.txt',
        'MME/eval_tool/Your_Results/landmark.txt',
        'MME/eval_tool/Your_Results/position.txt',
        'MME/eval_tool/Your_Results/existence.txt',
        'MME/eval_tool/Your_Results/numerical_calculation.txt',
        'MME/eval_tool/Your_Results/OCR.txt',
        'MME/eval_tool/Your_Results/celebrity.txt',
        'MME/eval_tool/Your_Results/commonsense_reasoning.txt',
        'MME/eval_tool/Your_Results/text_translation.txt',
        'MME/sub_qas_val_hf_beam_and_greedy_N1_processed.json',
    ]
    main(ann_paths, split)
    