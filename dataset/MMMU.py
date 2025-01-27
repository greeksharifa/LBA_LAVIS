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

class MMMUDataset(BaseDataset):
    """
    {
        'id': 'validation_Accounting_1',
        'question': '<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?',
        'options': "['$6', '$7', '$8', '$9']",
        'explanation': '',
        'image_1': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=733x237>,
        'image_2': None,
        'image_3': None,
        'image_4': None,
        'image_5': None,
        'image_6': None,
        'image_7': None,
        'img_type': "['Tables']",
        'answer': 'B',
        'topic_difficulty': 'Medium',
        'question_type': 'multiple-choice',
        'subfield': 'Managerial Accounting'
    }
    """
    # image 수:
    """
    split: dev       : Counter({1: 146, 2: 2, 4: 2}) 
    split: validation: Counter({1: 857, 2: 24, 4: 8, 5: 6, 3: 5}) 
    split: test      : Counter({1: 9702, 2: 409, 4: 202, 5: 110, 3: 67, 6: 8, 7: 2}) 
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs): # vqa_acc=True, 
        self.vis_root = vis_root
        print('ann_paths : ', ann_paths)

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
        print('ann_paths : ', ann_paths)
        # samples = load_dataset('MMMU/MMMU', cache_dir="/data/MMMU/")["dev"] # token='your_token_here'
        # samples = pickle.load(open(ann_path, 'rb'))[kwargs.get('split', 'val')]
        
        # run for each subject
        split = kwargs.get('split', 'validation')
        if split == 'val':
            split = 'validation'
        # split = 'validation'
        print('MMMU split : ', split)
        
        
        samples = json.load(open(ann_path, 'r'))
        """
        [
            {   
                'question_id': 'validation_Accounting_1',
                'question': '<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. 
                            The following cost data represents average variable costs per unit for 25,000 units of production. 
                            If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?',
                'options': ['$6', '$7', '$8', '$9'],
                'answer': 'B',
                'image_path_list': ['/data/MMMU/mmmu_images/validation/validation_Accounting_1_1.png'],
                'question_type': 'multiple-choice',
                
                'explanation': '',
                'img_type': ['Tables'],
                'subfield': 'Managerial Accounting',
                'topic_difficulty': 'Medium'
            },
            ...
            # 간혹 'question_type': "open" 도 있음. 제외?
        ]
        """
        for sample in samples:
            if sample['question_type'] == 'open':
                continue
            question = sample['question']
            images = []
            for j, image_path in enumerate(sample['image_path_list']):
                images.append(Image.open(image_path))
                question = question.replace(f'<image {j}>', '')
                
            gt_ans = sample['answer']
            if gt_ans in string.ascii_lowercase + string.ascii_uppercase:
                gt_ans = '(' + gt_ans + ')'

            ann = {
                "image_list": images,
                "text_input": question,
                "question_id": sample['question_id'],
                "candidate_list": sample['options'],
                "gt_ans": gt_ans,
            }
            
            self.annotation.append(ann)
        
        '''
        if False:
            sub_dataset_list = []
            for i, subject in enumerate(CAT_SHORT2LONG.values()):
                print(f'loading sub_dataset [{i:2d}/{len(CAT_SHORT2LONG)}] : {subject}', end='\r')
                sub_dataset = load_dataset("MMMU/MMMU", subject, split=split, cache_dir="/data/MMMU/hf_datasets_local_downloaded/")
                sub_dataset_list.append(sub_dataset)
                # break

            # merge all dataset
            samples = concatenate_datasets(sub_dataset_list)

            for sample in samples:
                question = sample['question']
                images = []
                for j in range(1, 8):
                    if sample[f'image_{j}'] is not None:
                        images.append(sample[f'image_{j}'])
                        question = question.replace(f'<image_{j}>', '')

                ann = {
                    "image_list": images,
                    "text_input": question,
                    "question_id": sample['id'],
                    "candidate_list": ast.literal_eval(sample['options']),
                    "gt_ans": sample['answer'],
                }
                
                self.annotation.append(ann)
        '''
        
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
            "vision": ann["image_list"],
            "text_input": ann["text_input"],
            "question_id": question_id,
            "gt_ans": ann["gt_ans"],
            "candidate_list": ann["candidate_list"],
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
        }

def main(ann_paths, split):
    dataset = MMMUDataset(vis_processor=None, text_processor=None, vis_root='dummy_vis_root', 
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
        '/data/MMMU/mmmu_dataset_concatenated.pkl'
        # 'MMMU/MMMU'
    ]
    main(ann_paths, split)
    