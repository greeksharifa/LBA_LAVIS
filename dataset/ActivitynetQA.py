import glob
import json
import os
from collections import OrderedDict
from PIL import Image

import numpy as np
import torch
from torchvision import transforms

# from multimodal_classification_datasets import MultimodalClassificationDataset
# from utils.load_video import load_video_to_sampled_frames
from dataset.video import read_video_pyav

from dataset.VideoQA import VideoEvalDataset


class ActivityNetQAEvalDataset(VideoEvalDataset):
    """
    <class 'list'>
    len: 18000
    
    val_q.json
    [
        {
            'video_name': 'TIEzvhv6xaI', 
            'question': 'is the no.3 athlete playing indoor', 
            'question_id': 'v_TIEzvhv6xaI_2'
        },
        {'video_name': '7X3wPRKuAsU', 'question': 'is the athlete wearing long sleeve', 'question_id': 'v_7X3wPRKuAsU_3'},
        ...
    ]
    val_a.json
    [
        {
            'answer': 'yes', 
            'type': 3, 
            'question_id': 'v_TIEzvhv6xaI_2'
        },
        {'answer': 'no', 'type': 3, 'question_id': 'v_7X3wPRKuAsU_3'},
        ...
    ]
    """
    
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.vis_root = vis_root
        self.annotation = []
        self.n_frms = kwargs['n_frms'] # default: 4
        
        if len(ann_paths) == 2:
            ann_q_path, ann_a_path = ann_paths
        else:
            ann_q_path, ann_a_path, sub_qas_path = ann_paths
            if os.path.exists(sub_qas_path):
                self.sub_qas = json.load(open(sub_qas_path, 'r'))
                
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        q_data = json.load(open(ann_q_path, 'r'))
        a_data = json.load(open(ann_a_path, 'r'))
        
        
        if num_data == -1: # use all dataset
            len_loaded = len(q_data)
        else:
            len_loaded = min(len(q_data), num_data)
        
        for i, (q, a) in enumerate(zip(q_data, a_data)):
            assert q['question_id'] == a['question_id']
            self.annotation.append({
                'video': q['video_name'],
                'question': q['question'],
                'question_id': q['question_id'],
                'answer': a['answer'],
                'type': a['type'],
            })
            if len(self.annotation) >= len_loaded: # 0 <= num_data <= i:
                break
        
        self._add_instance_ids()
                
        print("\n" + self.__class__.__name__)
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))
        
                
    def __getitem__(self, index):
        ann = self.annotation[index]
        vid = ann["video"]
        question_id = ann["question_id"]
        
        vpath = os.path.join(self.vis_root, f'v_{vid}.mp4')
        
        # load images. output: list of PIL.Image
        if "start" in ann and "end" in ann:
            frms, frms_supple = read_video_pyav(vpath, n_frms=self.n_frms, n_supple=self.n_supple, start_time=ann["start"], end_time=ann["end"])
        else:
            frms, frms_supple = read_video_pyav(vpath, n_frms=self.n_frms, n_supple=self.n_supple)
        
        question = ann["question"] # question = self.text_processor(ann["que"])
        
        # gt_ans = self.__class__.ANSWER_MAPPING[ann["correct_idx"]]
        gt_ans = ann["answer"]
        
        sub_qa_list = self.sub_qas[str(question_id)] if hasattr(self, 'sub_qas') else None
        if type(sub_qa_list[0]) == list: # include sub_questions and sub_answers
            sub_questions = [sub_qa[0] for sub_qa in sub_qa_list]
            sub_answers = [sub_qa[1] for sub_qa in sub_qa_list]
        else:
            sub_questions = sub_qa_list
            sub_answers = None
            
        return {
            "vision": frms, # frms, # 이름은 image지만 list of ndarray, 즉 video랑 비슷
            "vision_supple": frms_supple, # list of list of ndarray
            # "vpath": vpath,
            "text_input": question,
            "question_id": question_id,
            "gt_ans": gt_ans,
            "type": ann['type'],
            "vid": vid,
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
            # "instance_id": ann["instance_id"],
        }
   