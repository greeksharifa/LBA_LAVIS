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


class MSVDQAEvalDataset(VideoEvalDataset):
    """
    <class 'list'>
    len: 12278
    
    val_q.json
    [
        {
            'answer': 'couch', 
            'category_id': 14, 
            'id': 158581, 
            'question': 'what are three people sitting on?', 
            'video_id': 6513
        }, 
        {'answer': 'coversation', 'category_id': 14, 'id': 158582, 'question': 'what is a family having?', 'video_id': 6513},
        ...
    ]
    """
                
    def __getitem__(self, index):
        ann = self.annotation[index]
        vid = ann["video_id"]
        question_id = str(ann["id"])
        
        vpath = os.path.join(self.vis_root, f'video{vid}.mp4')
        
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
            "type": ann['category_id'],
            "vid": vid,
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
            # "instance_id": ann["instance_id"],
        }
   