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


class MSVDQADataset(VideoEvalDataset):
    """
    <class 'list'>
    len: 6415
    
    val_qa.json
    [
        {
            'answer': 'someone', 
            'id': 30933, 
            'question': 'who pours liquid from a plastic container into a ziploc bag containing meat pieces?', 
            'video_id': 1201
        }, 
        {'answer': 'man', 'id': 30934, 'question': 'who pours a seasoning liquid from a plastic container over chicken pieces placed in a plastic pouch?', 'video_id': 1201},
        ...
    ]
    
    youtube_mapping.txt (if loaded by readlines())
    [
        '-4wsuPCjDBc_5_15 vid1\n', 
        '-7KMZQEsJW4_205_208 vid2\n', 
        '-8y1Q0rA3n8_108_115 vid3\n',
        ...
        'zzit5b_-ukg_5_20 vid1970'
    ]
    """
    
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.vis_root = vis_root
        self.annotation = []
        self.n_frms = kwargs['n_frms'] # default: 4
        
        if len(ann_paths) == 2:
            ann_path, youtube_mapping_path = ann_paths
        else:
            ann_path, youtube_mapping_path, sub_qas_path = ann_paths
            if os.path.exists(sub_qas_path):
                self.sub_qas = json.load(open(sub_qas_path, 'r'))
                
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        for k, v in kwargs.items():
            setattr(self, k, v)
            
        qa_data = json.load(open(ann_path, 'r'))
        map_data = open(youtube_mapping_path, 'r').readlines()
        
        for qa in qa_data:
            original_vid = qa["video_id"]
            vid = map_data[original_vid - 1].split()[0]
            
            qa["video_id"] = vid
            qa["original_vid"] = f'vid{original_vid}'
            
            self.annotation.append(qa)
            
        
        if num_data != -1:
            if num_data < len(self.annotation):
                # uniform_sampling
                idxs = np.linspace(0, len(self.annotation)-1, num_data, dtype=int)
                self.annotation = [self.annotation[i] for i in idxs]
                # self.annotation = self.annotation[:num_data]

            
        if kwargs.get("eval_chatgpt", False):
            print("eval_chatgpt")
            self.create_openai_client()
        
        self._add_instance_ids()
                
        print("\n" + self.__class__.__name__)
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))

                
    def __getitem__(self, index):
        ann = self.annotation[index]
        vid = ann["video_id"]
        question_id = str(ann["id"])
        
        vpath = os.path.join(self.vis_root, f'{vid}.avi')
        
        # load images. output: list of PIL.Image
        if "start" in ann and "end" in ann:
            frms, frms_supple = read_video_pyav(vpath, n_frms=self.n_frms, n_supple=self.n_supple, start_time=ann["start"], end_time=ann["end"])
        else:
            frms, frms_supple = read_video_pyav(vpath, n_frms=self.n_frms, n_supple=self.n_supple)
        
        question = ann["question"] # question = self.text_processor(ann["que"])
        
        # gt_ans = self.__class__.ANSWER_MAPPING[ann["correct_idx"]]
        gt_ans = ann["answer"]
        
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
            "vision": frms, # frms, # 이름은 image지만 list of ndarray, 즉 video랑 비슷
            "vision_supple": frms_supple, # list of list of ndarray
            # "vpath": vpath,
            "text_input": question,
            "question_id": question_id,
            "gt_ans": gt_ans,
            "candidate_list": None,
            # "answer_sentence": candidate_list[gt_ans],
            # "type": ann['type'],
            "vid": vid,
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
            # "instance_id": ann["instance_id"],
        }
   