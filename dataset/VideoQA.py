import glob
import json
import os
from collections import OrderedDict
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from dataset.video import read_video_pyav

from dataset.base_dataset import BaseDataset


class VideoEvalDataset(BaseDataset):
    
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.vis_root = vis_root
        self.annotation = []
        self.n_frms = kwargs['n_frms'] # default: 4
        
        if len(ann_paths) == 1:
            ann_path = ann_paths[0]
        else:
            ann_path, sub_qas_path = ann_paths
            if os.path.exists(sub_qas_path):
                self.sub_qas = json.load(open(sub_qas_path, 'r'))
            """
            sub_questions: {
                "Interaction_T1_13": [
                    "What objects are present in the living room scene?",
                    ...
                ], 
                ...
            }
            sub_qas: {
                "Interaction_T1_13": [
                    [
                        "What objects are present in the living room scene?",
                        "couch"
                    ],
                    ...
                ],
                ...
            }
            """
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        with open(ann_path, "r") as f:
            if ann_path.endswith('.json'):
                loaded = json.load(f)
            elif ann_path.endswith('.jsonl'):
                loaded = [json.loads(line) for line in f]
            elif ann_path.endswith('.csv'):
                df = pd.read_csv(f)
                loaded = df.to_dict(orient='records')
            else:
                raise ValueError(f"Unsupported file extension: {ann_path}")
            
            if num_data == -1: # use all dataset
                len_loaded = len(loaded)
            else:
                len_loaded = min(len(loaded), num_data)
                
            for i, sample in enumerate(loaded):
                if len(self.annotation) >= len_loaded: # 0 <= num_data <= i:
                    break
                    
                for key in ["video", "vid", "video_id", "video_name"]:
                    if key in sample:
                        vid = sample[key]
                        break
                else:
                    raise ValueError(f"Unsupported vid: {ann_path}, {sample}")
                print(f'\r{i+1:6d}/{len_loaded:6d} : {vid}', end='')
                
                self.annotation.append(sample)

        self._add_instance_ids()
        
        self.ANSWER_MAPPING = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}
        
        print("\n" + self.__class__.__name__)
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))
        
           
    # @staticmethod
    def answer_mapping(self, answer):
        return self.ANSWER_MAPPING[answer]

    
    def image_path_sampling(self, image_paths):
        idxs = np.linspace(0, len(image_paths)-1, self.n_frms, dtype=int)
        return [image_paths[idx] for idx in idxs]
    
    
    def get_frames(self, image_paths):
        image_paths_base = self.image_path_sampling(image_paths)
        frames_base = [np.array(Image.open(img_path)) for img_path in image_paths_base]

        frames_supple = []
        if self.n_supple > 0:
            segment_size = len(image_paths) / self.n_supple
            for i in range(self.n_supple):
                st = int(i * segment_size)
                en = min(max(int((i + 1) * segment_size), st+1), len(image_paths)) # clip [st+1, len(image_paths)]
                image_paths_supple = self.image_path_sampling(image_paths[st:en])
                frames_supple.append([np.array(Image.open(img_path)) for img_path in image_paths_supple])
            
        return frames_base, frames_supple


    def __getitem__(self, index):
        ann = self.annotation[index]
        vid = ann["video"]
        question_id = ann["qid"]
        
        vpath = os.path.join(self.vis_root, f'{vid}.mp4')
        
        # load images. output: list of PIL.Image
        if "start" in ann and "end" in ann:
            frms, frms_supple = read_video_pyav(vpath, n_frms=self.n_frms, n_supple=self.n_supple, start_time=ann["start"], end_time=ann["end"])
        else:
            frms, frms_supple = read_video_pyav(vpath, n_frms=self.n_frms, n_supple=self.n_supple)
        
        question = ann["question"] # question = self.text_processor(ann["que"])
        
        # gt_ans = self.__class__.ANSWER_MAPPING[ann["correct_idx"]]
        gt_ans = ann["answer"]
        
        candidate_list = []
        for i in range(ann["num_option"]):
            candidate_list.append(ann[f'a{i}'])
            
        question_type = ann['qid'].split('_')[0]
        # NExTQA : Causal, Temporal, Descriptive -> C, T, D
        # STAR   : Interaction, Sequence, Prediction, Feasibility 
        
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
            "vision": frms, # frms, # list of ndarray, 즉 video랑 비슷
            "vision_supple": frms_supple, # list of list of ndarray
            "text_input": question,
            "question_id": question_id,
            "gt_ans": gt_ans,
            "candidate_list": candidate_list,
            "answer_sentence": candidate_list[gt_ans],
            "type": question_type,
            "vid": vid,
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
            # "instance_id": ann["instance_id"],
        }
     
