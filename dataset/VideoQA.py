import glob
import json
import os
from collections import OrderedDict
from PIL import Image

import numpy as np
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
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        with open(ann_paths[0], "r") as f:
            loaded = json.load(f)
            
            if num_data == -1: # use all dataset
                len_loaded = len(loaded)
            else:
                len_loaded = min(len(loaded), num_data)
                
            for i, sample in enumerate(loaded):
                if len(self.annotation) >= len_loaded: # 0 <= num_data <= i:
                    break
                    
                vid = sample["video"]
                print(f'\r{i:6d}/{len_loaded:6d} : {vid}', end='')
                
                self.annotation.append(sample)

        self._add_instance_ids()
        
        print("\n" + self.__class__.__name__)
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))
        
           
    @staticmethod
    def answer_mapping(answer):
        ANSWER_MAPPING = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}
        return ANSWER_MAPPING[answer]

        
    def collater(self, samples):
        (
            image_list,
            text_input_list,
            question_id_list,
            gt_ans_list,
            candidate_list_list,
            answer_sentence_list,
        ) = ([], [], [], [], [], [])
        
        for sample in samples:
            image_list.append(sample["image"])
            text_input_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            gt_ans_list.append(sample["gt_ans"])
            candidate_list_list.append(sample["candidate_list"])
            answer_sentence_list.append(sample["answer_sentence"])
            
        return {
            "image": image_list, #torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "question_id": question_id_list,
            "gt_ans": gt_ans_list, 
            "candidate_list": candidate_list_list,
            "answer_sentence": answer_sentence_list,
        }
  