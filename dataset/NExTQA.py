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


class NExTQAEvalDataset(VideoEvalDataset):
    '''
    <class 'list'>
    4996
    {'num_option': 5, 
    'qid': 'TN_6233408665_8', 
    'question': 'what did the people on the sofa do after the lady in pink finished singing?', 
    'video': '1150/6233408665',
    'a0': 'sitting.', 
    'a1': 'give it to the girl.', 
    'a2': 'take music sheet.', 
    'a3': 'clap.', 
    'a4': 'walk in circles.', 
    'answer': 3}
    '''
    def __getitem__(self, index):
        ann = self.annotation[index]

        vid = ann["video"]
        vpath = os.path.join(self.vis_root, f'{vid}.mp4')
        
        # load images. output: list of PIL.Image
        frms = read_video_pyav(vpath, n_frms=self.n_frms)
        
        question = ann["question"] # question = self.text_processor(ann["que"])
        
        # gt_ans = self.__class__.ANSWER_MAPPING[ann["correct_idx"]]
        gt_ans = ann["answer"]
        
        candidate_list = []
        for i in range(ann["num_option"]):
            candidate_list.append(ann[f'a{i}'])

        return {
            "image": frms, # frms, # 이름은 image지만 list of ndarray, 즉 video랑 비슷
            "text_input": question,
            "question_id": ann["qid"],
            "gt_ans": gt_ans,
            "candidate_list": candidate_list,
            "answer_sentence": candidate_list[gt_ans],
            # "instance_id": ann["instance_id"],
        }
     
