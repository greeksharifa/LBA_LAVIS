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
from dataset.video import read_video_pyav, process_video_cv2

from dataset.VideoQA import VideoEvalDataset


class STAREvalDataset(VideoEvalDataset):
    '''
    <class 'list'>
    7098 {'Interaction': 2398, 'Sequence': 3586, 'Prediction': 624, 'Feasibility': 490}
    {'video': '6H78U',
    'num_option': 4,
    'qid': 'Interaction_T1_13',
    'a0': 'The closet/cabinet.',
    'a1': 'The blanket.',
    'a2': 'The clothes.',
    'a3': 'The table.',
    'answer': 2,
    'question': 'Which object was tidied up by the person?',
    'start': 11.1,
    'end': 19.6}
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
            
        question_type = ann['qid'].split('_')[0]

        return {
            "image": frms, # frms, # 이름은 image지만 list of PIL.Image, 즉 video랑 비슷
            "text_input": question,
            "question_id": ann["qid"],
            "gt_ans": gt_ans,
            "candidate_list": candidate_list,
            "answer_sentence": candidate_list[gt_ans],
            # "instance_id": ann["instance_id"],
        }
     
