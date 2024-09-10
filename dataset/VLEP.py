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


class VLEPEvalDataset(VideoEvalDataset):
    """
    <class 'list'>
    total 4392
    {'a0': 'Ross will stop, turn and point at Monica.',
    'a1': 'Ross will stop and ask Monica why she is pointing at him.',
    'answer': 0,
    'end': 40.37,
    'num_option': 2,
    'qid': 'VLEP_20142',
    'start': 38.81,
    'video': 'friends_s03e09_seg02_clip_07_ep'}
    """

    def get_image_path(self, vid, random=False):
        dir_path = os.path.join(self.vis_root, vid)
        image_paths = glob.glob(os.path.join(dir_path, '*.jpg'))

        return sorted(image_paths)
        
    def __getitem__(self, index):
        ann = self.annotation[index]
        vid = ann["video"]
        question_id = ann["qid"]
        
        image_paths = self.get_image_path(vid)
        frms, frms_supple = self.get_frames(image_paths)
        
        
        question = "Which event is more likely to happen right after?" # event prediction
        
        # gt_ans = self.__class__.ANSWER_MAPPING[ann["correct_idx"]]
        gt_ans = ann["answer"]
        
        candidate_list = []
        for i in range(ann["num_option"]):
            candidate_list.append(ann[f'a{i}'])
            
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
            "text_input": question,
            "question_id": question_id,
            "gt_ans": gt_ans,
            "candidate_list": candidate_list,
            "answer_sentence": candidate_list[gt_ans],
            # "type": question_type,
            "vid": vid,
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
            # "instance_id": ann["instance_id"],
        }
   