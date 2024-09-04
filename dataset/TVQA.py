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


class TVQAEvalDataset(VideoEvalDataset):
       
    def get_image_path(self, vid, random=False):
        dir_path = os.path.join(self.vis_root, 
                                f"{vid.split('_')[0]}_frames" if vid.count('_') == 4 else f"bbt_frames",
                                vid)
        image_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        return sorted(image_paths)

        
    def __getitem__(self, index):
        ann = self.annotation[index]
        vid = ann["video"]
        question_id = ann["qid"]
        
        image_paths = self.get_image_path(vid)
        frms, frms_supple = self.get_frames(image_paths)
        
        question = ann["question"] # question = self.text_processor(ann["que"])
        
        # gt_ans = self.__class__.ANSWER_MAPPING[ann["correct_idx"]]
        gt_ans = ann["answer"]
        
        candidate_list = []
        for i in range(ann["num_option"]):
            candidate_list.append(ann[f'a{i}'])
        
        sub_question_list = self.sub_questions[str(question_id)] if hasattr(self, 'sub_questions') else None
            
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
            "sub_question_list": sub_question_list,
            # "instance_id": ann["instance_id"],
        }
   