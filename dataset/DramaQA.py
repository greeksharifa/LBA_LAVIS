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


class DramaQAEvalDataset(VideoEvalDataset):
            
    def get_image_path(self, vid, supple=False):
        # import pdb; pdb.set_trace()
        if vid.endswith('0000'):
            scene_dir_path = os.path.join(self.vis_root, vid.replace('_', '/'))[:-4] # ex. /data/AnotherMissOh/AnotherMissOh_images/AnotherMissOh01/001/0078
            dir_paths = glob.glob(os.path.join(scene_dir_path, '*/'))

            image_paths = []
            for dir_path in dir_paths:
                image_paths += glob.glob(dir_path + '*.jpg')
        else:
            dir_path = os.path.join(self.vis_root, vid.replace('_', '/'))
            image_paths = glob.glob(os.path.join(dir_path, '*.jpg'))

        return sorted(image_paths)


    def __getitem__(self, index):
        ann = self.annotation[index]
        vid = ann["vid"]
        question_id = ann["qid"]
        
        image_paths = self.get_image_path(vid)
        frms, frms_supple = self.get_frames(image_paths)
            
        question = ann["que"] # question = self.text_processor(ann["que"])
        # gt_ans = self.__class__.ANSWER_MAPPING[ann["correct_idx"]]
        gt_ans = ann["correct_idx"]
        question_type = "Level " + str(ann["q_level_mem"])

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
            "vision_supple": frms_supple,
            "text_input": question,
            "question_id": question_id,
            "gt_ans": gt_ans, #ann["correct_idx"],
            "candidate_list": ann["answers"],
            "answer_sentence": ann["answers"][ann["correct_idx"]],
            "type": question_type,
            "vid": vid,
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
            # "instance_id": ann["instance_id"],
        }
   