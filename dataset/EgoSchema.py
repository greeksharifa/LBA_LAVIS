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


class EgoSchemaEvalDataset(VideoEvalDataset):
    """
,video_name,question_id,question,answer,a0,a1,a2,a3,a4
1,0074f737-11cb-497d-8d07-77c3a8127391,1ZdZ8aUcBNzndj135bqFrxb9L816EMGp1,"Taking into account all the actions performed by c, what can you deduce about the primary objective and focus within the video content?",C is cleaning dishes.,C is cooking.,C is doing laundry.,C is cleaning the kitchen.,C is cleaning dishes.,C is cleaning the bathroom.
2,00b9a0de-c59e-49cb-a127-6081e2fb8c8e,1bVNvPX6BNPIqcqMk-ZJr6WLsXNQybg64,"What was the primary purpose of the cup of water in this video, and how did it contribute to the overall painting process?",To clean the paintbrush.,To provide a source of water for the paintbrush.,To provide a place to store the paintbrush.,To provide a place to dispose of the paintbrush.,To provide a place to rest the paintbrush.,To clean the paintbrush.
3,00f93e1e-cf4e-4835-88b4-4ad68216e86f,1X6J8bS-ntnAqB7zmXpxi4X4esnbe2U5g,"What is the overarching theme of the video, considering the activities performed by both characters?","The overarching theme of the video is that people can be both engaged in challenging activities and enjoying leisurely activities at the same time. the video shows that it is possible to be both productive and relaxed, and that it is important to find a balance between the two.","The overarching central theme presented in the video is that individuals can be both sociable and independent simultaneously. the visual content demonstrates that it is entirely possible to be both connected to others meaningfully and to savor solitary moments, emphasizing that it is crucial to find a harmonious balance between these two aspects.","The overarching theme of the video is that people can be both engaged in challenging activities and enjoying leisurely activities at the same time. the video shows that it is possible to be both productive and relaxed, and that it is important to find a balance between the two.","The primary, overarching theme presented in the video emphasizes that individuals can truly be both creative and practical simultaneously. the enlightening video demonstrates the realistic possibility of being both highly imaginative and remarkably efficient, while stressing the significance of discovering an equilibrium between these two essential aspects.","The overarching theme of the video is that people can be both ambitious and humble. the video shows that it is possible to be both driven and modest, and that it is important to find a balance between the two.","The primary overarching theme presented in the video is that individuals can simultaneously possess and exhibit both intelligence and emotional aspects. effectively, the video demonstrates that the coexistence of rational and intuitive qualities is feasible, emphasizing the significance of establishing equilibrium between these two crucial elements."
    """
    def __getitem__(self, index):
        ann = self.annotation[index]
        vid = ann["video_name"]
        question_id = ann["question_id"]
        
        vpath = os.path.join(self.vis_root, f'{vid}.mp4')
        
        # load images. output: list of PIL.Image
        if "start" in ann and "end" in ann:
            frms, frms_supple = read_video_pyav(vpath, n_frms=self.n_frms, n_supple=self.n_supple, start_time=ann["start"], end_time=ann["end"])
        else:
            frms, frms_supple = read_video_pyav(vpath, n_frms=self.n_frms, n_supple=self.n_supple)
        
            
        question = ann["question"].lstrip('"').rstrip('"')
        
        gt_ans = ann["answer"].lstrip('"').rstrip('"')
        candidate_list = []
        for i in range(5):
            candidate = ann[f'a{i}'].lstrip('"').rstrip('"')
            candidate_list.append(candidate)
            if candidate == gt_ans:
                gt_ans = i
        
        
        question_type = "EgoSchema"

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
   