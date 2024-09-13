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


class IntentQAEvalDataset(VideoEvalDataset):
    """
video_id,frame_count,width,height,question,answer,qid,type,a0,a1,a2,a3,a4,action,lemma,lemma_id
13884124143,853,640,360,how did the guy on the board managed to not fall off from the board at the start of the video,4,1,CH,jump,nod,push the chair away,playing the piano,move left right to balance,left,leave,130
7093500213,1110,640,360,what does the man in checkered do after walking onto the stage with microphone stands at the start,1,6,TN,bow to people,set up the stand,take away the stand,takes out some paper,hands him a bottle,walking,walk,159
5465138661,936,320,240,why does the girl in jeans bend down at the start of the video,3,1,CW,dance move,prevent jeans from getting wet,tired of bouncing,play with baby,pick up snow,bend,bend,156
4254574573,975,640,360,why does the user touch the screen,4,2,CW,it is a barrier,hand gesture for bear,typing,live webcam,to check the effects,touch,touch,154
    """
    def __getitem__(self, index):
        ann = self.annotation[index]
        vid = ann["video_id"]
        question_id = f'{vid}_{ann["type"]}_{ann["qid"]}'
        
        vpath = os.path.join(self.vis_root, f'{vid}.mp4')
        
        # load images. output: list of PIL.Image
        if "start" in ann and "end" in ann:
            frms, frms_supple = read_video_pyav(vpath, n_frms=self.n_frms, n_supple=self.n_supple, start_time=ann["start"], end_time=ann["end"])
        else:
            frms, frms_supple = read_video_pyav(vpath, n_frms=self.n_frms, n_supple=self.n_supple)
        
            
        question = ann["question"].rstrip('?') + '?'
        
        gt_ans = ann["answer"]
        candidate_list = []
        for i in range(5):
            candidate_list.append(ann[f'a{i}'])
        
        question_type = str(ann["type"])

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
   