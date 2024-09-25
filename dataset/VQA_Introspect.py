import json
import os
from PIL import Image
from pprint import pprint

import torch
from torch.utils.data import DataLoader

try:
    from dataset.base_dataset import BaseDataset
except:
    from base_dataset import BaseDataset


class VQAIntrospectDataset(BaseDataset):
    """
    {   
        'gt_ans': ['yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes'],
        'gt_sub_qas': [('is there sand and whitewater?', 'yes'), ('are the woman on the horses near to the sea?', 'yes')],
        'question_id': '565248001',
        'reasoning_answer_most_common': 'yes',
        'sub_answer_list': ['yes', 'riding horses', 'sunny', 'yes', 'yes'],
        'sub_question_list': ['Yes there is sand?', 'Yes they are?', 'What is the weather like?', 'Are they at the beach?', 'Yes they are at the beach?'],
        'text_input': 'are they at the beach?',
        'vision': <PIL.Image.Image image mode=RGB size=640x483 at 0x7FE5D9B01420>
    }
    len(dataset): 22793
    """
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs): # vqa_acc=True, 
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        # Initialize your dataset here
        self.vis_root = vis_root
        vqa_introspect_annotation = json.load(open(ann_paths[0]))
        
        vqav2_json_data = json.load(open(ann_paths[1]))
        vqav2_answers = dict()
        
        for ann in vqav2_json_data["annotations"]:
            vqav2_answers[str(ann["question_id"])] = [x["answer"] for x in ann["answers"]]
            
        print('len of vqav2_answers : ', len(vqav2_answers))
        
        self.annotation = []
        for k, v in vqa_introspect_annotation.items(): # add question_id(str) and sub_qas(list of list: [N_i,2]) to each sample
            gt_sub_qas = []
            for introspect in v["introspect"]:
                sub_qa_list = introspect["sub_qa"]
                if introspect["pred_q_type"] == "invalid":
                    continue
                for sub_qa in sub_qa_list:
                    if sub_qa["sub_answer"] == 'yea':
                        sub_qa["sub_answer"] = 'yes'
                    gt_sub_qas.append(
                        (sub_qa["sub_question"], sub_qa["sub_answer"])
                    )
            gt_sub_qas = list(set(gt_sub_qas))
            v.update({
                "gt_sub_qas": gt_sub_qas,
                "question_id": k,
                "gt_ans": vqav2_answers[k]# v["reasoning_answer_most_common"]# 
            })
            
            self.annotation.append(v)
            # if 0 <= num_data <= len(self.annotation):
            #     break
            
        if num_data != -1:
            self.annotation = self.annotation[:num_data]
            # import random
            # self.annotation = random.sample(self.annotation, num_data)
            
        if len(ann_paths) == 3:
            if os.path.exists(ann_paths[2]):
                self.sub_qas = json.load(open(ann_paths[2], 'r'))
            
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        # self.vqa_acc = vqa_acc
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # import spacy
        # self.lemmatizer = spacy.load("en_core_web_sm")
        
        self._add_instance_ids()
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        question_id = ann["question_id"]

        # ex. /data/coco/images/val2014/COCO_val2014_000000284623.jpg
        image_path = os.path.join(self.vis_root, f'{self.split}2014', f'COCO_{self.split}2014_000000{ann["image_id"]:06}.jpg')
        # print('image_path : ', image_path)
        image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        # text_input = self.text_processor(ann["reasoning_question"])
        text_input = ann["reasoning_question"]
        # reasoning_answer_most_common = self.text_processor(ann["reasoning_answer_most_common"])
        reasoning_answer_most_common = ann["reasoning_answer_most_common"]
        
        
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
            "vision": image,
            "text_input": text_input,
            "question_id": question_id,
            "reasoning_answer_most_common": reasoning_answer_most_common,
            "gt_sub_qas": ann["gt_sub_qas"],
            "gt_ans": ann["gt_ans"], # vqav2 answers list of str(len=10)
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
        }


def main(ann_paths, split):
    dataset = VQAIntrospectDataset(vis_processor=None, text_processor=None, 
                                   vis_root='/data/coco/images/', 
                                   ann_paths=ann_paths, 
                                   num_data=-1, split=split)
    
    for i in range(len(dataset)):
        pprint(dataset[i], width=300)
        break
    print('len(dataset):', len(dataset))

if __name__ == '__main__':
    split = 'val' # 'train', 'val', 'test'
    ann_paths = [
        '/data/VQA_Introspect/VQAIntrospect_valv1.0.json',
        '/data/VQA/v2/v2_mscoco_val2014_annotations.json',
        '/data/VQA_Introspect/sub_qas_val_xl_fewshot_vqaintrospect_unique.json'
    ]
    main(ann_paths, split)
    