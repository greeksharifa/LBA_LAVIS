import json
import os
from PIL import Image

import torch
from torch.utils.data import DataLoader

from dataset.base_dataset import BaseDataset


class VQAIntrospectDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
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
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        import spacy
        self.lemmatizer = spacy.load("en_core_web_sm")
        
        self.split = "val"
        if "train" in ann_paths[0]:
            self.split = "train"
        elif "test" in ann_paths[0]:
            self.split = "test"

        self._add_instance_ids()
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        # ex. /data1/coco/images/val2014/COCO_val2014_000000284623.jpg
        image_path = os.path.join(self.vis_root, f'{self.split}2014', f'COCO_{self.split}2014_000000{ann["image_id"]:06}.jpg')
        # print('image_path : ', image_path)
        image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        # text_input = self.text_processor(ann["reasoning_question"])
        text_input = ann["reasoning_question"]
        # reasoning_answer_most_common = self.text_processor(ann["reasoning_answer_most_common"])
        reasoning_answer_most_common = ann["reasoning_answer_most_common"]
        
        return {
            "image": image,
            "text_input": text_input,
            "question_id": ann["question_id"],
            "reasoning_answer_most_common": reasoning_answer_most_common,
            "gt_sub_qas": ann["gt_sub_qas"],
            "gt_ans": ann["gt_ans"], # vqav2 answers list of str(len=10)
        }

    def collater(self, samples):
        (
            image_list,
            text_input_list,
            question_id_list,
            # instance_id_list,
            reasoning_answer_most_common_list,
            gt_sub_qas_list,
            gt_ans_list,
        ) = ([], [], [], [], [], []) # ([], [], [], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            text_input_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            # instance_id_list.append(sample["instance_id"])
            reasoning_answer_most_common_list.append(sample["reasoning_answer_most_common"])
            gt_sub_qas_list.append(sample["gt_sub_qas"])
            gt_ans_list.append(sample["gt_ans"])

        return {
            "image": image_list,#torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "question_id": question_id_list,
            # "instance_id": instance_id_list,
            "reasoning_answer_most_common": reasoning_answer_most_common_list,
            "gt_sub_qas": gt_sub_qas_list, # list: [bs, N_i, 2]
            "gt_ans": gt_ans_list, # list: [bs, 10]
        }
        
    @staticmethod
    def get_accuracy(outputs, targets):
        """
        args
        - outputs: list of str.         shape: [bsz]
        - targets: list of list of str. shape: [bsz, 10]
        """
        # get vqa_acc
        acc_list = []
        for out, target_list in zip(outputs, targets):
            num_match = sum([out == target for target in target_list])
            vqa_acc = min(1.0, num_match / 3.0)
            acc_list.append(vqa_acc)
        
        return acc_list