from collections import OrderedDict
import json
import os
import torch
import random

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


class VQAIntrospectEvalDataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. gqa/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        annotation = json.load(open(ann_paths[0]))
        
        self.annotation = []
        for k, v in annotation.items(): # add question_id(str) and sub_qas(list of list: [N_i,2]) to each sample
            v.update({"question_id": k})
            sub_qas = []
            for introspect in v["introspect"]:
                sub_qa_list = introspect["sub_qa"]
                # pred_q_type = introspect["pred_q_type"]
                for sub_qa in sub_qa_list:
                    if sub_qa["sub_answer"] == 'yea':
                        sub_qa["sub_answer"] = 'yes'
                    sub_qas.append(
                        (sub_qa["sub_question"], sub_qa["sub_answer"])
                    )
            sub_qas = list(set(sub_qas))
            v.update({"sub_qas": sub_qas})
            
            self.annotation.append(v)
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        self.split = "val"
        if "train" in ann_paths[0]:
            self.split = "train"
        elif "test" in ann_paths[0]:
            self.split = "test"

        self._add_instance_ids()
        
    def collater(self, samples):
        (
            image_list,
            text_input_list,
            question_id_list,
            # instance_id_list,
            reasoning_answer_most_common_list,
            sub_qas_list,
        ) = ([], [], [], [], []) # ([], [], [], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            text_input_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            # instance_id_list.append(sample["instance_id"])
            reasoning_answer_most_common_list.append(sample["reasoning_answer_most_common"])
            sub_qas_list.append(sample["sub_qas"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "question_id": question_id_list,
            # "instance_id": instance_id_list,
            "reasoning_answer_most_common": reasoning_answer_most_common_list,
            "sub_qas": sub_qas_list, # list: [bs, N_i, 2]
        }

    def __getitem__(self, index):
        """
        ann example:
        {
            "question_id": "284885001",
            "image_id": 284885,
            "reasoning_answer_most_common": "no",
            "reasoning_question": "is the house new?",
            "introspect": [
                {
                    "sub_qa": [
                        {
                            "sub_question": "is the window of the house boarded up?",
                            "sub_answer": "yes"
                        },
                        {...}
                    ],
                    "pred_q_type": "reasoning"
                }, 
                {...}
            ]
        }
        """
        ann = self.annotation[index]
        # self.vis_root : /data1/coco/images. 
        # lavis/configs/default.yaml의 cache_root + lavis/configs/datasets/vqa_introspect/defaults.yaml의 images.storage

        # /data1/coco/images/val2014/COCO_val2014_000000284623.jpg
        image_path = os.path.join(self.vis_root, f'{self.split}2014', f'COCO_{self.split}2014_000000{ann["image_id"]:06}.jpg')
        # print('image_path : ', image_path)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        text_input = self.text_processor(ann["reasoning_question"])
        reasoning_answer_most_common = self.text_processor(ann["reasoning_answer_most_common"])
        sub_qas = ann["sub_qas"]

        return {
            "image": image,
            "text_input": text_input,
            "question_id": ann["question_id"],
            "reasoning_answer_most_common": reasoning_answer_most_common,
            "sub_qas": sub_qas,
        }