"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json

from PIL import Image

import torch

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class VQAIntrospectDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        print('in datasets.datasets.vqa_introspect_datasets.py VQAIntrosectDataset class')
        print('vis_processor', vis_processor)
        print('text_processor', text_processor)
        print('vis_root', vis_root)
        print('ann_paths', ann_paths)
        
        self.vis_root = vis_root
        
        # TODO: annotation 불러오기
        # TODO: naive: 1개의 main_Q, N개의 sub_q가 있다면 (main_Q, 각 sub_q) pair를 N개 생성
        self.annotation = []
        _cnt = 0
        for ann_path in ann_paths:
            json_data = json.load(open(ann_path, "r"))
            
            for question_id, value in json_data.items():
                _cnt += 1
                if _cnt > 50: break
                image_id = value["image_id"]
                main_question = value["reasoning_question"]
                main_answer = value["reasoning_answer_most_common"]
                
                for introspect in value["introspect"]:
                    sub_qa_list = introspect["sub_qa"]
                    pred_q_type = introspect["pred_q_type"]
                    
                    for sub_qa in sub_qa_list:
                        _sample = {
                            "image_id": image_id,
                            "question_id": question_id,
                            "main_question": main_question,
                            "main_answer": main_answer,
                            "sub_question": sub_qa["sub_question"],
                            "sub_answer": sub_qa["sub_answer"],
                            "pred_q_type": pred_q_type,
                        }
                        self.annotation.append(_sample)
                        
        from pprint import pprint
        print('self.annotation[0]:')
        pprint(self.annotation[0])

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]
        
        # train2014/COCO_train2014_000000265814.jpg
        image_path = os.path.join(self.vis_root, f'train2014/COCO_train2014_{ann["image_id"]:012}.jpg')
        # image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["main_question"])

        # answers = [ann["answer"]]
        # weights = [1]

        return {
            "image": image,
            "text_input": question,
            "text_output": ann["sub_question"],
            # "weights": weights,
        }
    
    def collater(self, samples):
        image_list, question_list, answer_list = [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])


            answers = sample["text_output"]

            answer_list.extend(answers)
            num_answers.append(len(answers))

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": question_list,
            "text_output": answer_list,
            # "weight": torch.Tensor(weight_list),
            # "n_answers": torch.LongTensor(num_answers),
        }
    

class VQAIntrospectEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. gqa/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        
        # TODO: annotation 불러오기
        # TODO: naive: 1개의 main_Q, N개의 sub_q가 있다면 (main_Q, 각 sub_q) pair를 N개 생성
        self.annotation = []
        _cnt = 0
        for ann_path in ann_paths:
            json_data = json.load(open(ann_path, "r"))
            
            for question_id, value in json_data.items():
                _cnt += 1
                if _cnt > 50: break
                image_id = value["image_id"]
                main_question = value["reasoning_question"]
                main_answer = value["reasoning_answer_most_common"]
                
                for introspect in value["introspect"]:
                    sub_qa_list = introspect["sub_qa"]
                    pred_q_type = introspect["pred_q_type"]
                    
                    for sub_qa in sub_qa_list:
                        _sample = {
                            "image_id": image_id,
                            "question_id": question_id,
                            "main_question": main_question,
                            "main_answer": main_answer,
                            "sub_question": sub_qa["sub_question"],
                            "sub_answer": sub_qa["sub_answer"],
                            "pred_q_type": pred_q_type,
                        }
                        self.annotation.append(_sample)
    
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        # val2014/COCO_val2014_000000265814.jpg
        image_path = os.path.join(self.vis_root, f'val2014/COCO_val2014_{ann["image_id"]:012}.jpg')
        # image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["main_question"])


        return {
            "image": image,
            "text_input": question,
            # "answer": answer,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "sub_answer": ann["sub_answer"],
        }
