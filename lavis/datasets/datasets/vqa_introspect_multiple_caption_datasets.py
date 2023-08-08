"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random
import logging
from collections import OrderedDict

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset
from lavis.common.registry import registry

COCOCapDataset = CaptionDataset


def _init_VQAIntrospectMultipleCapDataset(ann_paths, max_sample_num=99999999):
    annotation = []
    
    sub_question_id = 0
    for ann_path in ann_paths:
        logging.info(f"Loading {ann_path}")
        json_data = json.load(open(ann_path, "r"))
        
        for main_question_id, value in json_data.items():
            image_id = value["image_id"]
            main_question = value["reasoning_question"]
            main_answer = value["reasoning_answer_most_common"]
            
            sub_q_set = set()
            
            sub_q_list = []
            sub_a_list = []
            sub_q_id_list = []
            
            pred_q_type_list = []
            
            for introspect in value["introspect"]:
                sub_qa_list = introspect["sub_qa"]
                pred_q_type = introspect["pred_q_type"]
                
                for sub_qa in sub_qa_list:
                    if sub_qa["sub_question"] in sub_q_set:
                        pass  # 중복
                    else:
                        sub_q_set.add(sub_qa["sub_question"])
                        
                        sub_q_list.append(sub_qa["sub_question"])
                        sub_a_list.append(sub_qa["sub_answer"])
                        pred_q_type_list.append(pred_q_type)
                        
                        sub_q_id_list.append(sub_question_id)
                        sub_question_id += 1
            
            _sample = {
                "image_id": image_id,
                "main_question_id": int(main_question_id),
                "main_question": main_question,
                "main_answer": main_answer,
                "sub_question_id_list": sub_q_id_list,
                "sub_question_list": sub_q_list,
                "sub_answer_list": sub_a_list,
                "pred_q_type_list": pred_q_type_list,
            }
            annotation.append(_sample)
            
            if sub_question_id >= max_sample_num: break
    
    return annotation


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


# _prompt_file_path = os.path.join(registry.get_path("cache_root"), "coco_gt")
_prompt_file_path = "prompts.json"


def _apply_VQAIntrospect_MultipleSubQ_prompt(main_question, sub_question_list, sub_answer_list):
    multiple_prompts = json.load(open(_prompt_file_path, "r"))["multiple"]
    # text_output으로 1개, previous generated sub_qa로 0~2개 비복원추출
    sub_qa_pair_num = random.randint(1, min(3, len(sub_question_list)))
    sub_qa_indices = random.sample(range(len(sub_question_list)), sub_qa_pair_num)
    
    text_input = multiple_prompts["init_prompt"].format(main_question)
    if sub_qa_pair_num > 1:
        text_input += multiple_prompts["after_prompt"]
        for i in range(1, sub_qa_pair_num):
            index = sub_qa_indices[i]
            if i > 1:
                text_input += ', '
            text_input += multiple_prompts["pair_prompt"].format(i, sub_question_list[index], i, sub_answer_list[index])
    
    text_output = sub_question_list[sub_qa_indices[0]]
    
    return text_input, text_output


class VQAIntrospectMultipleCapDataset(CaptionDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_root = vis_root
        
        # TODO: annotation 불러오기
        # TODO: naive: 1개의 main_Q, N개의 sub_q가 있다면 (main_Q, 각 sub_q) pair를 N개 생성
        self.annotation = _init_VQAIntrospectMultipleCapDataset(ann_paths, 100)
        
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            # img_id = ann["main_question_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        self._add_instance_ids()
        
        logging.info(f"VQAIntrospectMultipleCapDataset: {len(self.annotation)}")

        
    def __getitem__(self, index):
        ann = self.annotation[index]

        # train2014/COCO_train2014_000000216531.jpg
        image_path = os.path.join(self.vis_root, f'train2014/COCO_train2014_{ann["image_id"]:012}.jpg')
        # image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        # text_input = self.text_processor(_apply_VQAIntrospect_prompt(ann["main_question"]))
        text_input, text_output = _apply_VQAIntrospect_MultipleSubQ_prompt(
            ann["main_question"],
            ann["sub_question_list"],
            ann["sub_answer_list"],
        )

        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
        }


class VQAIntrospectMultipleCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_root = vis_root
        
        self.annotation = _init_VQAIntrospectMultipleCapDataset(ann_paths, 50)
        
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            # img_id = ann["main_question_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        self._add_instance_ids()
        
        logging.info(f"VQAIntrospectMultipleCapEvalDataset: {len(self.annotation)}")
    
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        # val2014/COCO_val2014_000000265814.jpg
        image_path = os.path.join(self.vis_root, f'val2014/COCO_val2014_{ann["image_id"]:012}.jpg')
        # image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        
        image = self.vis_processor(image)
        # text_input = self.text_processor(_apply_VQAIntrospect_prompt(ann["main_question"]))
        text_input, text_output = _apply_VQAIntrospect_MultipleSubQ_prompt(
            ann["main_question"],
            ann["sub_question_list"],
            ann["sub_answer_list"],
        )
        
        _return = {
            "image": image,
            "image_id": ann["image_id"],
            "main_question_id": ann["main_question_id"],
            "instance_id": ann["instance_id"],
            "text_input": text_input,
            "prompt": text_input,
            # "sub_question_id": ann["sub_question_id"],
            "text_output": text_output,
        }
        # print('_return:', _return)

        return _return

