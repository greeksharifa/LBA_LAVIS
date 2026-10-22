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

from colors import Colors, print_sample

from collections import OrderedDict
import json
import os
import torch

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


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


# class dummy_VQAIntrospectVQATestDataset(VQADataset, __DisplMixin):
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
#         self.vis_root = vis_root
#
#         self.annotation = []
#         vqav2_questions = json.load(open(ann_paths[0], "r"))["questions"]
#         vqav2_answers = json.load(open(ann_paths[1], "r"))["annotations"]
#         vqa_introspects = json.load(open(ann_paths[2], "r"))
#
#         question_ids = list(vqa_introspects.keys())
#
#         """
#         >>> pprint(data['questions'][0])
#         {'image_id': 458752,
#          'question': 'What is this photo taken looking through?',
#          'question_id': 458752000}
#         >>> pprint(ann['annotations'][0])
#         {'answer_type': 'other',
#          'answers': [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1},
#                      {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 2},
#                      {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 3},
#                      {'answer': 'netting', 'answer_confidence': 'yes', 'answer_id': 4},
#                      {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 5},
#                      {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 6},
#                      {'answer': 'mesh', 'answer_confidence': 'maybe', 'answer_id': 7},
#                      {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 8},
#                      {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 9},
#                      {'answer': 'net', 'answer_confidence': 'yes', 'answer_id': 10}],
#          'image_id': 458752,
#          'multiple_choice_answer': 'net',
#          'question_id': 458752000,
#          'question_type': 'what is this'}
# """
#         vqav2_QAs = dict()
#         for question_sample, answer_sample in zip(vqav2_questions, vqav2_answers):
#             question_id = question_sample["question_id"]
#             _sample = {
#                 "image_id": question_sample["image_id"],
#                 "question_id": question_id,
#                 "question": question_sample["question"],
#                 "answer": answer_sample["answers"],
#
#             }
#             vqav2_QAs[question_id] = _sample
#
#         for answer_sample in vqav2_answers:
#             question_id = answer_sample["question_id"]
#             if question_id in vqav2_QAs.keys():
#                 vqav2_QAs[question_id]["answer"] = answer_sample["answers"]
#
#         for
#
#
#         for ann_path in ann_paths:
#             self.annotation.extend(json.load(open(ann_path, "r")))
#
#         _sample = {
#             "image_id": image_id,
#             "main_question_id": int(main_question_id),
#             "main_question": main_question,
#         }
#
#         self.vis_processor = vis_processor
#         self.text_processor = text_processor
#
#         self._add_instance_ids()
#
#     def __getitem__(self, index):
#         ann = self.annotation[index]
#
#         image_path = os.path.join(self.vis_root, ann["image"])
#         image = Image.open(image_path).convert("RGB")
#
#         image = self.vis_processor(image)
#         question = self.text_processor(ann["question"])
#
#         answer_key = "direct_answers"
#
#         answer_weight = {}
#         for answer in ann[answer_key]:
#             if answer in answer_weight.keys():
#                 answer_weight[answer] += 1 / len(ann[answer_key])
#             else:
#                 answer_weight[answer] = 1 / len(ann[answer_key])
#
#         answers = list(answer_weight.keys())
#         weights = list(answer_weight.values())
#
#         return {
#             "image": image,
#             "text_input": question,
#             "answers": answers,
#             "weights": weights,
#         }


class VQAIntrospectTestDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_root = vis_root
        
        self.annotation = []
        json_data = json.load(open(ann_paths[0], "r"))
        for question_id, value in json_data.items():
            _sample = {
                "image_id": value["image_id"],
                "question_id": question_id,
                "question": value["reasoning_question"],
                "answer": value["reasoning_answer_most_common"],
            }
            self.annotation.append(_sample)
        
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
        
        # logging.info(f"{Colors.BRIGHT_MAGENTA}VQAIntrospectQARVQAEvalDataset len: {len(self.annotation)}{Colors.RESET}")
        print(f"{Colors.BRIGHT_MAGENTA}VQAIntrospectTestDataset len: {len(self.annotation)}{Colors.RESET}")



class VQAIntrospectTestEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_root = vis_root

        self.annotation = []
        json_data = json.load(open(ann_paths[0], "r"))
        for i, (main_question_id, value) in enumerate(json_data.items()):
            # if i >= 2000:    break
            if i % 50 != 0:
                continue
            _sample = {
                "image_id": value["image_id"],
                "question_id": main_question_id,
                "question": value["reasoning_question"],
                "answer": value["reasoning_answer_most_common"],
            }
            self.annotation.append(_sample)

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

        # logging.info(f"{Colors.BRIGHT_MAGENTA}VQAIntrospectQARVQAEvalDataset len: {len(self.annotation)}{Colors.RESET}")
        print(f"{Colors.BRIGHT_MAGENTA}VQAIntrospectTestEvalDataset len: {len(self.annotation)}{Colors.RESET}")
    
    
    def collater(self, samples):
        (
            image_list,
            image_id_list,
            instance_id_list,
            question_list,
            question_id_list,
            answer_list,
        ) = ([], [], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            image_id_list.append(sample["image_id"])
            instance_id_list.append(sample["instance_id"])
            question_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            answer_list.append(sample["answer"])

        return {
            "image": torch.stack(image_list, dim=0),
            "image_id": image_id_list,
            "instance_id": instance_id_list,
            "text_input": question_list,
            "question_id": question_id_list,
            "answer": answer_list,
        }
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        # val2014/COCO_val2014_000000265814.jpg
        image_path = os.path.join(self.vis_root, f'val2014/COCO_val2014_{ann["image_id"]:012}.jpg')
        # image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        
        question = self.text_processor(ann["question"])
        
        _return = {
            "image": image,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
            "text_input": question,
            "question_id": ann["question_id"],
            "answer": ann["answer"],
        }
        # logging.info(f"_return: {_return}")

        return _return

