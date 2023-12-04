"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import glob
import json
import os
from collections import OrderedDict

import torch
from PIL import Image

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)

from colors import Colors, print_color


class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )


class VideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def _build_class_labels(self, ans_path):
        ans2label = json.load(open(ans_path))

        self.class_labels = ans2label

    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."

        ann = self.annotation[index]
        
        print_color(msg="in class VideoQADataset", color=Colors.BRIGHT_RED)

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(vpath)
        question = self.text_processor(ann["question"])

        return {
            "video": frms,
            "text_input": question,
            "answers": self._get_answer_label(ann["answer"]),
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }


class DramaQAEvalDataset(VideoQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        # print("self.annotation[index]:", self.annotation[index])
        # print("len(self.annotation):", len(self.annotation))
        # self.annotation[index]: {
        #   'videoType': 'scene',
        #   'answers': [
        #       'Since Dokyung wanted to kick the old man.',
        #       'Just because Dokyung wanted to buy some snacks from the old man.',
        #       "Because Dokyung tried to give Dokyung's umbrella to the old man.",
        #       'As Dokyung wanted to take the clothes away from the old man.',
        #       'It was because Dokyung wanted to run away from Haeyoung1.'
        #   ],
        #   'vid': 'AnotherMissOh17_001_0000',
        #   'qid': 13041,
        #   'shot_contained': [25, 49],
        #   'q_level_mem': 3,
        #   'que': 'Why did Dokyung go to the old man?',
        #   'q_level_logic': 4,
        #   'instance_id': '0'
        # }
        # len(self.annotation): 4019
        ann = self.annotation[index]
        
        # print_color(msg="in class DramaQAEvalDataset", color=Colors.BRIGHT_RED)
        
        # image_path = os.path.join(self.vis_root, ann["image"])
        # image = Image.open(image_path).convert("RGB")
        
        image_path_list = []
        
        # TODO: 경로 수정
        vname = ann["vid"] # AnotherMissOh17_001_0000
        
        
        if vname.endswith("0000"): # scene
            vpath = os.path.join(self.vis_root, vname[:-5].replace('_', '/'))
            for shot in glob.glob(vpath + "/*"):
                for frm in glob.glob(shot + "/*"):
                    image_path_list.append(frm)
                    break
        else: # shot
            vpath = os.path.join(self.vis_root, vname.replace('_', '/'))
            for frm in glob.glob(vpath + "/*"):
                image_path_list.append(frm)
        
        # print_color(msg="image_list: {}, {}".format(len(image_path_list), image_path_list[0]), color=Colors.BRIGHT_RED)
        # shot_contained = ann["shot_contained"]
        # print_color(msg="vname: {}".format(vname), color=Colors.BRIGHT_RED)
        # print_color(msg="vpath: {}".format(vpath), color=Colors.BRIGHT_RED)
        
        frms = None
        for image_path in image_path_list:
            image = Image.open(image_path).convert("RGB")
            # print_color(msg="image_path: {}".format(image_path), color=Colors.BRIGHT_RED)
            image = self.vis_processor(image)
            # print_color(msg="image: {}".format(image.shape), color=Colors.BRIGHT_RED)
            
            # 1
            frms = image#.unsqueeze(0)
            break
            # 2
            # if frms is None:
            #     frms = image.unsqueeze(0).unsqueeze(2)
            # else:
            #     frms = torch.cat((frms, image.unsqueeze(0).unsqueeze(2)), dim=2)
                
        # frms = self.vis_processor(vpath)
        # print_color(msg="frms    : {}, {}".format(type(frms), frms.shape), color=Colors.BRIGHT_RED)
        # question = self.text_processor(ann["que"])
        # print_color(msg="question: {}".format(question), color=Colors.BRIGHT_RED)

        # assert False, "DramaQAEvalDataset __getitem__() is not implemented yet."
        
        # print('self.class_labels;', self.class_labels)
        # print('len(self.class_labels);', len(self.class_labels))
        # print('type(self.class_labels);', type(self.class_labels))
        '''
        self.class_labels; {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        len(self.class_labels); 5
        type(self.class_labels); <class 'dict'>
        '''
        
        question = ann["que"]
        # text_input = ""
        # text_input += "Choose a number of answer from the following question.\n"
        # for i, cand in enumerate(ann["answers"]):
        #     text_input += "{}. {}\n".format(i, cand)
        # text_input += "Question: {}\n".format(question)
        # text_input += "Answer:"
        
        return {
            "image": frms,
            "text_input": question, #text_input,
            # "answers": self._get_answer_label(ann["correct_idx"]), # answer list가 아니라 answer 하나만 들어가야 함
            "answer": ann["correct_idx"],
            "answer_list": ann["answers"],
            "gt_ans": ann["answers"][ann["correct_idx"]],
            "question_id": ann["qid"],
            "instance_id": ann["instance_id"],
        }
        
        