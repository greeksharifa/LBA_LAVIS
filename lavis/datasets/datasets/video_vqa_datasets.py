"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from collections import OrderedDict

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
        
        print_color(msg="DramaQAEvalDataset __getitem__() is not implemented yet.", color=Colors.BRIGHT_RED)
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
        
        print_color(msg="in class DramaQAEvalDataset", color=Colors.BRIGHT_RED)
        
        vname = ann["vid"]
        # TODO: 경로 수정
        vpath = os.path.join(self.vis_root, vname)
        print_color(msg="vname: {}".format(vname), color=Colors.BRIGHT_RED)
        print_color(msg="vpath: {}".format(vpath), color=Colors.BRIGHT_RED)
        
        frms = self.vis_processor(vpath)
        question = self.text_processor(ann["que"])
        # print_color(msg="frms    : {}".format(frms), color=Colors.BRIGHT_RED)
        # print_color(msg="question: {}".format(question), color=Colors.BRIGHT_RED)

        assert False, "DramaQAEvalDataset __getitem__() is not implemented yet."

        return {
            "video": frms,
            "text_input": question,
            "answers": self._get_answer_label(ann["answer"]),
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
        
        