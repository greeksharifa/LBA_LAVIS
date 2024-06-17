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

class VideoQAInstructDataset(VideoQADataset):
    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(vpath)
        question = self.text_processor(ann["question"])

        return {
            "video": frms,
            "text_input": question,
            "answer": ann["answer"],
            "text_output": ann["answer"],
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            ## add weight to use with vqa eval script
            "weight": [1.]
        }


class DramaQAEvalDataset(VideoQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.vis_root = vis_root
        self.annotation = []
        
        vid_error_list = []
        
        for ann_path in ann_paths:
            with open(ann_path, "r") as f:
                loaded = json.load(f)
                len_loaded = len(loaded)
                for i, sample in enumerate(loaded):
                    vid = sample["vid"]
                    print(f'\r{i:6d}/{len_loaded:6d} : {vid}', end='')
                    if vid in ["AnotherMissOh14_017_0000", "AnotherMissOh14_017_0520", "AnotherMissOh14_017_0521", "AnotherMissOh14_017_0522"]:
                        continue
                    if os.path.isfile(os.path.join(vis_root, f'{vid}.mp4')):
                        try:
                            frms = vis_processor(os.path.join(vis_root, f'{vid}.mp4'))
                            # import pdb
                            # pdb.set_trace()
                            self.annotation.append(sample)
                        except Exception as e:
                            print('\nvideo processing error:', vid)
                            vid_error_list.append(vid)
                            # assert False, "No!"
                        
        print('\n\nvid_error_list:', vid_error_list)
        # json.dump(vid_error_list, open('vid_error_list.json', 'w'))
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()
        
        print("DramaQAEvalDataset")
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))
        
        
    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["vid"]
        vpath = os.path.join(self.vis_root, f'{vname}.mp4')

        try:
            frms = self.vis_processor(vpath)
        except Exception as e:
            print('*' * 200 + f"\nError processing {vpath}\n" + '*' * 200)
            assert False, e
        question = self.text_processor(ann["que"])

        return {
            "video": frms,
            "text_input": question,
            "answer": ann["answers"][ann["correct_idx"]],
            "question_id": ann["qid"],
            "instance_id": ann["instance_id"],
        }