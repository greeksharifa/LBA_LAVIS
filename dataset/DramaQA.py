"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from collections import OrderedDict

import torch
from torchvision import transforms

# from multimodal_classification_datasets import MultimodalClassificationDataset
from utils.load_video import load_video_to_sampled_frames

from dataset.base_dataset import BaseDataset

'''
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
'''

# class DramaQAEvalDataset(VideoQADataset):
class DramaQAEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.vis_root = vis_root
        self.annotation = []
        
        vid_error_list = []
        
        for ann_path in ann_paths:
            with open(ann_path, "r") as f:
                loaded = json.load(f)
                len_loaded = num_data #len(loaded)
                for i, sample in enumerate(loaded):
                    if 0 <= num_data <= i:
                        break
                    vid = sample["vid"]
                    print(f'\r{i:6d}/{len_loaded:6d} : {vid}', end='')
                    if vid in ["AnotherMissOh14_017_0000", "AnotherMissOh14_017_0520", "AnotherMissOh14_017_0521", "AnotherMissOh14_017_0522"]:
                        continue
                    if os.path.isfile(os.path.join(vis_root, f'{vid}.mp4')):
                        try:
                            frms = load_video_to_sampled_frames(os.path.join(vis_root, f'{vid}.mp4'), n_frms=1)
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
        
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._add_instance_ids()
        
        print("DramaQAEvalDataset")
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))
        
        
    def collater(self, samples):
        (
            image_list,
            text_input_list,
            question_id_list,
            gt_ans_list,
            candidate_list_list,
            answer_sentence_list,
        ) = ([], [], [], [], [], [])
        
        for sample in samples:
            image_list.append(sample["image"])
            text_input_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            gt_ans_list.append(sample["gt_ans"])
            candidate_list_list.append(sample["candidate_list"])
            answer_sentence_list.append(sample["answer_sentence"])
            
        return {
            "image": image_list, #torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "question_id": question_id_list,
            "gt_ans": gt_ans_list, 
            "candidate_list": candidate_list_list,
            "answer_sentence": answer_sentence_list,
        }
        
    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["vid"]
        vpath = os.path.join(self.vis_root, f'{vname}.mp4')

        try:
            frms = load_video_to_sampled_frames(vpath, n_frms=self.n_frms) # list of PIL.Image
            transform = transforms.ToTensor()
            tensors = [transform(img) for img in frms]
            stacked_tensor = torch.stack(tensors)
            # frms = self.vis_processor(vpath)
        except Exception as e:
            print('*' * 200 + f"\nError processing {vpath}\n" + '*' * 200)
            assert False, e
        question = ann["que"] # question = self.text_processor(ann["que"])

        return {
            "image": frms, # frms, # 이름은 image지만 list of PIL.Image, 즉 video랑 비슷
            "text_input": question,
            "question_id": ann["qid"],
            "gt_ans": ann["correct_idx"],
            "candidate_list": ann["answers"],
            "answer_sentence": ann["answers"][ann["correct_idx"]],
            # "instance_id": ann["instance_id"],
        }
    
    # def get_accuracy(self, outputs, targets, match1ok=False):