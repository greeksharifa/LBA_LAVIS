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
from PIL import Image

import numpy as np
import torch
from torchvision import transforms

# from multimodal_classification_datasets import MultimodalClassificationDataset
# from utils.load_video import load_video_to_sampled_frames
from dataset.video import read_video_pyav

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


class DramaQAEvalDataset(BaseDataset):
    # vid_error_list: ["AnotherMissOh14_005_0172", "AnotherMissOh14_009_0297", "AnotherMissOh14_012_0422", "AnotherMissOh14_017_0521", "AnotherMissOh14_017_0522", "AnotherMissOh13_001_0035", "AnotherMissOh13_001_0035", "AnotherMissOh13_005_0172", "AnotherMissOh13_005_0172", "AnotherMissOh13_015_0532", "AnotherMissOh13_019_0647", "AnotherMissOh13_019_0647", "AnotherMissOh13_021_0714", "AnotherMissOh13_021_0714", "AnotherMissOh13_037_1213", "AnotherMissOh13_040_1346", "AnotherMissOh15_001_0061", "AnotherMissOh15_001_0061", "AnotherMissOh15_002_0072", "AnotherMissOh15_002_0072", "AnotherMissOh15_004_0122", "AnotherMissOh15_004_0122", "AnotherMissOh15_004_0146", "AnotherMissOh15_006_0189", "AnotherMissOh15_006_0189", "AnotherMissOh15_006_0196", "AnotherMissOh15_006_0196", "AnotherMissOh15_015_0479", "AnotherMissOh15_024_0683", "AnotherMissOh15_024_0683", "AnotherMissOh15_029_0802", "AnotherMissOh15_029_0804", "AnotherMissOh15_030_0860"]
    ANSWER_MAPPING = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}
    
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.vis_root = vis_root
        self.annotation = []
        self.n_frms = kwargs['n_frms'] # default: 5
        
        ann_path, vis_path = ann_paths
        
        self.vis_features = torch.load(vis_path)
        
        # vid_error_list = []
        
        with open(ann_path, "r") as f:
            loaded = json.load(f)
            
            if num_data == -1: # use all dataset
                len_loaded = len(loaded)
            else:
                len_loaded = min(len(loaded), num_data)
                
            for i, sample in enumerate(loaded):
                if len(self.annotation) >= len_loaded: # 0 <= num_data <= i:
                    break
                vid = sample["vid"]
                print(f'\r{i:6d}/{len_loaded:6d} : {vid}', end='')
                
                self.annotation.append(sample)
                '''
                try:
                    frms = load_video_to_sampled_frames(os.path.join(vis_root, f'{video_id}.mp4'), n_frms=self.n_frms)
                    transform = transforms.ToTensor()
                    tensors = [transform(img) for img in frms]
                    stacked_tensor = torch.stack(tensors)
                    self.annotation.append(sample)
                except Exception as e:
                    print('\nvideo processing error:', video_id)
                    vid_error_list.append(video_id)
                '''
                        
        # json.dump(vid_error_list, open('DramaQA_vid_error_list.json', 'w'))
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # self.features_dim = 

        self._add_instance_ids()
        
        print("DramaQAEvalDataset")
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))
        print('type(self.vis_features), len(self.vis_features):', type(self.vis_features), len(self.vis_features))
        
        
    def collater(self, samples):
        (
            image_list,
            # video_list,
            text_input_list,
            question_id_list,
            gt_ans_list,
            candidate_list_list,
            answer_sentence_list,
        ) = ([], [], [], [], [], [])
        
        for sample in samples:
            image_list.append(sample["image"])
            # video_list.append(sample["video"])
            text_input_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            gt_ans_list.append(sample["gt_ans"])
            candidate_list_list.append(sample["candidate_list"])
            answer_sentence_list.append(sample["answer_sentence"])
            
        return {
            "image": image_list, #torch.stack(image_list, dim=0),
            # "video": video_list,
            "text_input": text_input_list,
            "question_id": question_id_list,
            "gt_ans": gt_ans_list, 
            "candidate_list": candidate_list_list,
            "answer_sentence": answer_sentence_list,
        }
       
    def get_image_path(self, vid):

        if vid.endswith('0000'):
            scene_dir_path = os.path.join(self.vis_root, vid.replace('_', '/'))[:-4] # ex. /data1/AnotherMissOh/AnotherMissOh_images/AnotherMissOh01/001/0078
            dir_paths = sorted(glob.glob(os.path.join(scene_dir_path, '*/')))

            if self.n_frms < len(dir_paths):
                idxs = np.linspace(-1, len(dir_paths), self.n_frms+2, dtype=int)
                idxs = idxs[1:-1]
                dir_paths = [dir_paths[idx] for idx in idxs]

            # shot_contained = sample["shot_contained"]
            image_paths = []
            for dir_path in dir_paths:
                images = sorted(glob.glob(dir_path + '*.jpg'))
                image_paths.append(images[len(images) // 2]) # shot 중 가운데 frame만 선택
        else:
            dir_path = os.path.join(self.vis_root, vid.replace('_', '/'))
            image_paths = sorted(glob.glob(os.path.join(dir_path, '*.jpg')))
            idxs = np.linspace(-1, len(image_paths), self.n_frms+2, dtype=int)
            idxs = idxs[1:-1]
            image_paths = [image_paths[idx] for idx in idxs]
            
        # print('image_paths:', image_paths)

        return image_paths
        
    def __getitem__(self, index):
        ann = self.annotation[index]

        vid = ann["vid"]
        vpath = os.path.join(self.vis_root, f'{vid}.mp4')
        
        clip = read_video_pyav(vpath, self.n_frms)
        
        # # load images. output: list of PIL.Image
        # frms = []
        # image_paths = self.get_image_path(vid)
        # for img_path in image_paths:
        #     frms.append(Image.open(img_path))
        # if len(frms) < self.n_frms:
        #     frms = [Image.new('RGB', frms[0].size)] * (self.n_frms - len(frms)) + frms
        
        
        """
        # directly read Video    
        try:
            frms = load_video_to_sampled_frames(vpath, n_frms=self.n_frms) # list of PIL.Image
            transform = transforms.ToTensor()
            tensors = [transform(img) for img in frms]
            stacked_tensor = torch.stack(tensors)
            # frms = self.vis_processor(vpath)
        except Exception as e:
            print('*' * 200 + f"\nError processing {vpath}\n" + '*' * 200)
            assert False, e
        """
        
        '''
        # get_video
        if video_id[-4:] == '0000':
            shots = ann['shot_contained']
            start, end = shots[0], shots[1]

            for i in range(start, end+1):
                v_name = video_id[:-4] + f'{i:04}'

                if v_name not in self.vis_features.keys(): 
                    print(v_name, " Not in features")
                    nxt_vid = torch.zeros(1, self.features_dim)
                else: nxt_vid = self.vis_features[v_name].float()

                if i == start: video = nxt_vid
                else: video = torch.concat((video, nxt_vid), dim = 0)
        # Shot
        else:
            scene = False
            if video_id not in self.vis_features.keys():
                print(video_id, "Not in freatures")
                video = torch.zeros(1, self.features_dim)
            else:
                video = self.vis_features[video_id].float()

        
        if len(video) > self.n_frms:
            sampled = []
            for j in range(self.n_frms):
                sampled.append(video[(j * len(video)) // self.n_frms])
            video = torch.stack(sampled)
            video_len = self.n_frms
        elif len(video) < self.n_frms:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.n_frms - video_len, self.features_dim)], 0)
        else:
            video_len = self.n_frms
        '''
            
        question = ann["que"] # question = self.text_processor(ann["que"])
        
        gt_ans = self.__class__.ANSWER_MAPPING[ann["correct_idx"]]

        return {
            "image": clip, # frms, # 이름은 image지만 list of PIL.Image, 즉 video랑 비슷
            # "video": video, # [min(n_frms, len(video)), 768]
            "text_input": question,
            "question_id": ann["qid"],
            "gt_ans": gt_ans, #ann["correct_idx"],
            "candidate_list": ann["answers"],
            "answer_sentence": ann["answers"][ann["correct_idx"]],
            # "instance_id": ann["instance_id"],
        }
     