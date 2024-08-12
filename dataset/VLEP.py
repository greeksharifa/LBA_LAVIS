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

from dataset.VideoQA import VideoEvalDataset


class VLEPEvalDataset(VideoEvalDataset):
    """
    <class 'list'>
    total 4392
    {'a0': 'Ross will stop, turn and point at Monica.',
    'a1': 'Ross will stop and ask Monica why she is pointing at him.',
    'answer': 0,
    'end': 40.37,
    'num_option': 2,
    'qid': 'VLEP_20142',
    'start': 38.81,
    'video': 'friends_s03e09_seg02_clip_07_ep'}
    """

    def get_image_path(self, vid, random=False):
        dir_path = os.path.join(self.vis_root, vid)
        image_paths = glob.glob(os.path.join(dir_path, '*.jpg'))
        
        if random:
            idxs = sorted(np.random.choice(len(image_paths), self.n_frms, replace=len(image_paths) <= self.n_frms))
        else:
            idxs = np.linspace(-1, len(image_paths), self.n_frms+2, dtype=int)[1:-1]
            
        image_paths = [image_paths[idx] for idx in idxs]
        # print('image_paths:', image_paths)

        return image_paths
        
    def __getitem__(self, index):
        ann = self.annotation[index]

        vid = ann["video"]
        
        # load images. output: list of PIL.Image
        def _get_frames(random=False):
            _frms = []
            image_paths = self.get_image_path(vid, random=random)
            for img_path in image_paths:
                # _frms.append(Image.open(img_path))
                _frms.append(np.array(Image.open(img_path)))
            if len(_frms) < self.n_frms:
                # _frms = [Image.new('RGB', _frms[0].size)] * (self.n_frms - len(_frms)) + _frms
                _frms += [np.zeros_like(_frms[0])] * (self.n_frms - len(_frms))
            return _frms
                
        frms = _get_frames(False)
        
        frms_supple = []
        for i in range(self.supple_n):
            frms_supple.append(_get_frames(True))
        # print(len(frms_supple), len(frms_supple[0]), frms_supple[0][0].shape)
        
        question = "Which event is more likely to happen right after?" # event prediction
        
        # gt_ans = self.__class__.ANSWER_MAPPING[ann["correct_idx"]]
        gt_ans = ann["answer"]
        
        candidate_list = []
        for i in range(ann["num_option"]):
            candidate_list.append(ann[f'a{i}'])
            
        return {
            "vision": frms, # frms, # 이름은 image지만 list of ndarray, 즉 video랑 비슷
            "vision_supple": frms_supple, # list of list of ndarray
            "text_input": question,
            "question_id": ann["qid"],
            "gt_ans": gt_ans,
            "candidate_list": candidate_list,
            "answer_sentence": candidate_list[gt_ans],
            # "type": question_type,
            "vid": vid,
            # "instance_id": ann["instance_id"],
        }
   