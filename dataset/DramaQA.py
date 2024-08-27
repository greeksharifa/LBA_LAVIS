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


class DramaQAEvalDataset(VideoEvalDataset):
    # vid_error_list: ["AnotherMissOh14_005_0172", "AnotherMissOh14_009_0297", "AnotherMissOh14_012_0422", "AnotherMissOh14_017_0521", "AnotherMissOh14_017_0522", "AnotherMissOh13_001_0035", "AnotherMissOh13_001_0035", "AnotherMissOh13_005_0172", "AnotherMissOh13_005_0172", "AnotherMissOh13_015_0532", "AnotherMissOh13_019_0647", "AnotherMissOh13_019_0647", "AnotherMissOh13_021_0714", "AnotherMissOh13_021_0714", "AnotherMissOh13_037_1213", "AnotherMissOh13_040_1346", "AnotherMissOh15_001_0061", "AnotherMissOh15_001_0061", "AnotherMissOh15_002_0072", "AnotherMissOh15_002_0072", "AnotherMissOh15_004_0122", "AnotherMissOh15_004_0122", "AnotherMissOh15_004_0146", "AnotherMissOh15_006_0189", "AnotherMissOh15_006_0189", "AnotherMissOh15_006_0196", "AnotherMissOh15_006_0196", "AnotherMissOh15_015_0479", "AnotherMissOh15_024_0683", "AnotherMissOh15_024_0683", "AnotherMissOh15_029_0802", "AnotherMissOh15_029_0804", "AnotherMissOh15_030_0860"]
    
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, num_data=-1, **kwargs):
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.vis_root = vis_root
        self.annotation = []
        self.n_frms = kwargs['n_frms'] # default: 5
        
        ann_path, sub_questions_path = ann_paths
        # ann_path, vis_path = ann_paths
        # self.vis_features = torch.load(vis_path)
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        
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
                
                if self.datasets_cfg.get("only_scene", False) and not sample["vid"].endswith('0000'):
                    continue
                    
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
                        
        self.sub_questions = json.load(open(sub_questions_path, 'r'))
        
        # json.dump(vid_error_list, open('DramaQA_vid_error_list.json', 'w'))
        
        self._add_instance_ids()
        
        self.ANSWER_MAPPING = {0: "(A)", 1: "(B)", 2: "(C)", 3: "(D)", 4: "(E)"}
        
        print("\n" + self.__class__.__name__)
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))
        print('len(sub_questions):', len(self.sub_questions))
        # print('type(self.vis_features), len(self.vis_features):', type(self.vis_features), len(self.vis_features))
        
    def get_image_path(self, vid, supple=False):
        # import pdb; pdb.set_trace()
        if vid.endswith('0000'):
            scene_dir_path = os.path.join(self.vis_root, vid.replace('_', '/'))[:-4] # ex. /data1/AnotherMissOh/AnotherMissOh_images/AnotherMissOh01/001/0078
            dir_paths = glob.glob(os.path.join(scene_dir_path, '*/'))

            image_paths = []
            for dir_path in dir_paths:
                image_paths += glob.glob(dir_path + '*.jpg')
        else:
            dir_path = os.path.join(self.vis_root, vid.replace('_', '/'))
            image_paths = glob.glob(os.path.join(dir_path, '*.jpg'))

        return sorted(image_paths)
     
    # def get_image_path_2(self, vid):
        
           
    def __getitem__(self, index):
        ann = self.annotation[index]
        vid = ann["vid"]
        question_id = ann["qid"]
        
        image_paths = self.get_image_path(vid)
        frms, frms_supple = self.get_frames(image_paths)
            
        question = ann["que"] # question = self.text_processor(ann["que"])
        # gt_ans = self.__class__.ANSWER_MAPPING[ann["correct_idx"]]
        gt_ans = ann["correct_idx"]
        question_type = "Level " + str(ann["q_level_mem"])

        return {
            "vision": frms, # frms, # 이름은 image지만 list of ndarray, 즉 video랑 비슷
            "vision_supple": frms_supple,
            # "video": video, # [min(n_frms, len(video)), 768]
            "text_input": question,
            "question_id": question_id,
            "gt_ans": gt_ans, #ann["correct_idx"],
            "candidate_list": ann["answers"],
            "answer_sentence": ann["answers"][ann["correct_idx"]],
            "type": question_type,
            "vid": vid,
            "sub_question_list": self.sub_questions[str(question_id)],
            # "instance_id": ann["instance_id"],
        }
   