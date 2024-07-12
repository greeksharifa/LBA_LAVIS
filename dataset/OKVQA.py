import json
import os
from PIL import Image

import torch
from torch.utils.data import DataLoader

from dataset.base_dataset import BaseDataset


class OKVQADataset(BaseDataset):
    """
    {
        "question_id": 2971475, 
        "question": "What sport can you use this for?", 
        "answer": ["race", "race", "race", "race", "race", "race", "motocross", "motocross", "ride", "ride"], 
        "image": "val2014/COCO_val2014_000000297147.jpg", 
        "dataset": "okvqa"
    },
    """
    def collater(self, samples):
        (
            image_list,
            text_input_list,
            question_id_list,
            # instance_id_list,
            gt_ans_list,
        ) = ([], [], [], [])
        
        for sample in samples:
            image_list.append(sample["image"])
            text_input_list.append(sample["text_input"])
            question_id_list.append(sample["question_id"])
            gt_ans_list.append(sample["gt_ans"])
            
        return {
            "image": image_list, #torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "question_id": question_id_list,
            "gt_ans": gt_ans_list, # list: [bs, 10]
        }
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        # text_input = self.text_processor(ann["question"])
        text_input = ann["question"]

        return {
            "image": image,
            "text_input": text_input,
            "question_id": ann["question_id"],
            "gt_ans": ann["answer"], # vqav2 answers list of str(len=10)
        }