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
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        # text_input = self.text_processor(ann["question"])
        text_input = ann["question"]

        return {
            "vision": image,
            "text_input": text_input,
            "question_id": ann["question_id"],
            "gt_ans": ann["answer"], # vqav2 answers list of str(len=10)
        }