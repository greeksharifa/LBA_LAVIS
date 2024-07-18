import json
import os
from PIL import Image

import torch
from torch.utils.data import DataLoader

from dataset.base_dataset import BaseDataset


class AOKVQADataset(BaseDataset):
    """
    {
        "split": "val", 
        "image_id": 461751, 
        "question_id": "22jbM6gDxdaMaunuzgrsBB", 
        "question": "What is in the motorcyclist's mouth?", 
        "choices": ["toothpick", "food", "popsicle stick", "cigarette"], 
        "correct_choice_idx": 3, 
        "direct_answers": ["cigarette", "cigarette", "cigarette", "cigarette", "cigarette", "cigarette", "cigarette", "cigarette", "cigarette", "cigarette"], "difficult_direct_answer": false, "rationales": ["He's smoking while riding.", "The motorcyclist has a lit cigarette in his mouth while he rides on the street.", "The man is smoking."], 
        "image": "val2014/COCO_val2014_000000461751.jpg", 
        "dataset": "aokvqa"
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
            "image": image,
            "text_input": text_input,
            "question_id": ann["question_id"],
            "gt_ans": ann["direct_answers"], # vqav2 answers list of str(len=10)
        }