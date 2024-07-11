import json
from typing import List
import pandas as pd
import torch

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.annotation = []
        for ann_path in ann_paths:
            if any(ext in ann_path for ext in ['csv', 'tsv']):
                df = pd.read_csv(ann_path)
                self.annotation.extend(df.to_dict(orient="records"))
                
            elif 'jsonl' in ann_path:
                with open(ann_path, "r") as f:
                    self.annotation.extend([json.loads(line) for line in f])

            else:
                with open(ann_path, "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        self.annotation.extend(loaded)
                    elif isinstance(loaded, dict):
                       self.annotation.extend([{"sample_id": k, **v} if isinstance(v, dict) else {"sample_id": k, "data": v} for k, v in loaded.items()])


        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        raise NotImplementedError

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)
            
    def get_e_cr_e_ic(acc_origin_list, acc_lba_list, vqa_acc:bool):
        if vqa_acc:
            e_cr = sum([1 if acc_lba > acc_origin and acc_origin < 0.5 else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if acc < 0.5 else 0 for acc in acc_origin_list]) * 100
            e_ic = sum([1 if acc_lba < acc_origin and acc_origin > 0.5 else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if acc > 0.5 else 0 for acc in acc_origin_list]) * 100
        else:
            e_cr = sum([1 if acc_lba and not acc_origin else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if not acc_origin else 0 for acc_origin in acc_origin_list]) * 100
            e_ic = sum([1 if not acc_lba and acc_origin else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if acc_origin else 0 for acc_origin in acc_origin_list]) * 100
        return e_cr, e_ic


def get_text_input(
    prompt_type:str="default",
    main_questions:List[str]='',
    sub_questions:List[str]='',
    sub_answers:List[str]='',
):
    assert prompt_type in ["default", "decomposer", "sub_answer", "recomposer"], f"Invalid prompt type: {prompt_type}"
    
    if prompt_type == "default": # for default vqa or generating sub-answer
        prompt = "Question: {main_question}? Short answer:"
        return [prompt.format(main_question=main_question.rstrip('?')) for main_question in main_questions]
    
    elif prompt_type == "decomposer":
        prompt = "Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {main_question}? Perception Question:"
        return [prompt.format(main_question=main_question.rstrip('?')) for main_question in main_questions]
    
    elif prompt_type == "sub_answer":
        prompt = "Question: {sub_question}? Short answer:"
        return [prompt.format(sub_question=sub_question.rstrip('?')) for sub_question in sub_questions]
        
    elif prompt_type == "recomposer":
        prompt = "Context: is the sky blue? no. are there clouds in the sky? yes. Question: what weather is likely? Short answer: rain.\nContext: {sub_question}? {sub_answer}. Question: {main_question}? Short answer:"
        return [prompt.format(main_question=main_question.rstrip('?'), sub_question=sub_question.rstrip('?'), sub_answer=sub_answer.rstrip('.')) 
                for main_question, sub_question, sub_answer in zip(main_questions, sub_questions, sub_answers)]
        
    else:
        raise NotImplementedError(f"Invalid prompt type: {prompt_type}")
    