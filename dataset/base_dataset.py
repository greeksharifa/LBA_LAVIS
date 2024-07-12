import json
from typing import List
import pandas as pd
import torch

from torch.utils.data import Dataset


def load_dataset(datasets_cfg):
    if datasets_cfg.dataset_name == "VQA_Introspect":
        from dataset.VQA_Introspect import VQAIntrospectDataset
        cls = VQAIntrospectDataset
    elif datasets_cfg.dataset_name == "AOKVQA":
        from dataset.AOKVQA import AOKVQADataset
        cls = AOKVQADataset
    elif datasets_cfg.dataset_name == "OKVQA":
        from dataset.OKVQA import OKVQADataset
        cls = OKVQADataset
    else:
        raise NotImplementedError(f"in dataset.base_dataset.py, load_dataset() | Invalid dataset name: {datasets_cfg.dataset_name}")
        
    dataset = cls(
        vis_processor=None,
        text_processor=None,
        vis_root=datasets_cfg.vis_root,
        ann_paths=datasets_cfg.ann_paths.get(datasets_cfg.split, 'val'),
        num_data=datasets_cfg.num_data,
        vqa_acc=datasets_cfg.vqa_acc
    )
    
    return dataset
    

class BaseDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[], num_data=-1, vqa_acc=False):
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

        if num_data != -1:
            self.annotation = self.annotation[:num_data]
            # import random
            # self.annotation = random.sample(self.annotation, num_data)
            
        print('len of self.annotation : ', len(self.annotation))

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        self.vqa_acc = vqa_acc

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
            
    def get_e_cr_e_ic(self, acc_origin_list, acc_lba_list):
        if self.vqa_acc:
            e_cr = sum([1 if acc_lba > acc_origin and acc_origin < 0.5 else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if acc < 0.5 else 0 for acc in acc_origin_list]) * 100
            e_ic = sum([1 if acc_lba < acc_origin and acc_origin > 0.5 else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if acc > 0.5 else 0 for acc in acc_origin_list]) * 100
        else:
            e_cr = sum([1 if acc_lba and not acc_origin else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if not acc_origin else 0 for acc_origin in acc_origin_list]) * 100
            e_ic = sum([1 if not acc_lba and acc_origin else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if acc_origin else 0 for acc_origin in acc_origin_list]) * 100
        return e_cr, e_ic
    
    def get_accuracy(self, outputs, targets, match1ok=False):
        """
        args
        - outputs: str          or list of str.         shape: [bsz]
        - targets: list of str  or list of list of str. shape: [bsz, 10]
        """
        def _get_acc(out, target):
            if self.vqa_acc:
                if match1ok:
                    return out in target
                
                num_match = sum([out == t for t in target])
                return min(1.0, num_match / 3.0)
            else:
                return 1.0 if out == target else 0.0
            
        if isinstance(outputs, str):    
            acc = _get_acc(outputs, targets)
            return acc
        else:
            acc_list = []
            for out, target_list in zip(outputs, targets):
                acc = _get_acc(out, target_list)
                acc_list.append(acc)
            return acc_list
            
            
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
    