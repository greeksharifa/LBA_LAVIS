import json
import os
from PIL import Image
from pprint import pprint

import torch
from torch.utils.data import DataLoader

try:
    from dataset.base_dataset import BaseDataset
except:
    from base_dataset import BaseDataset


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
        "dataset": "AOKVQA"
    },
    {
        'gt_ans': ['cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette', 'cigarette'],
        'question_id': '22jbM6gDxdaMaunuzgrsBB',
        'sub_answer_list': ['helmet', 'keys', 'cigarette', 'cigarette', 'flowers'],
        'sub_question_list': ['What is the motorcyclist holding?', "What is in the motorcyclist's pocket?", "What is in the woman's mouth?", "What is in the man's mouth?", 'What is in the snog of the motorcyclist?'],
        'text_input': "What is in the motorcyclist's mouth?",
        'vision': <PIL.Image.Image image mode=RGB size=640x569 at 0x7F109253D870>
    }
    len(dataset): 1145
    """
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        question_id = ann["question_id"]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        # image = self.vis_processor(image)
        # text_input = self.text_processor(ann["question"])
        text_input = ann["question"]
        
        
        sub_qa_list = self.sub_qas[str(question_id)] if hasattr(self, 'sub_qas') else None
        if sub_qa_list is None:
            sub_questions = None
            sub_answers = None
        elif type(sub_qa_list[0]) == list: # include sub_questions and sub_answers
            sub_questions = [sub_qa[0] for sub_qa in sub_qa_list]
            sub_answers = [sub_qa[1] for sub_qa in sub_qa_list]
        else:
            sub_questions = sub_qa_list
            sub_answers = None


        return {
            "vision": image,
            "text_input": text_input,
            "question_id": question_id,
            "gt_ans": ann["direct_answers"], # vqav2 answers list of str(len=10)
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
        } 
        
        
def main(ann_paths, split):
    dataset = AOKVQADataset(vis_processor=None, text_processor=None, 
                            vis_root='/data/coco/images/', 
                            ann_paths=ann_paths, 
                            num_data=-1, split=split)
    
    for i in range(len(dataset)):
        pprint(dataset[i], width=300)
        break
    print('len(dataset):', len(dataset))

if __name__ == '__main__':
    split = 'val' # 'train', 'val', 'test'
    ann_paths = [
        '/data/AOKVQA/annotations/aokvqa_v1p0_val.json',
        '/data/AOKVQA/sub_qas_val_xl_fewshot_vqaintrospect_unique.json'
    ]
    main(ann_paths, split)
    