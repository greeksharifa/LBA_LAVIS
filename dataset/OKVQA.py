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


class OKVQADataset(BaseDataset):
    """
    {
        "question_id": 2971475, 
        "question": "What sport can you use this for?", 
        "answer": ["race", "race", "race", "race", "race", "race", "motocross", "motocross", "ride", "ride"], 
        "image": "val2014/COCO_val2014_000000297147.jpg", 
        "dataset": "OKVQA"
    },
    {
        'gt_ans': ['race', 'race', 'race', 'race', 'race', 'race', 'motocross', 'motocross', 'ride', 'ride'],
        'question_id': 2971475,
        'sub_answer_list': ['transportation', 'racing', 'a motorcycle', 'motorcycle', 'racing'],
        'sub_question_list': ['What is the object used for?', 'What kind of sport can you use this for?', 'What can you use this for?', 'What is this object used for?', 'What sport can this be used for?'],
        'text_input': 'What sport can you use this for?',
        'vision': <PIL.Image.Image image mode=RGB size=640x480 at 0x7FC92C4C98D0>
    }
    len(dataset): 5046
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
            "gt_ans": ann["answer"], # vqav2 answers list of str(len=10)
            "sub_question_list": sub_questions,
            "sub_answer_list": sub_answers,
        }
        
        
def main(ann_paths, split):
    dataset = OKVQADataset(vis_processor=None, text_processor=None, 
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
        '/data/OKVQA/annotations/vqa_val_eval.json',
        '/data/OKVQA/sub_qas_val_xl_fewshot_vqaintrospect_unique.json'
    ]
    main(ann_paths, split)
    