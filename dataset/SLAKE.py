import json
import os
import pickle
from PIL import Image
from pprint import pprint

import torch
from torch.utils.data import DataLoader

try:
    from dataset.base_dataset import BaseDataset
except:
    from base_dataset import BaseDataset


class SLAKEDataset(BaseDataset):
    """
    len: 2099
    {
        'img_id': 0, 
        'img_name': 'xmlab0/source.jpg', 
        'question': 'What modality is used to take this image?', 
        'answer': 'MRI', 
        'q_lang': 'en', 
        'location': 'Abdomen', 
        'modality': 'MRI', 
        'answer_type': 'OPEN', 
        'base_type': 'vqa', 
        'content_type': 'Modality', 
        'triple': ['vhead', '_', '_'], 
        'qid': 9835
    }
    """
    def __getitem__(self, index):
        ann = self.annotation[index]
        question_id = ann["qid"]

        image_path = os.path.join(self.vis_root, f'{ann["img_name"]}')
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
    dataset = SLAKEDataset(vis_processor=None, text_processor=None, vis_root='/data/Slake1.0/imgs/', 
                             ann_paths=ann_paths, num_data=-1, split=split)
    for i in range(len(dataset)):
        pprint(dataset[i], width=200)
        break
    print('len(dataset):', len(dataset))

if __name__ == '__main__':
    split = 'val' # 'train', 'val', 'test'
    ann_paths = [
        '/data/Slake1.0/validate.json'
    ]
    main(ann_paths, split)
    