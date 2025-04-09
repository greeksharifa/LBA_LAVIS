import json

from typing import List
from pathlib import Path
from PIL import Image

from dataset.base_dataset import BaseDataset


class DramaQA(BaseDataset):
    """
        # multiple-choice (simgle-video)
        >>> pprint(val[0], width=350)
        {'answers': ['Haeyoung1 and Dokyung are in love and the two went through many things before starting to date.',
                    "Haeyoung1 is Dokyung's sister and the two often hung out each other.",
                    'Haeyoung1 and Dokyung are best friends and the two live in the same apartment.',
                    "Haeyoung1 is Dokyung's cousin and the two are very close.",
                    'Haeyoung1 and Dokyung are a married couple and Haeyoung1 is pregnant.'],
        'correct_idx': 0,
        'q_level_logic': 3,
        'q_level_mem': 3,
        'qid': 3205,
        'que': 'How is the relationship between Haeyoung1 and Dokyung when the two hug and kiss each other?',
        'shot_contained': [25, 81],
        'vid': 'AnotherMissOh14_001_0000',
        'videoType': 'scene'}
    """
    # image 수:
    """
        split: dev       : Counter({1: 146, 2: 2, 4: 2}) 
        split: validation: Counter({1: 857, 2: 24, 4: 8, 5: 6, 3: 5}) 
        split: test      : Counter({1: 9702, 2: 409, 4: 202, 5: 110, 3: 67, 6: 8, 7: 2}) 
    """

    # @abstractmethod
    def load_annotation(self, ann_paths: List[Path]):
        ann_path = ann_paths[0]
        samples = json.load(open(ann_path, 'r'))
        for sample in samples:
            gt_ans = sample['correct_idx']
            gt_ans = self.ANSWER_MAPPING.get(gt_ans, gt_ans)
            ann = {
                "main_q": sample['que'],
                "candidate_list": sample['answers'],
                "gt_ans": gt_ans,
                "qid": sample['qid'],
                "question_type": "multiple_choice",
                "video_list": [Image.open(self.vis_root / f"{sample['vid']}.mp4")],
                "vpath": [self.vis_root / f"{sample['vid']}.mp4"],
            }
            self.annotation.append(ann)
    
    def __getitem__(self, index):
        ann = self.annotation[index]

        result = {
            "vision": ann["image_list"],
            "vpath": ann["vpath"],
            "main_q": ann["main_q"],
            "qid": ann["qid"],
            "gt_ans": ann["gt_ans"],
            "candidate_list": ann["candidate_list"],
            "question_type": ann["question_type"], # "multiple-choice" or "open-ended"
        }
        result = self.load_additional_attr(ann, result)
        
        return result
