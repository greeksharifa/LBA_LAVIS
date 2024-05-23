import os
import json
import random

from PIL import Image

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset


class VQAIntrospectEvalDataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. gqa/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        annotation = json.load(open(ann_paths[0]))
        
        self.annotation = []
        for k, v in annotation.items():
            v.update({"question_id": k})
            self.annotation.append(v)
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        """
        ann example:
        {
            "question_id": "284885001",
            "image_id": 284885,
            "reasoning_answer_most_common": "no",
            "reasoning_question": "is the house new?",
            "introspect": [
                {
                    "sub_qa": [
                        {
                            "sub_question": "is the window of the house boarded up?",
                            "sub_answer": "yes"
                        },
                        {...}
                    ],
                    "pred_q_type": "reasoning"
                }, 
                {...}
            ]
        }
        """
        assert False, "*" * 160 + "\nTODO: Imgospect dataset does not support __getitem__"
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann["answer"]]
        weights = [1]

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }