import json

from typing import List
from pathlib import Path
from PIL import Image

from dataset.base_dataset import BaseDataset


class MMMU(BaseDataset):
    """
        # multiple-choice (simgle-image)
        {'answer': 'B',
        'explanation': '',
        'image_path_list': ['/data/MMMU/mmmu_images/validation/validation_Accounting_1_1.png'],
        'img_type': ['Tables'],
        'options': ['$6', '$7', '$8', '$9'],
        'question': '<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?',
        'question_id': 'validation_Accounting_1',
        'question_type': 'multiple-choice',
        'subfield': 'Managerial Accounting',
        'topic_difficulty': 'Medium'},

        # multiple-choice (multiple-image)
        {'answer': 'D',
        'explanation': '<image 2>The correct answer is:\n'
                        'Phytoplasmas: the symptoms of phytoplasma infection can look very similar to the regrowth after Glyphosate treatment. These symptoms are extremely typical of phytoplasma infection.\n'
                        'The incorrect answers were:\n'
                        'Bacteria: never induce these symptoms.\n'
                        "Fungi: fungal infection can induce a witches' broom /little leaf symptom but it is an unusual symptom of fungal infection.\n"
                        'Nematodes: can stunt plants and the developing leaves may appear smaller but they never induce this dramatic "little leaf" formation.\n',
        'image_path_list': ['/data/MMMU/mmmu_images/validation/validation_Agriculture_26_1.png', '/data/MMMU/mmmu_images/validation/validation_Agriculture_26_2.png'],
        'img_type': ['Photographs'],
        'options': ["I don't know and I don't want to guess", 'Nematodes', 'Fungi', 'Phytoplasmas', 'Bacteria'],
        'question': "<image 1> What group of pathogens, often mistaken for regrowth following glyphosate treatment, can cause a growth habit in blackberry plants that is near-identical to the 'little leaf' symptoms commonly witnessed post-glyphosate "
                    'treatment?',
        'question_id': 'validation_Agriculture_26',
        'question_type': 'multiple-choice',
        'subfield': 'Plant Pathology',
        'topic_difficulty': 'Easy'}

        # open-ended (single-image)
        {'answer': '1.06',
        'explanation': '',
        'image_path_list': ['/data/MMMU/mmmu_images/validation/validation_Architecture_and_Engineering_14_1.png'],
        'img_type': ['Diagrams', 'Technical Blueprints'],
        'options': [],
        'question': 'Using a finite summation, compute the  initial deflection at midspan for the beam in  Figure P8.42. Given: E = 3000 kips/in.2 .  Use 3-ft segments. Assume I = 0.5IG. <image 1>',
        'question_id': 'validation_Architecture_and_Engineering_14',
        'question_type': 'open',
        'subfield': 'Structural Engineering',
        'topic_difficulty': 'Hard'}
    """
    # image 수:
    """
        split: dev       : Counter({1: 146, 2: 2, 4: 2}) 
        split: validation: Counter({1: 857, 2: 24, 4: 8, 5: 6, 3: 5}) 
        split: test      : Counter({1: 9702, 2: 409, 4: 202, 5: 110, 3: 67, 6: 8, 7: 2}) 
    """
    def __init__(self, cfg, **kwargs):
        self.mme_eval_results = {}
        super().__init__(cfg, **kwargs)

    def load_annotation(self, ann_paths: List[Path]):
        for ann_path in ann_paths:
            samples = json.load(open(ann_path, 'r'))
            for sample in samples:
                question = sample['question']
                
                if "<image" not in question:
                    images = [Image.open(image_path) for image_path in sample['image_path_list']]
                    image_path_list = sample['image_path_list']
                else:
                    images = []
                    image_path_list = []
                    
                    for img_number, image_path in enumerate(sample['image_path_list'], 1):
                        if f"<image {img_number}>" in question:
                            images.append(Image.open(image_path))
                            image_path_list.append(image_path)
                        else:
                            break
                    
                if len(images) == 0:
                    import pdb; pdb.set_trace()
                
                gt_ans = sample['answer']

                ann = {
                    "image_list": images,
                    "vpath": image_path_list,
                    "main_q": question,
                    "qid": sample['question_id'],
                    "question_type": sample['question_type'],
                    "candidate_list": sample['options'],
                    "gt_ans": gt_ans,
                    "type": sample['subfield'], # sample["topic_difficulty"]
                }
                ann = self.preprocess_annotation(ann)
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
