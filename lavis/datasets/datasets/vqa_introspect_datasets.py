"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random
import logging
from collections import OrderedDict

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from lavis.datasets.datasets.caption_datasets import CaptionDataset, CaptionEvalDataset
from lavis.common.registry import registry

from colors import Colors, print_sample

COCOCapDataset = CaptionDataset


def _init_VQAIntrospectSingleSubQ(ann_paths, max_sample_num=99999999):
    """
    set annotation of DataLoader
    1 sample: (MQ, SQ)
    Args:
        ann_paths: list of annotation_file_path
        max_sample_num: limit of the number of samples

    Returns: annotation. about train=166927/val=71714 samples
    """
    annotation = []
    
    sub_question_id = 0
    for ann_path in ann_paths:
        logging.info(f"Loading {ann_path}")
        json_data = json.load(open(ann_path, "r"))
        
        for main_question_id, value in json_data.items():
            image_id = value["image_id"]
            main_question = value["reasoning_question"]
            main_answer = value["reasoning_answer_most_common"]
            
            sub_q_set = set()
            
            for introspect in value["introspect"]:
                sub_qa_list = introspect["sub_qa"]
                pred_q_type = introspect["pred_q_type"]
                
                for sub_qa in sub_qa_list:
                    if sub_qa["sub_question"] in sub_q_set:
                        pass  # 중복
                    else:
                        sub_q_set.add(sub_qa["sub_question"])
                        _sample = {
                            "image_id": image_id,
                            "main_question_id": int(main_question_id),
                            "main_question": main_question,
                            "main_answer": main_answer,
                            "sub_question_id": sub_question_id,
                            "sub_question": sub_qa["sub_question"],
                            "sub_answer": sub_qa["sub_answer"],
                            "pred_q_type": pred_q_type,
                        }
                        annotation.append(_sample)
                        
                        sub_question_id += 1
            if sub_question_id >= max_sample_num:
                break
    
    return annotation


def _init_VQAIntrospectMultipleSubQ(ann_paths, max_sample_num=99999999):
    """
    set annotation of DataLoader
    1 sample: (MQ, [SQ_i])
    Args:
        ann_paths: list of annotation_file_path
        max_sample_num: limit of the number of samples

    Returns: annotation. about train=55799-720/val=21677-1116 samples
    """
    annotation = []
    from collections import Counter
    counter = Counter()
    
    sub_question_id = 0
    for ann_path in ann_paths:
        logging.info(f"Loading {ann_path}")
        json_data = json.load(open(ann_path, "r"))
        
        for main_question_id, value in json_data.items():
            image_id = value["image_id"]
            main_question = value["reasoning_question"]
            main_answer = value["reasoning_answer_most_common"]
            
            sub_q_set = set()
            
            sub_q_list = []
            sub_a_list = []
            sub_q_id_list = []
            
            pred_q_type_list = []
            
            for introspect in value["introspect"]:
                sub_qa_list = introspect["sub_qa"]
                pred_q_type = introspect["pred_q_type"]
                
                for sub_qa in sub_qa_list:
                    if sub_qa["sub_question"] in sub_q_set:
                        pass  # 중복
                    else:
                        sub_q_set.add(sub_qa["sub_question"])
                        
                        sub_q_list.append(sub_qa["sub_question"])
                        sub_a_list.append(sub_qa["sub_answer"])
                        pred_q_type_list.append(pred_q_type)
                        
                        sub_q_id_list.append(sub_question_id)
                        sub_question_id += 1

            counter[len(sub_q_set)] += 1
            if len(sub_q_set) == 0:
                continue
            
            _sample = {
                "image_id": image_id,
                "main_question_id": int(main_question_id),
                "main_question": main_question,
                "main_answer": main_answer,
                "sub_question_id_list": sub_q_id_list,
                "sub_question_list": sub_q_list,
                "sub_answer_list": sub_a_list,
                "pred_q_type_list": pred_q_type_list,
            }
            annotation.append(_sample)
            
            if len(annotation) >= max_sample_num:
                break
    
    logging.info(f"counter: {sorted(list(counter.items()))}")
    return annotation


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


# _prompt_file_path = os.path.join(registry.get_path("cache_root"), "coco_gt")
_prompt_file_path = "prompts.json"


# 1개의 sub-question 생성
def _apply_VQAIntrospect_Questioner_SingleSubQ_prompt(main_question, sub_question):
    prompts = json.load(open(_prompt_file_path, "r"))["Questioner_SingleSubQ"]
    prompt = random.choice(prompts)
    
    text_input = prompt.format(main_question)
    text_output = sub_question
    
    return text_input, text_output


# sequential하게 sub-question 생성
def _apply_VQAIntrospect_Questioner_MultipleSubQ_prompt(main_question, sub_question_list, sub_answer_list, split):
    # logging.info('in _apply_VQAIntrospect_Questioner_MultipleSubQ_prompt()')
    
    prompts = json.load(open(_prompt_file_path, "r"))["Questioner_MultipleSubQ"]
    # text_output으로 1개, previous generated sub_qa로 0~2개 비복원추출
    # sub_qa_pair_num = random.randint(1, min(3, len(sub_question_list)))     # index 0은 text_output으로 사용
    sub_qa_pair_num = min(1+3, len(sub_question_list))     # index 0은 text_output으로 사용
    sub_qa_indices = random.sample(range(len(sub_question_list)), sub_qa_pair_num)
    
    text_input = prompts["init_prompt"].format(main_question)
    if sub_qa_pair_num > 1:
        text_input += prompts["after_prompt"]
        for i in range(1, sub_qa_pair_num):
            index = sub_qa_indices[i]
            if i > 1:
                text_input += ', '
            if prompts["pair_prompt"].count('{}') == 4:
                text_input += prompts["pair_prompt"].format(i, sub_question_list[index], i, sub_answer_list[index])
            else:
                text_input += prompts["pair_prompt"].format(i, sub_question_list[index])
        else:
            text_input += '.'
    
    if split == 'train':
        text_output = sub_question_list[sub_qa_indices[0]]
    else:
        text_output = set(sub_question_list)
        for i in range(1, sub_qa_pair_num):
            index = sub_qa_indices[i]
            text_output.remove(sub_question_list[index])
        text_output = '\n'.join(text_output)
    
    return text_input, text_output


# 생성된 sub-qa들로 main-question에 대한 답변 생성
def _apply_VQAIntrospect_Reasoner_prompt(main_question, main_answer, sub_question_list, sub_answer_list):
    # logging.info('in _apply_VQAIntrospect_Reasoner_prompt()')
    
    prompts = json.load(open(_prompt_file_path, "r"))["Reasoner"]
    # generated sub_qa를 1~3개 비복원추출
    sub_qa_pair_num = random.randint(1, min(3, len(sub_question_list)))
    sub_qa_indices = random.sample(range(len(sub_question_list)), sub_qa_pair_num)
    
    text_input = prompts["init_prompt"].format(main_question)
    # text_input = ""
    for i in range(0, sub_qa_pair_num):
        index = sub_qa_indices[i]
        # if i > 0:
        #     text_input += ', '
        if prompts["pair_prompt"].count('{}') == 4:
            text_input += prompts["pair_prompt"].format(i+1, sub_question_list[index], i+1, sub_answer_list[index])
        else:
            text_input += prompts["pair_prompt"].format(sub_question_list[index], sub_answer_list[index])
    
    text_input += prompts["final_prompt"].format(main_question)
    
    text_output = main_answer
    
    return text_input, text_output


# question에 대한 답변 생성
def _apply_VQAIntrospect_Answerer_prompt(sub_question, sub_answer):
    # logging.info('in _apply_VQAIntrospect_Answerer_prompt()')
    prompts = json.load(open(_prompt_file_path, "r"))["Answerer"]
    
    text_input = prompts["init_prompt"].format(sub_question)
    text_output = sub_answer
    
    return text_input, text_output


def _get_text_input_output(prompt_type, ann, split="train"):
    # logging.info('in _get_text_input_output()')
    if prompt_type == "Questioner_SingleSubQ":
        text_input, text_output = _apply_VQAIntrospect_Questioner_SingleSubQ_prompt(
            ann["main_question"],
            ann["sub_question"] if split == "train" else '\n'.join(ann["sub_question_list"]),
        )
    elif prompt_type == "Questioner_MultipleSubQ":
        text_input, text_output = _apply_VQAIntrospect_Questioner_MultipleSubQ_prompt(
            ann["main_question"],
            ann["sub_question_list"],
            ann["sub_answer_list"],
            split,
        )
    elif prompt_type == "Reasoner":
        text_input, text_output = _apply_VQAIntrospect_Reasoner_prompt(
            ann["main_question"],
            ann["main_answer"],
            ann["sub_question_list"],
            ann["sub_answer_list"],
        )
    elif prompt_type == "Answerer":
        text_input, text_output = _apply_VQAIntrospect_Answerer_prompt(
            ann["sub_question"],
            ann["sub_answer"],
        )
    else:
        raise Exception("prompt_type must be specified in lavis.configs.datasets.vqa_introspect.<blabla>.yaml")
    
    return text_input, text_output
    

class VQAIntrospectQARCapDataset(CaptionDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, prompt_type):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_root = vis_root
        self.prompt_type = prompt_type
        
        if prompt_type in ["Questioner_SingleSubQ", "Answerer"]:
            self.annotation = _init_VQAIntrospectSingleSubQ(ann_paths)#, 100)
        elif prompt_type in ["Questioner_MultipleSubQ", "Reasoner"]:
            self.annotation = _init_VQAIntrospectMultipleSubQ(ann_paths)#, 100)
        
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            # img_id = ann["main_question_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        self._add_instance_ids()
        
        logging.info(f"{Colors.BRIGHT_MAGENTA}VQAIntrospectMultipleCapDataset len: {len(self.annotation)}{Colors.RESET}")
        logging.info(f"{Colors.BRIGHT_MAGENTA}prompt_type: {self.prompt_type}{Colors.RESET}")

        
    def __getitem__(self, index):
        ann = self.annotation[index]

        # train2014/COCO_train2014_000000216531.jpg
        image_path = os.path.join(self.vis_root, f'train2014/COCO_train2014_{ann["image_id"]:012}.jpg')
        # image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        
        text_input, text_output = _get_text_input_output(self.prompt_type, ann, "train")
        
        return {
            "image": image,
            "text_input": text_input,
            "text_output": text_output,
        }


class VQAIntrospectQARCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, prompt_type):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        # super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.vis_root = vis_root
        self.prompt_type = prompt_type
        
        if prompt_type in ["Answerer"]:
            self.annotation = _init_VQAIntrospectSingleSubQ(ann_paths, 50)
        elif prompt_type in ["Questioner_SingleSubQ", "Questioner_MultipleSubQ", "Reasoner"]:
            self.annotation = _init_VQAIntrospectMultipleSubQ(ann_paths, 50)
        
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            # img_id = ann["main_question_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        self._add_instance_ids()
        
        logging.info(f"{Colors.BRIGHT_MAGENTA}VQAIntrospectMultipleCapEvalDataset len: {len(self.annotation)}{Colors.RESET}")
        logging.info(f"{Colors.BRIGHT_MAGENTA}prompt_type: {self.prompt_type}{Colors.RESET}")
    
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        
        # val2014/COCO_val2014_000000265814.jpg
        image_path = os.path.join(self.vis_root, f'val2014/COCO_val2014_{ann["image_id"]:012}.jpg')
        # image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        
        image = self.vis_processor(image)
        
        text_input, text_output = _get_text_input_output(self.prompt_type, ann, "eval")
        
        _return = {
            "image": image,
            "image_id": ann["image_id"],
            "main_question_id": ann["main_question_id"],
            "instance_id": ann["instance_id"],
            "text_input": text_input,
            "prompt": text_input,
            # "sub_question_id": ann["sub_question_id"],
            "text_output": text_output,
        }
        # logging.info(f"_return: {_return}")

        return _return

