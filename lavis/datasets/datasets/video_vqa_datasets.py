"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import glob
import json
import os
from collections import OrderedDict

import torch
from PIL import Image

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)

from colors import Colors, print_color


class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )


class VideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def _build_class_labels(self, ans_path):
        ans2label = json.load(open(ans_path))

        self.class_labels = ans2label

    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __getitem__(self, index):
        assert (
            self.class_labels
        ), f"class_labels of {__class__.__name__} is not built yet."

        ann = self.annotation[index]
        
        print_color(msg="in class VideoQADataset", color=Colors.BRIGHT_RED)

        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        frms = self.vis_processor(vpath)
        question = self.text_processor(ann["question"])

        return {
            "video": frms,
            "text_input": question,
            "answers": self._get_answer_label(ann["answer"]),
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }
    

import random

_prompt_file_path = "prompts.json"


class DramaQASQDataset(VideoQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.dataset_split = ann_paths[0].split('_')[-2]
        print(f'dataset split: {self.dataset_split}')

        self.shots_per_scene = 'multiple'
        self.max_sub_questions = 2
        self.full_evaluation = False

        assert self.shots_per_scene == 'multiple' or self.max_sub_questions == 1, 'for shots_per_scene="single", need max_sub_questions=1'

        self.sq_prompts = json.load(open(_prompt_file_path, 'r'))

        all_annotation = self.annotation
        
        shot_annotation_by_vname = {}
        for ann in all_annotation:
            vname = ann["vid"]
            if vname.endswith("0000"): # scene
                continue
            
            image_path_list = []
            vpath = os.path.join(self.vis_root, vname.replace('_', '/'))
            for frm in glob.glob(vpath + "/*"):
                image_path_list.append(frm)

            ann = ann.copy()
            ann["image_path_list"] = image_path_list
            shot_annotation_by_vname[vname] = ann

        successes = 0
        fails = 0

        if self.shots_per_scene == 'multiple':
            scene_annotation = []
            for ann in all_annotation:
                vname = ann["vid"]
                if not vname.endswith("0000"): # not scene
                    continue
                
                shot_anns = {}
                vpath = os.path.join(self.vis_root, vname[:-5].replace('_', '/'))
                for shot in glob.glob(vpath + "/*"):
                    shot_name = '_'.join(shot.split('/')[-3:])
                    if shot_name not in shot_annotation_by_vname:
                        fails += 1
                        continue
                    shot_anns[shot_name] = shot_annotation_by_vname[shot_name]
                    successes += 1

                if len(shot_anns) == 0:
                    # print(f'no shots for scene {vname}!')
                    continue

                ann = ann.copy()
                ann["shot_anns"] = shot_anns
                scene_annotation.append(ann)
        elif self.shots_per_scene == 'single':
            scene_annotation = []
            for ann in all_annotation:
                vname = ann["vid"]
                if not vname.endswith("0000"): # not scene
                    continue

                scene_anns_by_shot = {}
                vpath = os.path.join(self.vis_root, vname[:-5].replace('_', '/'))
                for shot in glob.glob(vpath + "/*"):
                    shot_anns = {}
                    
                    shot_name = '_'.join(shot.split('/')[-3:])
                    if shot_name not in shot_annotation_by_vname:
                        fails += 1
                        continue
                    shot_anns[shot_name] = shot_annotation_by_vname[shot_name]
                    successes += 1

                    if len(shot_anns) == 0:
                        # print(f'no shots for scene {vname}!')
                        continue

                    ann = ann.copy()
                    ann["shot_anns"] = shot_anns
                    ann["scene_anns_by_shot"] = scene_anns_by_shot
                    scene_anns_by_shot[shot_name] = ann
                    scene_annotation.append(ann)
        else:
            assert False, f'unrecognized value for `shorts_per_scene`: {self.shots_per_scene}'

        # print(f'{successes} successes, {fails} fails.')

        self.annotation = scene_annotation
        
    def __len__(self):
        if self.full_evaluation or self.dataset_split == 'train':
            return super().__len__()
        return 10 # HACK: hard-coded

    def __getitem__(self, index):
        if not self.full_evaluation and self.dataset_split != 'train':
            index = random.randint(0, super().__len__() - 1)

        # print("self.annotation[index]:", self.annotation[index])
        # print("len(self.annotation):", len(self.annotation))
        # self.annotation[index]: {
        #   'videoType': 'scene',
        #   'answers': [
        #       'Since Dokyung wanted to kick the old man.',
        #       'Just because Dokyung wanted to buy some snacks from the old man.',
        #       "Because Dokyung tried to give Dokyung's umbrella to the old man.",
        #       'As Dokyung wanted to take the clothes away from the old man.',
        #       'It was because Dokyung wanted to run away from Haeyoung1.'
        #   ],
        #   'vid': 'AnotherMissOh17_001_0000',
        #   'qid': 13041,
        #   'shot_contained': [25, 49],
        #   'q_level_mem': 3,
        #   'que': 'Why did Dokyung go to the old man?',
        #   'q_level_logic': 4,
        #   'instance_id': '0'
        # }
        # len(self.annotation): 4019
        ann = self.annotation[index]

        # shot_ann = random.choice(list(ann['shot_anns'].values()))
        shot_anns = list(ann['shot_anns'].values())
        random.shuffle(shot_anns)
        
        image_path_list = shot_anns[0]["image_path_list"]
        
        # TODO: 경로 수정
        vname = ann["vid"] # AnotherMissOh17_001_0000
        
        # 1
        image_path = random.choice(image_path_list)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        frms = image#.unsqueeze(0)
        
        # 2
        # frms = None
        # for image_path in image_path_list:
        #     image = Image.open(image_path).convert("RGB")
        #     image = self.vis_processor(image)
        #     if frms is None:
        #         frms = image.unsqueeze(0).unsqueeze(2)
        #     else:
        #         frms = torch.cat((frms, image.unsqueeze(0).unsqueeze(2)), dim=2)
                
        '''
        self.class_labels; {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        len(self.class_labels); 5
        type(self.class_labels); <class 'dict'>
        '''
        
        # question = ann["que"]
        # text_input = ""
        # text_input += "Choose a number of answer from the following question.\n"
        # for i, cand in enumerate(ann["answers"]):
        #     text_input += "{}. {}\n".format(i, cand)
        # text_input += "Question: {}\n".format(question)
        # text_input += "Answer:"

        prompt_type, = random.choices(['questioner', 'answerer', 'reasoner'], weights=[5,4,1])

        main_question = ann["que"]
        main_answer = ann["answers"][ann["correct_idx"]] if "correct_idx" in ann else ""

        if self.dataset_split != 'train':
            prompt_type = 'evaluation'
            num_sub_questions = 0
            text_input = main_question
            text_output = main_answer
        else:
            questioner_prompts = self.sq_prompts["Questioner_MultipleSubQ"]
            answerer_prompts = self.sq_prompts["Answerer"]
            reasoner_prompts = self.sq_prompts["Reasoner"]

            questioner_prompt = questioner_prompts["init_prompt"].format(main_question)
            
            reasoner_prompt = reasoner_prompts["init_prompt"]
            if reasoner_prompts["init_prompt"].count('{}') == 1:
                reasoner_prompt = reasoner_prompts["init_prompt"].format(main_question)  # reasoner_prompt

            sub_question_list = []
            sub_answer_list = []

            num_sub_questions = random.randint(1, min(len(shot_anns), self.max_sub_questions))
            for i in range(1, 1 + num_sub_questions):
                # Questioner
                sub_question = shot_anns[i-1]['que']
                if i == num_sub_questions and prompt_type == 'questioner':
                    text_input = questioner_prompt
                    if i > 1:
                        text_input += '.'
                    text_output = sub_question
                    break
                sub_question_list.append(sub_question)
                if i == 1:
                    questioner_prompt += questioner_prompts["after_prompt"]
                else:
                    questioner_prompt += ', '
                questioner_prompt += questioner_prompts["pair_prompt"].format(i, sub_question)

                # Answerer
                answerer_prompt = answerer_prompts["init_prompt"].format(sub_question)
                sub_answer = shot_anns[i-1]["answers"][shot_anns[i-1]["correct_idx"]]
                if i == num_sub_questions and prompt_type == 'answerer':
                    text_input = answerer_prompt
                    text_output = sub_answer
                    break
                sub_answer_list.append(sub_answer)

                # Reasoner
                if i > 1 and not reasoner_prompts["pair_prompt"].endswith('. '):
                    reasoner_prompt += ', '
                if reasoner_prompts["pair_prompt"].count('{}') == 4:
                    reasoner_prompt += reasoner_prompts["pair_prompt"].format(i, sub_question, i, sub_answer)
                else:
                    reasoner_prompt += reasoner_prompts["pair_prompt"].format(sub_question, sub_answer)
                    
            if prompt_type == 'reasoner':
                reasoner_prompt += reasoner_prompts["final_prompt"].format(main_question)
                text_input = reasoner_prompt
                text_output = main_answer
            
        return {
            "image": frms,
            "text_input": text_input,
            "text_output": text_output,
            "answer": ann["correct_idx"] if "correct_idx" in ann else -1,
            "answer_list": ann["answers"],
            "gt_ans": main_answer,
            "question_id": ann["qid"],
            "instance_id": ann["instance_id"],
            "prompt_type": prompt_type,
            "num_sub_questions": str(num_sub_questions), # HACK: for visualization
        }


class DramaQAEvalDataset(VideoQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        # print("self.annotation[index]:", self.annotation[index])
        # print("len(self.annotation):", len(self.annotation))
        # self.annotation[index]: {
        #   'videoType': 'scene',
        #   'answers': [
        #       'Since Dokyung wanted to kick the old man.',
        #       'Just because Dokyung wanted to buy some snacks from the old man.',
        #       "Because Dokyung tried to give Dokyung's umbrella to the old man.",
        #       'As Dokyung wanted to take the clothes away from the old man.',
        #       'It was because Dokyung wanted to run away from Haeyoung1.'
        #   ],
        #   'vid': 'AnotherMissOh17_001_0000',
        #   'qid': 13041,
        #   'shot_contained': [25, 49],
        #   'q_level_mem': 3,
        #   'que': 'Why did Dokyung go to the old man?',
        #   'q_level_logic': 4,
        #   'instance_id': '0'
        # }
        # len(self.annotation): 4019
        ann = self.annotation[index]
        
        # print_color(msg="in class DramaQAEvalDataset", color=Colors.BRIGHT_RED)
        
        # image_path = os.path.join(self.vis_root, ann["image"])
        # image = Image.open(image_path).convert("RGB")
        
        image_path_list = []
        
        # TODO: 경로 수정
        vname = ann["vid"] # AnotherMissOh17_001_0000
        
        
        if vname.endswith("0000"): # scene
            vpath = os.path.join(self.vis_root, vname[:-5].replace('_', '/'))
            for shot in glob.glob(vpath + "/*"):
                for frm in glob.glob(shot + "/*"):
                    image_path_list.append(frm)
                    break
        else: # shot
            vpath = os.path.join(self.vis_root, vname.replace('_', '/'))
            for frm in glob.glob(vpath + "/*"):
                image_path_list.append(frm)
        
        # print_color(msg="image_list: {}, {}".format(len(image_path_list), image_path_list[0]), color=Colors.BRIGHT_RED)
        # shot_contained = ann["shot_contained"]
        # print_color(msg="vname: {}".format(vname), color=Colors.BRIGHT_RED)
        # print_color(msg="vpath: {}".format(vpath), color=Colors.BRIGHT_RED)
        
        frms = None
        for image_path in image_path_list:
            image = Image.open(image_path).convert("RGB")
            # print_color(msg="image_path: {}".format(image_path), color=Colors.BRIGHT_RED)
            image = self.vis_processor(image)
            # print_color(msg="image: {}".format(image.shape), color=Colors.BRIGHT_RED)
            
            # 1
            frms = image#.unsqueeze(0)
            break
            # 2
            # if frms is None:
            #     frms = image.unsqueeze(0).unsqueeze(2)
            # else:
            #     frms = torch.cat((frms, image.unsqueeze(0).unsqueeze(2)), dim=2)
                
        # frms = self.vis_processor(vpath)
        # print_color(msg="frms    : {}, {}".format(type(frms), frms.shape), color=Colors.BRIGHT_RED)
        # question = self.text_processor(ann["que"])
        # print_color(msg="question: {}".format(question), color=Colors.BRIGHT_RED)

        # assert False, "DramaQAEvalDataset __getitem__() is not implemented yet."
        
        # print('self.class_labels;', self.class_labels)
        # print('len(self.class_labels);', len(self.class_labels))
        # print('type(self.class_labels);', type(self.class_labels))
        '''
        self.class_labels; {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
        len(self.class_labels); 5
        type(self.class_labels); <class 'dict'>
        '''
        
        question = ann["que"]
        # text_input = ""
        # text_input += "Choose a number of answer from the following question.\n"
        # for i, cand in enumerate(ann["answers"]):
        #     text_input += "{}. {}\n".format(i, cand)
        # text_input += "Question: {}\n".format(question)
        # text_input += "Answer:"
        
        return {
            "image": frms,
            "text_input": question, #text_input,
            # "answers": self._get_answer_label(ann["correct_idx"]), # answer list가 아니라 answer 하나만 들어가야 함
            "answer": ann["correct_idx"],
            "answer_list": ann["answers"],
            "gt_ans": ann["answers"][ann["correct_idx"]],
            "question_id": ann["qid"],
            "instance_id": ann["instance_id"],
        }
        
        