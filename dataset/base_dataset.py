import os
import json
from typing import List
import pandas as pd
import torch

from torch.utils.data import Dataset
from transformers import InstructBlipVideoProcessor

def load_dataset(datasets_cfg, split='val', n_supple=0):
    if datasets_cfg.dataset_name == "VQA_Introspect":
        from dataset.VQA_Introspect import VQAIntrospectDataset
        cls = VQAIntrospectDataset
    elif datasets_cfg.dataset_name == "AOKVQA":
        from dataset.AOKVQA import AOKVQADataset
        cls = AOKVQADataset
    elif datasets_cfg.dataset_name == "OKVQA":
        from dataset.OKVQA import OKVQADataset
        cls = OKVQADataset
    elif datasets_cfg.dataset_name == "DramaQA":
        from dataset.DramaQA import DramaQAEvalDataset
        cls = DramaQAEvalDataset
    elif datasets_cfg.dataset_name in ["NExTQA", "STAR"]:#, "VLEP"]:# "How2QA"]:
        from dataset.VideoQA import VideoEvalDataset
        cls = VideoEvalDataset
    elif datasets_cfg.dataset_name == "TVQA": 
        from dataset.TVQA import TVQAEvalDataset
        cls = TVQAEvalDataset
    elif datasets_cfg.dataset_name == "VLEP": # '/data1/VLEP/vlep_frames/friends_s03e09_seg02_clip_07_ep.mp4'
        from dataset.VLEP import VLEPEvalDataset
        cls = VLEPEvalDataset
    else:
        raise NotImplementedError(f"in dataset.base_dataset.py, load_dataset() | Invalid dataset name: {datasets_cfg.dataset_name}")
        
    dataset = cls(
        vis_processor=None,
        text_processor=None,
        vis_root=os.path.join(datasets_cfg.root_dir, datasets_cfg.vis_root),
        ann_paths=[os.path.join(datasets_cfg.root_dir, path) for path in datasets_cfg.ann_paths.get(datasets_cfg.split, split)],
        num_data=datasets_cfg.num_data,
        vqa_acc=datasets_cfg.vqa_acc,
        n_frms=datasets_cfg.get("n_frms", 4),
        datasets_cfg=datasets_cfg,
        n_supple=n_supple, #datasets_cfg.get("n_supple"),
    )
    
    return dataset
    

class BaseDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[], num_data=-1, **kwargs):
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
        
        # self.vqa_acc = vqa_acc
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._add_instance_ids()

    @staticmethod
    def answer_mapping(answer):
        return answer

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        result = {}
        for k, v in samples[0].items():
            # if isinstance(v, torch.Tensor): # no use, 240809
            #     result[k] = torch.stack([sample[k] for sample in samples], dim=0)
            # else:
            result[k] = [sample[k] for sample in samples]
        
        return result

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
    
    def get_accuracy(self, outputs, targets):#, match1ok=False):
        """
        args
        - outputs: str          or list of str.         shape: [bsz]
        - targets: list of str  or list of list of str. shape: [bsz, 10]
        """
        def _get_acc(out, target):
            if self.vqa_acc:
                return out in target
                # if match1ok:
                #     return out in target
                
                # num_match = sum([out == t for t in target])
                # return min(1.0, num_match / 3.0)
            else:
                if type(out) == str:
                    if len(out) == 1:
                        out = '(' + out + ')'
                    elif len(out) > 3 and out[0] == '(':
                        out = out[:3]
                return 1.0 if out == target else 0.0
            
        if isinstance(outputs, (str, int)):
            acc = _get_acc(outputs, targets)
            return acc
        else:
            acc_list = []
            for out, target_list in zip(outputs, targets):
                acc = _get_acc(out, target_list)
                acc_list.append(acc)
            return acc_list
    
    
def get_train_examplar(datasets_cfg):
    train_dataset = load_dataset(datasets_cfg, split='train')
    example = train_dataset[0]
    
    main_question = example["text_input"].strip().rstrip('?')
    candidate_list = example["candidate_list"]
    gt_ans = train_dataset.answer_mapping(example["gt_ans"])
    answer_sentence = example["answer_sentence"].strip().rstrip('.')
    
    prompt = """Context: {main_question}? {answer_sentence}.\nQuestion: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is {gt_ans}"""
    for candidate in candidate_list:
        choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
    
    return prompt.format(main_question=main_question, answer_sentence=answer_sentence, choices=choices, gt_ans=gt_ans)
            
def get_text_input(
    prompt_type:str="default",
    main_questions:List[str]='',
    sub_questions:List[str]='',
    sub_answers:List[str]='',
    candidate_lists: List[List[str]]=[],
    gt_answers:List[str]=[],
    question_ids:List[str]=[],
    **kwargs,
):
    video_default_examplar = """Context: Who is waving his hand with a smile? Haeyoung1 is waving her hand with a smile. Who is about to hug Haeyoung1? Dokyung is about to hug Haeyoung1.
Question: Why did Dokyung pull Haeyoung1's arm hard?
Choices:
(A) Dokyung pulled Haeyoung1's arm to hug her hard.
(B) It is because Dokyung did not want Haeyoung1 to fall.
(C) This is because Dokyung and Haeyoung1 were dancing on the street.
(D) Dokyung pulled Haeyoung1's arm since Haeyoung1 tried to run away.
(E) Because Dokyung needed Haeyoung1 to go to the police station.
Answer: The answer is (A)\n"""
    # add <video> in front of prompt if video_llava
    if prompt_type == "default_image": # for default vqa or generating sub-answer
        prompt = "Question: {main_question}? Short answer:"
        return [prompt.format(main_question=main_question.rstrip('?')) for main_question in main_questions]
    
    elif prompt_type == "decomposer":
        prompt = "Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {main_question}? Perception Question:"
        return [prompt.format(main_question=main_question.rstrip('?')) for main_question in main_questions]
    
    elif prompt_type == "sub_answer":
        prompt = "Question: {sub_question}? Short answer:"
        return [prompt.format(sub_question=sub_question.rstrip('?')) for sub_question in sub_questions]
        
    elif prompt_type == "recomposer_image":
        examplar = "Context: is the sky blue? no. are there clouds in the sky? yes. Question: what weather is likely? Short answer: rain.\n"
        prompt = examplar + "Context: {sub_question}? {sub_answer}. Question: {main_question}? Short answer:"
        return [prompt.format(main_question=main_question.rstrip('?'), sub_question=sub_question.rstrip('?'), sub_answer=sub_answer.rstrip('.')) 
                for main_question, sub_question, sub_answer in zip(main_questions, sub_questions, sub_answers)]
    
    elif prompt_type == "default_video":
        prompt = video_default_examplar if kwargs.get("add_examplar", False) else ""
        prompt += "Question: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is "
        ret = []
        for main_question, candidate_list in zip(main_questions, candidate_lists):
            choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
            ret.append(prompt.format(main_question=main_question.rstrip('?'), choices=choices))
        return ret
        
        """
        [SOS] Video: <v_1> <v_2> · · · <v_Nv>
        Question: <question>
        Choices:
        (A) <option 1>
        (B) <option 2>
        (C) <option 3>
        (D) <option 4>
        (E) <option 5>
        Answer: The answer is <answer> [EOS]
        """
    elif prompt_type == "recomposer_video":
        prompt = kwargs.get("examplar") if kwargs.get("train_recomposer_examplar", False) else video_default_examplar
        prompt += "Context: {sub_question}? {sub_answer}.\nQuestion: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is "
        
        ret = []
        for main_question, sub_question, sub_answer, candidate_list in zip(main_questions, sub_questions, sub_answers, candidate_lists):
            choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
            ret.append(prompt.format(main_question=main_question.rstrip('?'), sub_question=sub_question.rstrip('?'), sub_answer=sub_answer.rstrip('.'), choices=choices))
        return ret
    elif prompt_type == "recomposer_video_description":
        prompt = "Video Description: {description}.\nQuestion: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is "
        
        ret = []
        for description, main_question, candidate_list in zip(kwargs.get('descriptions'), main_questions, candidate_lists):
            choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
            ret.append(prompt.format(description=description, main_question=main_question.rstrip('?'), choices=choices))
        return ret
    else:
        raise NotImplementedError(f"Invalid prompt type: {prompt_type}")
    
    
def get_sevila_input(
    prompt_type:str="default",
    # text_inputs:List[str]=[],
    batch: List=[],
    processor: InstructBlipVideoProcessor=None,
    sub_questions:List[str]=[],
    sub_answers:List[str]=[],
    device="cuda:0",
    **kwargs,
):
    main_questions = batch['text_input']
    candidate_lists = batch['candidate_list']
    gt_answers = batch['gt_ans']
    question_ids = batch['question_id']

    # pixel_values = []
    # for video in vision: # video: [n_frms, 640, 480]
    #     # [n_frms, 640, 480] -> [n_frms, 3, 224, 224]
    #     pixel_values.append(self.processor(images=video, return_tensors="pt", padding=True)['pixel_values'])  # [n_frms, 3, 224, 224]
    # # [n_frms, 3, 224, 224] -> [bsz, n_frms, 3, 224, 224]
    # stacked = torch.stack(pixel_values, dim=0)#.to(self.model.device)
    # # 미적용중.."""# [bsz, n_frms, 3, 224, 224] -> [bsz, 3, n_frms, 224, 224]"""
    # inputs["pixel_values"] = stacked#.transpose(2, 1)
    
    ret = {
        "video": processor(batch['vision'], return_tensors="pt", padding=True)['pixel_values'].to(device),
        "qa_input": [],
        "loc_input": [],
        "qa_output": gt_answers,
        "question_id": question_ids,
        "duration": [1 for _ in range(len(main_questions))],
    }
    
    default_examplar = """Context: Who is waving his hand with a smile? Haeyoung1 is waving her hand with a smile. Who is about to hug Haeyoung1? Dokyung is about to hug Haeyoung1.
Question: Why did Dokyung pull Haeyoung1's arm hard?
Option A: Dokyung pulled Haeyoung1's arm to hug her hard.
Option B: It is because Dokyung did not want Haeyoung1 to fall.
Option C: This is because Dokyung and Haeyoung1 were dancing on the street.
Option D: Dokyung pulled Haeyoung1's arm since Haeyoung1 tried to run away.
Option E: Because Dokyung needed Haeyoung1 to go to the police station.
select the correct answer from the options: (A)\n"""

    for i, (main_question, candidate_list) in enumerate(zip(main_questions, candidate_lists)):
        # qa_prompt ex. 'Question: how do the two man play the instrument? 
        # Option A: roll the handle. Option B: tap their feet. Option C: strum the string. Option D: hit with sticks. Option E: pat with hand. 
        # Considering the information presented in the frame, select the correct answer from the options.'
        qa_prompt = f"Question: {main_question} "
        qa_prompt += ' '.join([f"Option {chr(65+i)}: {c}" for i, c in enumerate(candidate_list)])
        qa_prompt += " Considering the information presented in the frame, select the correct answer from the options."
        
        # loc_prompt ex.'Question: how do the two man play the instrument? 
        # Options: (roll the handle. tap their feet. strum the string. hit with sticks. pat with hand.) 
        # Does the information within the frame provide the necessary details to accurately answer the given question?'
        loc_prompt = f"Question: {main_question}"
        loc_prompt += f" Options: ({' '.join(candidate_list)})"
        loc_prompt += " Does the information within the frame provide the necessary details to accurately answer the given question?"

        if prompt_type == "default":
            pass
        elif prompt_type == "decomposer":
            raise NotImplementedError("sevila decomposer not implemented")
        elif prompt_type == "recomposer":
            examplar = "" if kwargs.get("train_recomposer_examplar", False) else default_examplar
            qa_prompt = examplar + f"Context: {sub_questions[i].rstrip('?')}? {sub_answers[i]}.\n" + qa_prompt
            # loc_prompt: pass
        else:
            raise NotImplementedError(f"Invalid prompt type: {prompt_type}")
    
        ret["qa_input"].append(qa_prompt)
        ret["loc_input"].append(loc_prompt)

    return ret
