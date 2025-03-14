import os
import json
import re
from typing import List, Union, Any
import logging

import numpy as np
import pandas as pd
import torch

from pathlib import Path
from abc import ABC, abstractmethod

from util.logger import get_logger
from util.utils import create_answer_mapping
from util.path import get_sub_qas_path


class BaseDataset(ABC):
    
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.logger = get_logger()

        runner_cfg = cfg.runner_cfg
        dataset_cfg = cfg.dataset_cfg
        model_cfg = cfg.model_cfg

        self.root_dir = Path(dataset_cfg.root_dir)
        self.vis_root = self.root_dir / dataset_cfg.vis_root

        # load annotation 
        self.annotation = []
        split = dataset_cfg.split
        ann_paths = [self.root_dir / ann_path for ann_path in dataset_cfg.ann_paths[split]]
        self.load_annotation(ann_paths)
        
        # add qid if not exists
        self._add_instance_ids(key="qid", prefix=self.__class__.__name__ + "_")
        for ann in self.annotation:
            ann["qid"] = str(ann["qid"])

        # load answer mapping
        if self.cfg.dataset_cfg.question_type != "open_ended": # == "multiple_choice":
            self.ANSWER_MAPPING = create_answer_mapping()

        # load sub-qas
        if runner_cfg.mode == "subq":
            pass
        elif runner_cfg.mode == "suba":
            sub_qs_path, sub_as_path = get_sub_qas_path(self.cfg)
            self.sub_qs = json.load(open(sub_qs_path, 'r')) if sub_qs_path.exists() else None
        else:
            sub_qs_path, sub_as_path = get_sub_qas_path(self.cfg)
            self.sub_qs = json.load(open(sub_qs_path, 'r')) if sub_qs_path.exists() else None
            self.sub_as = json.load(open(sub_as_path, 'r')) if sub_as_path.exists() else None
        # if runner_cfg.mode != "subqa":
        #     sub_qas_path, sub_as_path = get_sub_qas_path(self.cfg)
        #     self.sub_qas = json.load(open(sub_qas_path, 'r')) if sub_qas_path.exists() else None
        # else:
        #     self.sub_qas = None

        self.logger.info(f"Original dataset size: {len(self.annotation)}")
        self.logger.info(f"Original qids        : {self.annotation[0]['qid']} ... {self.annotation[-1]['qid']}")

        # adjust dataset size
        try:
            num_data = dataset_cfg.num_data.get(dataset_cfg.split, -1)
        except: 
            num_data = dataset_cfg.num_data
        if num_data != -1 and num_data < len(self.annotation): 
            # uniform_sampling
            idxs = np.linspace(0, len(self.annotation)-1, num_data, dtype=int)
            self.annotation = [self.annotation[i] for i in idxs]
            self.logger.info(f"Adjusted dataset size: {len(self.annotation)}")
            self.logger.info(f"Adjusted qids        : {self.annotation[0]['qid']} ... {self.annotation[-1]['qid']}")
        # if runner_cfg.start_pnt != -1:
        #     start_idx = int((len(self.annotation) / model_cfg.batch_size) * (runner_cfg.start_pnt / 100))
        #     self.logger.info(f"start_idx: {start_idx}")
        #     if runner_cfg.end_pnt != 100:
        #         end_idx = int((len(self.annotation) / model_cfg.batch_size) * (runner_cfg.end_pnt / 100))
        #         self.annotation = self.annotation[model_cfg.batch_size * start_idx:model_cfg.batch_size * end_idx]
        #         self.logger.info(f"end_idx: {end_idx}")
        #     else:
        #         self.annotation = self.annotation[model_cfg.batch_size * start_idx:]
        #     self.logger.info(f"Adjusted dataset size: {len(self.annotation)}")
        #     self.logger.info(f"Adjusted qids        : {self.annotation[0]['qid']} ... {self.annotation[-1]['qid']}")


        # evaluate by chatgpt
        # if kwargs.get("eval_by_chatgpt", False):
        #     self.logger.info("Enable: evaluating by ChatGPT")
        #     self.create_openai_client()

        # print dataset info
        self.logger.info(f"Dataset: {self.__class__.__name__}")
        self.logger.info(f"Loaded {len(self.annotation)} data")

        # load base answers if exists
        # self.base_answers = self.load_base_answers_from_path(model_cfg.base_answer_path, dataset_cfg.dataset_name)

    @abstractmethod
    def load_annotation(self, ann_paths: List[Path]):
        pass
          
    @abstractmethod
    def __getitem__(self, index):
        pass 
    
    def __len__(self):
        return len(self.annotation)

    def _add_instance_ids(self, key, prefix):
        for idx, ann in enumerate(self.annotation):
            if key not in ann:
                ann[key] = prefix + str(idx)
    
    def collater(self, samples):
        result = {}
        for key in samples[0].keys():
            result[key] = [sample[key] for sample in samples]
        
        return result
        
    def preprocess_annotation(self, qid, main_q, gt_ans):
        qid = str(qid)
        main_q = main_q.strip() #.rstrip("?") + "?"
        
        if self.cfg.dataset_cfg.question_type != "open_ended":
            gt_ans = self.ANSWER_MAPPING.get(gt_ans, gt_ans)
        
        if self.cfg.dataset_cfg.vqa_acc:
            gt_ans = [g.strip().lower() for g in gt_ans]
        else:
            gt_ans = gt_ans.strip().lower()
        
        return qid, main_q, gt_ans

    def get_subqs(self, ann):
        qid = ann["qid"]
        sub_qs = self.sub_qs[qid] if self.sub_qs else None
        if sub_qs is None:
            raise ValueError(f"sub_qs is not found for qid {qid}")
            # return None, None
        
        subq_list = sub_qs["subq_list"]
        conf_subq_list = sub_qs["conf_subq"][self.cfg.runner_cfg.confidence_type]
        
        return subq_list, conf_subq_list

    def get_subas(self, ann):
        raise NotImplementedError("get_subas is not implemented")


    # def get_subqas(self, ann):
    #     qid = ann["qid"]
    #     sub_qas = self.sub_qas[qid] if self.sub_qas else None
    #     if sub_qas is None:
    #         return None, None, None, None, None
        
    #     sub_q_list = sub_qas["sub_q_list"]
    #     sub_a_list = sub_qas["sub_a_list"]
    #     sub_a_conf_list = sub_qas["sub_a_conf_list"]
    #     sub_a_ppl_list = sub_qas["sub_a_ppl_list"]
    #     sub_a_min_prob_list = sub_qas["sub_a_min_prob_list"]
        
        
    #     if self.cfg.runner_cfg.get("LLM_Judge", False):
    #         # self.logger.info("Enable: using sub-QA judged by LLM")
    #         mode = self.cfg.runner_cfg.get("LLM_Judge", False)
    #         indices = sub_qas[f"judged_{mode}_indices"]
    #         # get element from indices
    #         sub_q_list = [sub_q_list[i] for i in indices]
    #         sub_a_list = [sub_a_list[i] for i in indices]
    #         sub_a_conf_list = [sub_a_conf_list[i] for i in indices]
    #         sub_a_ppl_list = [sub_a_ppl_list[i] for i in indices]
    #         sub_a_min_prob_list = [sub_a_min_prob_list[i] for i in indices]

    #     # preprocess sub-qas
    #     sub_q_list = [sub_q.strip().rstrip("?") + "?" for sub_q in sub_q_list]
    #     sub_a_list = [sub_a.strip().rstrip(".") + "." for sub_a in sub_a_list]
            
    #     return sub_q_list, sub_a_list, sub_a_conf_list, sub_a_ppl_list, sub_a_min_prob_list


    # def cleanse_prediction(self, prediction, main_question):
    #     prediction = prediction.strip().lower().rstrip(".")
    #     if self.cfg.runner_cfg.get("CoT", False):
    #         # base_pattern = r"\s*:?\s*(?:option)?\s*\(?([A-La-l])"
    #         if self.cfg.dataset_cfg.question_type == "multiple_choice":
    #             base_pattern = r"\s*[:|\s|\"|(?:option)]*\(?([A-La-l])"
    #         else:
    #             base_pattern = r"\s*[:|\s|\"]*(.*)"
    #         if "answer is" in prediction:
    #             # pattern = r"answer is\s*:?\s*(?:option)?\s*\(?([A-La-l])"
    #             pattern = "answer is" + base_pattern
    #             match = re.search(pattern, prediction)
    #             if match:
    #                 prediction = match.group(1).lower()
    #         else: # if "is" in prediction:
    #             pattern = "is" + base_pattern
    #             match = re.search(pattern, prediction)
    #             if match:
    #                 prediction = match.group(1).lower()
    #             else:
    #                 prediction = prediction.split(main_question[-10:])[-1].strip()
    #                 prediction = prediction[:3]

    #     if self.cfg.dataset_cfg.question_type != "open_ended":
    #         prediction = prediction.split(".")[0].strip()
    #         prediction = self.ANSWER_MAPPING.get(prediction, prediction)

    #     # remove parentheses
    #     if re.compile(r"\(([A-La-l])\)"):
    #         prediction = re.sub(r"\(([A-La-l])\)", r"\1", prediction)

    #     return prediction
    
    # def cleanse_target(self, target):
    #     return target.strip().lower().rstrip(".")

    # def get_accuracy(self, predicts, targets, main_question=None):

    #     def _get_acc(predict: str, target: Union[str, List[str]]):
    #         predict = self.cleanse_prediction(predict, main_question)
    #         # print("="*100 + '\n' + f"target: {target}" + f"\tpredict: {predict}")
    #         if self.cfg.dataset_cfg.vqa_acc:
    #             target = [self.cleanse_target(t) for t in target]
    #             return 1 if predict in target else 0
    #         else:
    #             target = self.cleanse_target(target)
    #             return 1 if predict.startswith(target) else 0
    #             # return 1 if predict == target else 0
                
    #     if isinstance(predicts, list):
    #         accs = [_get_acc(p, t) for p, t in zip(predicts, targets)]
    #     else:
    #         accs = _get_acc(predicts, targets)
    #     return accs


    # def load_base_answers_from_path(self, base_answer_path: Any, dataset_name: str):
    #     if self.cfg.runner_cfg.K == 0:
    #         return None
    #     path = base_answer_path.get(dataset_name, None)
    #     if path is None:
    #         return None
        
    #     loaded = json.load(open(path, 'r'))

    #     base_answers = {}
    #     for qid, result in loaded.items():
    #         base_answers[qid] = {
    #             "base_answer": result["base_answer"],
    #             "base_conf": result["base_conf"],
    #             "base_ppl": result["base_ppl"],
    #             "base_min_prob": result["base_min_prob"]
    #         }
    #     return base_answers

    # def load_base_answers(self, qids: List[str]):
    #     if self.base_answers is None:
    #         return None, None, None, None
        
    #     base_answers, base_confs, base_ppls, base_min_probs = [], [], [], []
    #     # load from self.base_answers
    #     for qid in qids:
    #         if qid not in self.base_answers:
    #             return None, None, None, None
    #         sample = self.base_answers[qid]
    #         base_answers.append(sample["base_answer"])
    #         base_confs.append(sample["base_conf"])
    #         base_ppls.append(sample["base_ppl"])
    #         base_min_probs.append(sample["base_min_prob"])

    #     return base_answers, base_confs, base_ppls, base_min_probs

# class OpenAIEvalMixin:
#     def create_openai_client(self):
#         api_key = json.load(open("temp/api_key.json", "r"))["LBA"]
#         from openai import OpenAI
#         self.client = OpenAI(api_key=api_key)
        
#         """
#         self.response_list:
#         [
#             {
#                 "main_question": ...,
#                 "outputs": ...,
#                 "targets": ...,
#                 "pred": "yes" or "no",
#                 "score": 0.0 ~ 5.0
#             },
#             {
#                 "main_question": "what are three people sitting on?",
#                 "outputs": "The three people in the video are sitting on a couch.",
#                 "targets": "couch",
#                 "pred": "yes",
#                 "score": 5
#             }
#         ]
#         """
#         path = os.path.join(self.output_dir, "response_df.csv")
#         if os.path.exists(path):
#             self.response_df = pd.read_csv(path) 
#             # self.response_list = json.load(open(path, 'r'))
#         else:
#             required_columns = [
#                 "main_question",
#                 "outputs",
#                 "targets",
#                 "pred",
#                 "score"
#             ]
#             self.response_df = pd.DataFrame(columns=required_columns)
#             # self.response_list = []
       
#     def save_response_list(self):
#         save_path = os.path.join(self.output_dir, "response_df.csv")
#         self.response_df.to_csv(save_path, index=False)
#         # json.dump(self.response_list, open(save_path, 'w'), indent=4)
#         return save_path
    
#     def get_openai_eval_response(self, pred_answer, gt_answer, main_question):
#         message = [
#             {
#                 "role": "system",
#                 "content":
#                     "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
#                     "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
#                     "------"
#                     "##INSTRUCTIONS: "
#                     "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
#                     "- Consider synonyms or paraphrases as valid matches.\n"
#                     "- Evaluate the correctness of the prediction compared to the answer."
#             },
#             {
#                 "role": "user",
#                 "content":
#                     "Please evaluate the following video-based question-answer pair:\n\n"
#                     f"Question: {main_question}\n"
#                     f"Correct Answer: {gt_answer}\n"
#                     f"Predicted Answer: {pred_answer}\n\n"
#                     "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
#                     "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
#                     "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
#                     "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
#             }
#         ]
#         completion = self.client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=message,
#             temperature=0.7,
#             max_tokens=800,
#             top_p=0.95,
#             frequency_penalty=0,
#             presence_penalty=0,
#             stop=None
#         )
#         # import pdb; pdb.set_trace()
#         response_message = completion.choices[0].message.content
#         response_dict = ast.literal_eval(response_message)
        
#         return response_dict
