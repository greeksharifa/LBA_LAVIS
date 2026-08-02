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
from util.path import get_sub_qas_path, get_output_dir
from subqa.schema import (
    normalize_hierarchy_config,
    project_depth_one_questions,
    validate_confidence_mapping,
    validate_suba_artifact_entry,
    validate_subq_tree,
)
from util.artifacts import (
    MANIFEST_FILENAME,
    STAGE_DEPENDENCIES,
    validate_completed_stage,
)


class BaseDataset(ABC):
    
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.logger = get_logger()

        runner_cfg = cfg.runner_cfg
        dataset_cfg = cfg.dataset_cfg
        model_cfg = cfg.model_cfg
        self.hierarchy_config = normalize_hierarchy_config(runner_cfg)

        self.root_dir = Path(dataset_cfg.root_dir)
        self.vis_root = self.root_dir / dataset_cfg.vis_root

        
        # adjust dataset size
        try:
            self.num_data = dataset_cfg.num_data.get(dataset_cfg.split, -1)
        except: 
            self.num_data = dataset_cfg.num_data

        # load answer mapping
        if self.cfg.dataset_cfg.question_type != "open_ended": # == "multiple_choice":
            self.ANSWER_MAPPING = create_answer_mapping()
        # load annotation 
        self.annotation = []
        split = dataset_cfg.split
        ann_paths = [self.root_dir / ann_path for ann_path in dataset_cfg.ann_paths[split]]
        self.load_annotation(ann_paths)
        
        # add qid if not exists
        self._add_instance_ids(key="qid", prefix=self.__class__.__name__ + "_")
        for ann in self.annotation:
            ann["qid"] = str(ann["qid"])


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

        # Dependencies are intentionally loaded after sampling so qid validation
        # applies to the exact examples in this run.
        self._load_stage_dependencies()
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
        
    def preprocess_annotation(self, ann):
        qid = str(ann["qid"])
        main_q = ann["main_q"]
        gt_ans = ann["gt_ans"]
        
        main_q = main_q.strip() #.rstrip("?") + "?"
        
        if ann["question_type"] in ("multiple_choice", "multiple-choice"):
            gt_ans = self.ANSWER_MAPPING.get(gt_ans, gt_ans)
        
        if isinstance(gt_ans, list):
            gt_ans = [g.strip().lower() for g in gt_ans]
        elif self.cfg.dataset_cfg.vqa_acc:
            gt_ans = [g.strip().lower() for g in gt_ans]
        else:
            gt_ans = gt_ans.strip().lower()
        
        ann["qid"] = qid
        ann["main_q"] = main_q
        ann["gt_ans"] = gt_ans

        return ann
    
    def load_additional_attr(self, ann, result):
        if "data_type" not in result:
            result["data_type"] = self.cfg.dataset_cfg.data_type
            
        for dependency in self._required_artifact_stages():
            value, confidence = self._get_base_or_subs(ann, dependency)
            value_key = "base_answer" if dependency == "base" else f"{dependency}_list"
            confidence_key = (
                "conf_base" if dependency == "base" else f"conf_{dependency}_list"
            )
            result.update({value_key: value, confidence_key: confidence})
            if (
                dependency == "subq"
                and self.hierarchy_config.enabled
                and self.cfg.runner_cfg.mode == "suba"
            ):
                result["subq_tree"] = self.subqs[str(ann["qid"])]["subq_tree"]
        
        if self.cfg.runner_cfg.few_shot:
            # Get few-shot samples for the current sub-category
            few_shot_samples = self.few_shot_samples.get(ann["sub_category"], [])
            few_shot_str = "\n\n".join(few_shot_samples) if few_shot_samples else ""
            result.update({
                "few_shot_samples": few_shot_str,
            })

        return result

    def _required_artifact_stages(self):
        mode = self.cfg.runner_cfg.mode
        if mode in STAGE_DEPENDENCIES:
            return STAGE_DEPENDENCIES[mode]
        return ("subq", "suba")

    def _load_stage_dependencies(self):
        subq_path, suba_path = get_sub_qas_path(self.cfg)
        paths = {
            "subq": subq_path,
            "suba": suba_path,
            "base": get_output_dir(self.cfg) / "base_outputs.json",
        }
        selected_qids = [ann["qid"] for ann in self.annotation]
        manifest_path = get_output_dir(self.cfg) / MANIFEST_FILENAME
        required_stages = self._required_artifact_stages()
        self.dependency_generations = {}

        for stage in required_stages:
            manifest = validate_completed_stage(
                manifest_path,
                stage,
                self.cfg,
                selected_qids,
            )
            self.dependency_generations[stage] = manifest["stages"][stage][
                "generation_id"
            ]

        for stage in required_stages:
            path = paths[stage]
            if not path.is_file():
                raise FileNotFoundError(f"required {stage} artifact not found: {path}")
            with path.open("r", encoding="utf-8") as handle:
                artifact = json.load(handle)
            self._validate_stage_artifact(stage, path, artifact, selected_qids)
            setattr(self, f"{stage}s", artifact)

    def _validate_stage_artifact(self, stage, path, artifact, qids):
        value_key = "base_answer" if stage == "base" else f"{stage}_list"
        confidence_key = f"conf_{stage}"
        confidence_type = self.cfg.runner_cfg.confidence_type

        if not isinstance(artifact, dict):
            raise KeyError(f"{path}: artifact must be keyed by qid")
        for qid in qids:
            if qid not in artifact:
                raise KeyError(f"{path}: missing qid {qid}")
            entry = artifact[qid]
            if not isinstance(entry, dict):
                raise KeyError(f"{path}: qid {qid} must map to an object")
            for key in (value_key, confidence_key):
                if key not in entry:
                    raise KeyError(f"{path}: qid {qid} missing key {key}")
            confidence = entry[confidence_key]
            if not isinstance(confidence, dict) or confidence_type not in confidence:
                raise KeyError(
                    f"{path}: qid {qid} missing key "
                    f"{confidence_key}.{confidence_type}"
                )
            if not self.hierarchy_config.enabled:
                continue
            if stage == "subq":
                self._validate_hierarchy_subq_entry(path, qid, entry)
            elif stage == "suba" and self.cfg.runner_cfg.mode == "refined":
                tree = self.subqs[qid]["subq_tree"]
                self._validate_hierarchy_suba_entry(path, qid, entry, tree)

    def _validate_hierarchy_subq_entry(self, path, qid, entry):
        try:
            if "subq_tree" not in entry:
                raise KeyError("missing key subq_tree")
            tree = entry["subq_tree"]
            validate_subq_tree(
                tree,
                branching=self.hierarchy_config.branching,
            )
            validate_confidence_mapping(
                entry["conf_subq"],
                "conf_subq",
                self.hierarchy_config.required_confidence_types,
            )
            for node in tree["nodes"]:
                if (
                    node["depth"] < tree["max_depth"]
                    and node["expansion_confidence"] is None
                ):
                    raise ValueError(
                        f"node {node['id']!r} expansion_confidence is required"
                    )
                if node["depth"] < tree["max_depth"]:
                    validate_confidence_mapping(
                        node["expansion_confidence"],
                        f"node {node['id']!r} expansion_confidence",
                        self.hierarchy_config.required_confidence_types,
                    )
            expected = project_depth_one_questions(
                tree,
                expected_count=self.hierarchy_config.branching[0],
            )
            if entry["subq_list"] != expected:
                raise ValueError(
                    "subq_list is not the exact depth-1 tree projection"
                )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: qid {qid} invalid subq_tree: {error}"
            ) from error

    def _validate_hierarchy_suba_entry(self, path, qid, entry, tree):
        try:
            validate_suba_artifact_entry(
                entry, tree, self.hierarchy_config
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: qid {qid} invalid answers_by_node: {error}"
            ) from error

    def _get_base_or_subs(self, ann, mode: str):
        qid = ann["qid"]
        subs = getattr(self, f"{mode}s")[qid]
        if subs is None:
            raise ValueError(f"{mode}s is not found for qid {qid}")

        target_key = "base_answer" if mode == "base" else f"{mode}_list"
        sub_list = subs[target_key]
        conf_sub_list = subs[f"conf_{mode}"][self.cfg.runner_cfg.confidence_type]
        
        return sub_list, conf_sub_list

    def get_score(self, pred, gt_ans, question_type: str, main_q: str = None):
        if self.cfg.dataset_cfg.vqa_acc:
            raise NotImplementedError("VQA accuracy is not implemented")
        else:
            pred = self.cleanse_answer(pred, question_type, main_q)
            gt_ans = gt_ans.strip().lower().rstrip(".")
            return 1 if pred == gt_ans else 0

    def cleanse_answer(self, ans: str, question_type: str, main_q: str = None):
        ans = ans.strip().lower().rstrip(".")
        if self.cfg.runner_cfg.mode == "CoT":
            # base_pattern = r"\s*:?\s*(?:option)?\s*\(?([A-La-l])"
            if self.cfg.dataset_cfg.question_type == "multiple_choice":
                base_pattern = r"\s*[:|\s|\"|(?:option)]*\(?([A-La-l])"
            else:
                base_pattern = r"\s*[:|\s|\"]*(.*)"
            if "answer is" in ans:
                # pattern = r"answer is\s*:?\s*(?:option)?\s*\(?([A-La-l])"
                pattern = "answer is" + base_pattern
                match = re.search(pattern, ans)
                if match:
                    ans = match.group(1).lower()
            else: # if "is" in prediction:
                pattern = "is" + base_pattern
                match = re.search(pattern, ans)
                if match:
                    ans = match.group(1).lower()
                else:
                    if main_q is not None:
                        ans = ans.split(main_q[-10:])[-1].strip()
                    ans = ans[:3]

        if question_type in ("multiple_choice", "multiple-choice"):
            ans = ans.split(".")[0].strip()
            ans = self.ANSWER_MAPPING.get(ans, ans)

        # remove parentheses
        if re.compile(r"\(([A-La-l])\)"):
            ans = re.sub(r"\(([A-La-l])\)", r"\1", ans)

        return ans

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
