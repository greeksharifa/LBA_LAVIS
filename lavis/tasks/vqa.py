"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import torch
from tqdm import tqdm

from lavis.common.utils import is_convertible_to_int
import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.tasks.base_task import BaseTask


@registry.register_task("vqa")
class VQATask(BaseTask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
        sample_id_key = "",
        ques_files=dict(),
        anno_files=dict(),
        valid_splits=['val']
    ):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = ques_files
        self.anno_files = anno_files

        # generalize to non coco data
        self.sample_id_key = sample_id_key

        self.valid_splits = valid_splits

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)

        prompt = run_cfg.get("prompt", "")

        # generalize to non coco data
        sample_id_key = run_cfg.get("sample_id_key", "instance_id")
        ques_files = run_cfg.get("ques_files", dict())
        anno_files = run_cfg.get("anno_files", dict())
        valid_splits = run_cfg.get("valid_splits", ["val"])


        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
            sample_id_key = sample_id_key,
            ques_files=ques_files,
            anno_files=anno_files,
            valid_splits=valid_splits
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for ds_name, dataset in datasets.items():
            for split in self.valid_splits:
                if split not in dataset:
                    print(f"Split {split} not found in {ds_name}.")
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file
                else:
                    if split not in self.ques_files: # precomputed and passed in task builder
                        self.ques_files[split] = os.path.join(registry.get_path("cache_root"),f'{ds_name}_gt', f'{ds_name}_{split}_questions.json')
                        self.anno_files[split] = os.path.join(registry.get_path("cache_root"), f'{ds_name}_gt', f'{ds_name}_{split}_annotations.json')
                        if dist_utils.get_rank() == 0:
                            os.makedirs(os.path.join(registry.get_path("cache_root"),f'{ds_name}_gt'), exist_ok=True)
                            try:
                                convert_to_coco_gt(dataset, self.ques_files[split], self.anno_files[split], split, self.sample_id_key)
                            except:
                                pass # tasks like vizwiz with no gt answer
                try:
                    self.answer_list = dataset[split].answer_list
                except AttributeError:
                    # if answer_list is not provided, then set it to None
                    pass

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets

    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        for answer, ques_id in zip(answers, question_id):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            if ques_id != int and is_convertible_to_int(ques_id):
                ques_id = int(ques_id)
            pred_qa_pairs.append({"question_id": ques_id, "answer": answer})

        return pred_qa_pairs

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)

        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Use official VQA evaluation script to report metrics.
        """
        metrics = {}

        if split in self.ques_files and split in self.anno_files:
            vqa = VQA(self.anno_files[split], self.ques_files[split])
            vqa_result = vqa.loadRes(
                resFile=result_file, quesFile=self.ques_files[split]
            )
            # create vqaEval object by taking vqa and vqaRes
            # n is precision of accuracy (number of places after decimal), default is 2
            vqa_scorer = VQAEval(vqa, vqa_result, n=2)
            logging.info("Start VQA evaluation.")
            vqa_scorer.evaluate()

            # print accuracies
            overall_acc = vqa_scorer.accuracy["overall"]
            metrics["agg_metrics"] = overall_acc

            logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
            logging.info("Per Answer Type Accuracy is the following:")

            for ans_type in vqa_scorer.accuracy["perAnswerType"]:
                logging.info(
                    "%s : %.02f"
                    % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
                )
                metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

            with open(
                os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")
        return metrics

def convert_to_coco_gt(data, outpath_questions, outpath_annotations, split, sample_id_key):
    if split not in data:
        return
    questions_data = {'info':"", 'task_type':"", 'data_type':"", 'license':"", 'data_subtype':"", 'questions':[]}
    annotations_data = {'info':"", 'task_type':"", 'data_type':"", 'license':"", 'data_subtype':"", 'annotations':[]}
    print("Generating ground truth annotations...")
    # print(data.keys(), outpath_questions, outpath_annotations, split, sample_id_key)
    # dict_keys(['val']) /data1/vqa_introspect_gt/vqa_introspect_val_questions.json 
    # /data1/vqa_introspect_gt/vqa_introspect_val_annotations.json val instance_id
    for ann in tqdm(data[split]):
        if ann == None:
            continue
        # if ann[sample_id_key] not in img_ids:
        #     continue
        ques_id = ann["question_id"]
        ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
        if ques_id != int and is_convertible_to_int(ques_id):
            ques_id = int(ques_id)
        questions_data["questions"].append({"question": ann["text_input"], "image_id": ann[sample_id_key], "question_id": ques_id})
        annotations_data["annotations"].append({
            "question_type": "" if "question_type" not in ann else ann["question_type"],
            "multiple_choice_answer": ann["answers"][0] if isinstance(ann["answers"], list) else ann["answers"],
            "answers": [{"answer":ans, "answer_id":i} for i,ans in enumerate(ann["answers"])] if isinstance(ann["answers"], list) else [{"answer":ann["answers"], "answer_id":0}], 
            "image_id": ann[sample_id_key], 
            "question_id": ques_id,
            "answer_type": "" if "answer_type" not in ann else ann["answer_type"],
        })
       
    json.dump(questions_data, open(outpath_questions, 'w'))
    print(f"Saved questions data at {outpath_questions}")
    json.dump(annotations_data, open(outpath_annotations, 'w'))
    print(f"Saved annotation data at {outpath_annotations}")

        

@registry.register_task("aok_vqa")
class AOKVQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
        )

        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["direct_answers"]

        for pred_answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            pred_qa_pairs.append(
                {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer}
            )

        return pred_qa_pairs

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Implementing accuracy computation for AOKVQA, see
        https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py#L45 for details.
        """
        # TODO add evaluation for multi-choice

        results = json.load(open(result_file, "r"))
        acc = []

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            pred = res["pred_ans"]
            gt_ans = res["gt_ans"]

            num_match = sum([pred == gt for gt in gt_ans])
            vqa_acc = min(1.0, num_match / 3.0)

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics

    @dist_utils.main_process
    def _save_result_leaderboard(self, results):
        """
        Saving the results in the format required for leaderboard evaluation.

        [TODO] add support for multi-choice.
        """
        result_leaderboard = dict()
        for res in results:
            result_leaderboard[res["question_id"]] = {
                "direct_answer": res["pred_ans"],
                "multiple_choice": "",
            }

        result_file = registry.get_path("result_dir") + "_leaderboard.json"

        with open(result_file, "w") as f:
            json.dump(result_leaderboard, f)

        logging.info(f"Saved results for leaderboard evaluation at {result_file}")



@registry.register_task("gqa")
class GQATask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        
        for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs
    
    def build_datasets(self, cfg):
        datasets = BaseTask.build_datasets(self,cfg)

        # get question file, annotation file and anwser list in COCO format
        for ds_name, dataset in datasets.items():
            for split in dataset:
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."

        return datasets
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        TODO: add other evaluation metrics for GQA
        """

        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return

            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]

            # if self.inference_method == "generate":
            pred = vqa_tool.processPunctuation(pred)
            pred = vqa_tool.processDigitArticle(pred)

            # added to ensure that the ground truth format of answers is as expected for non-gqa but similar tasks
            gt_ans = vqa_tool.processPunctuation(gt_ans)
            gt_ans = vqa_tool.processDigitArticle(gt_ans)

            vqa_acc = 1 if pred == gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics


@registry.register_task("discrn_qa")
class DisCRNTask(VQATask):
    def valid_step(self, model, samples):
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )

        if answers == None: # corrupt videos
            return []
            
        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["answer"]
        
        for answer, ques_id, gt_answer in zip(answers, question_id, gt_answers):
            ques_id = int(ques_id.item()) if isinstance(ques_id, torch.Tensor) else ques_id
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs


    def build_datasets(self, cfg):
        datasets = BaseTask.build_datasets(self, cfg)
        return datasets
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        results = json.load(open(result_file, "r"))
        acc = []
        vqa_tool = VQAEval()

        for res in results:
            gt_ans = res["gt_ans"]
            pred = res["pred_ans"]
            
            # gt_ans = [vqa_tool.processPunctuation(g) for g in gt_ans]
            # gt_ans = [vqa_tool.processDigitArticle(g) for g in gt_ans]

            # if self.inference_method == "generate":
            pred = vqa_tool.processPunctuation(pred)
            pred = vqa_tool.processDigitArticle(pred)

            tokenized_pred = pred.strip().split(" ")
            for ans in gt_ans:
                if ans in tokenized_pred:
                    pred = ans
                    break

            vqa_acc = 1 if pred in gt_ans else 0

            acc.append(vqa_acc)

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics


class VQALBATask(VQATask):
    def __init__(
        self,
        num_beams,
        max_len,
        min_len,
        evaluate,
        num_ans_candidates,
        inference_method="rank",
        prompt="",
        sample_id_key = "",
        ques_files=dict(),
        anno_files=dict(),
        valid_splits=['val'],
        # surprisal_threshold=1e-5,
    ):
        super().__init__(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
            sample_id_key = sample_id_key,
            ques_files=ques_files,
            anno_files=anno_files,
            valid_splits=valid_splits,
        )
        
        # LBA
        # self.surprisal_threshold = surprisal_threshold

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.get("num_beams", 3)
        max_len = run_cfg.get("max_len", 10)
        min_len = run_cfg.get("min_len", 1)

        evaluate = run_cfg.get("evaluate", False)

        inference_method = run_cfg.get("inference_method", "rank")
        num_ans_candidates = run_cfg.get("num_ans_candidates", 128)

        prompt = run_cfg.get("prompt", "")

        # generalize to non coco data
        sample_id_key = run_cfg.get("sample_id_key", "instance_id")
        ques_files = run_cfg.get("ques_files", dict())
        anno_files = run_cfg.get("anno_files", dict())
        valid_splits = run_cfg.get("valid_splits", ["val"])
        
        # LBA
        # surprisal_threshold = run_cfg.get("surprisal_threshold", 1e-5) # meaning of default value: almost always generate sub-q
        

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
            sample_id_key = sample_id_key,
            ques_files=ques_files,
            anno_files=anno_files,
            valid_splits=valid_splits,
            # surprisal_threshold=surprisal_threshold
        )

@registry.register_task("vqa_introspect")
# class VQAIntrospectTask(VQALBATask):
class VQAIntrospectTask(VQATask):
    def valid_step(self, model, samples):
        # print('self.prompt in task instance:', self.prompt)
        answers = model.predict_answers_by_lba(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
            prompt=self.prompt,
        )
        
        pred_qa_pairs = []
        
        question_id = samples["question_id"]
        # --------------------------------------------------------------------------------------
        # exact match with reasoning_answer_most_common
        # gt_answers = samples["reasoning_answer_most_common"]
        # --------------------------------------------------------------------------------------
        # original vqa evaluation
        gt_answers = samples["gt_ans"]

        if "confidences" in answers:
            output_texts_origin = answers["output_texts_origin"]
            output_texts_lba = answers["output_texts_lba"]
            confidences = answers["confidences"]
            for output_text_origin, output_text_lba, ques_id, gt_answer, confidence in zip(output_texts_origin, output_texts_lba, question_id, gt_answers, confidences):
                pred_qa_pairs.append(
                    {
                        "question_id": ques_id, 
                        "confidence": confidence,
                        "output_text_origin": output_text_origin,
                        "output_text_lba": output_text_lba, 
                        "gt_ans": ','.join(gt_answer),
                    }
                )
        else:
            pred_answers = answers["pred_answers"]
            for pred_answer, ques_id, gt_answer in zip(pred_answers, question_id, gt_answers):
                pred_qa_pairs.append(
                    {"question_id": ques_id, "pred_ans": pred_answer, "gt_ans": ','.join(gt_answer)}
                )
            

        return pred_qa_pairs

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Implementing accuracy computation for AOKVQA, see
        https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py#L45 for details.
        """
        # TODO add evaluation for multi-choice
        # assert False, "TODO!"

        results = json.load(open(result_file, "r"))
        results.sort(key=lambda x: x["confidence"])
            
        # --------------------------------------------------------------------------------------
        # exact match with reasoning_answer_most_common
        '''
        acc = []

        for res in results:
            # if res["gt_ans"] is None:
            #     # prepare test results for leaderboard evaluation
            #     self._save_result_leaderboard(results)
            #     return
            acc.append(res["pred_ans"] == res["gt_ans"])

        accuracy = sum(acc) / len(acc) * 100
        metrics = {"agg_metrics": accuracy, "acc": accuracy}
        '''
        # --------------------------------------------------------------------------------------
        # original vqa evaluation
        # acc = []
        acc_origin_list = []
        acc_lba_list = []
        
        correct_num = 0.0

        for i, res in enumerate(results):
            '''
            if res["gt_ans"] is None:
                # prepare test results for leaderboard evaluation
                self._save_result_leaderboard(results)
                return
            '''
            output_text_origin = res["output_text_origin"]
            output_text_lba = res["output_text_lba"]
            # pred = res["pred_ans"]
            gt_ans = res["gt_ans"].split(',')
            if i<10:
                print(f'{i:2} | output_text_origin: {output_text_origin:12s} | gt_ans: {gt_ans}')
            '''
            # num_match = sum([pred == gt for gt in gt_ans])
            # vqa_acc = min(1.0, num_match / 3.0)
            # acc.append(vqa_acc)
            '''
            num_match_origin = sum([output_text_origin == gt for gt in gt_ans])
            vqa_acc_origin = min(1.0, num_match_origin / 3.0)
            acc_origin_list.append(vqa_acc_origin)
            
            num_match_lba = sum([output_text_lba == gt for gt in gt_ans])
            vqa_acc_lba = min(1.0, num_match_lba / 3.0)
            acc_lba_list.append(vqa_acc_lba)

        # E_CR, E_IC: Error Correction raio / Error Induction ratio
        e_cr = sum([1 if acc_lba > acc_origin else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / len(acc_origin_list) * 100
        e_ic = sum([1 if acc_lba < acc_origin else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / len(acc_origin_list) * 100

        # accuracy = sum(acc) / len(acc) * 100
        correct_num = sum(acc_origin_list)
        
        correct_num_by_tau = [correct_num]
        max_num_by_tau = correct_num
        max_arg_confidence = 0.
        max_arg_confidence_percentile = 0.
        for i, res in enumerate(results):
            output_text_origin = res["output_text_origin"]
            output_text_lba = res["output_text_lba"]
            gt_ans = res["gt_ans"].split(',')
            
            original_num_match = sum([output_text_origin == gt for gt in gt_ans])
            original_vqa_acc = min(1.0, original_num_match / 3.0)
            lba_num_match = sum([output_text_lba == gt for gt in gt_ans])
            lba_vqa_acc = min(1.0, lba_num_match / 3.0)
            
            score_change = lba_vqa_acc - original_vqa_acc
            new_num = correct_num_by_tau[-1] + score_change
            if new_num > max_num_by_tau:
                max_num_by_tau = new_num
                max_arg_confidence = res["confidence"]
                max_arg_confidence_percentile = (i+1) / len(results) * 100
                
            correct_num_by_tau.append(new_num)
            
        accuracy_by_tau = [c / len(results) * 100 for c in correct_num_by_tau]
        
        import matplotlib.pyplot as plt
        plt.plot([i / len(results) * 100 for i, _ in enumerate(accuracy_by_tau)], accuracy_by_tau)
        plt.xlabel('Confidence Percentile')
        plt.ylabel('Accuracy')
        plt.xticks([0, 25, 50, 75, 100])
        plt.savefig(os.path.join(registry.get_path("output_dir"), "accuracy_by_tau.png"))
        
        metrics = {
            # "agg_metrics": accuracy, 
            # "acc": accuracy, 
            "acc_origin": correct_num / len(results) * 100,
            "max_acc_by_tau": max(accuracy_by_tau), 
            "max_arg_confidence": max_arg_confidence,
            "max_arg_confidence_percentile": max_arg_confidence_percentile,
            "E_CR": e_cr,
            "E_IC": e_ic,
        }

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)

        return metrics
