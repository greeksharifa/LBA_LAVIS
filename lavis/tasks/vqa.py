"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
from lavis.common.logger import setup_logger


import lavis.common.dist_utils as dist_utils
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
from lavis.tasks.base_task import BaseTask
from colors import Colors, print_sample

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
    ):
        super().__init__()
        
        setup_logger()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

        self.evaluate = evaluate
        self.inference_method = inference_method
        self.num_ans_candidates = num_ans_candidates
        self.prompt = prompt

        self.answer_list = None

        self.ques_files = dict()
        self.anno_files = dict()

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
        logging.info(Colors.BRIGHT_RED + "in setup_task(), prompt: " + prompt + Colors.RESET)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            num_ans_candidates=num_ans_candidates,
            inference_method=inference_method,
            prompt=prompt,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)

        # get question file, annotation file and anwser list in COCO format
        for dataset in datasets.values():
            for split in dataset:
                if (
                    hasattr(dataset[split], "coco_fmt_qust_file")
                    and dataset[split].coco_fmt_qust_file is not None
                ):
                    self.ques_files[split] = dataset[split].coco_fmt_qust_file
                    self.anno_files[split] = dataset[split].coco_fmt_anno_file

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
            ques_id = int(ques_id.item())
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
            ques_id = int(ques_id.item())
            pred_qa_pairs.append({"question_id": ques_id, "pred_ans": answer, "gt_ans": gt_answer})

        return pred_qa_pairs
        
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
        

@registry.register_task("aok_vqa")
class AOKVQATask(VQATask):
    _cnt = 0
    def valid_step(self, model, samples):
        # if AOKVQATask._cnt == 0:
        #     AOKVQATask._cnt += 1
        #     print_sample(samples, msg="AOK-VQA samples:", color=Colors.BRIGHT_YELLOW)
        answers = model.predict_answers(
            samples=samples,
            answer_list=self.answer_list,
            inference_method=self.inference_method,
            num_beams=self.num_beams,
            max_len=self.max_len,
            min_len=self.min_len,
            num_ans_candidates=self.num_ans_candidates,
        )
        pred_choice_idxs = model.predict_class(
            samples=samples,
            candidates=samples["choices"],
            n_segments=1
        ).cpu().numpy()

        pred_qa_pairs = []

        question_id = samples["question_id"]
        gt_answers = samples["direct_answers"]
        correct_choice_idxs = samples["correct_choice_idx"]
        choices = samples["choices"]
        # image_ids = samples["image_id"]

        for pred_answer, ques_id, gt_answer, pred_choice_idx_ndarray, correct_choice_idx, choice in zip(answers, question_id, gt_answers, pred_choice_idxs, correct_choice_idxs, choices):
            pred_choice_idx = int(pred_choice_idx_ndarray[0])
            pred_qa_pairs.append(
                {#"image_id": image_id,
                 "question_id": ques_id, "pred_ans": pred_answer, "gt_ans": gt_answer,
                 "pred_choice_idx": pred_choice_idx, "correct_choice_idx": correct_choice_idx,
                 "predicted_class": choice[pred_choice_idx], "choices": choice,
                 }
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
        mc_acc = []

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
            
            pred_choice_idx = res["pred_choice_idx"]
            correct_choice_idx = res["correct_choice_idx"]
            mc_acc.append(1 if pred_choice_idx == correct_choice_idx else 0)

        accuracy = sum(acc) / len(acc) * 100
        mc_accuracy = sum(mc_acc) / len(mc_acc) * 100
        metrics = {"agg_metrics": accuracy, "mc_acc": mc_accuracy,  # "acc": accuracy,
                   "da_acc": accuracy}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(metrics) + "\n")

        logging.info(metrics)
        print('metrics: ', metrics)

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
                "multiple_choice": res["predicted_class"],  # "",
            }

        result_file = registry.get_path("result_dir") + "_leaderboard.json"

        with open(result_file, "w") as f:
            json.dump(result_leaderboard, f)

        logging.info(f"Saved results for leaderboard evaluation at {result_file}")


@registry.register_task("vqa_introspect_test_task")
class VQAIntrospectTestTask(VQATask):
        _cnt = 0
        
        
        def valid_step(self, model, samples):
            if VQAIntrospectTestTask._cnt == 0:
                VQAIntrospectTestTask._cnt += 1
                print_sample(samples, msg=f"prompt: {self.prompt}, VQA-Introspect Test samples:", color=Colors.BRIGHT_YELLOW)
            result = model.predict_answers(
                samples=samples,
                answer_list=self.answer_list,
                inference_method=self.inference_method,
                num_beams=self.num_beams,
                max_len=self.max_len,
                min_len=self.min_len,
                num_ans_candidates=self.num_ans_candidates,
                prompt="Q: {} A:",
            )
            
            if type(result) == tuple:
                pred_answers, sub_q_lists, sub_a_lists = result
            else:
                pred_answers, sub_q_lists, sub_a_lists = result, None, None
            
            pred_qa_pairs = []
            
            image_ids = samples["image_id"]
            questions = samples["text_input"]
            question_ids = samples["question_id"]
            gt_answers = samples["answer"]
            # image_ids = samples["image_id"]
            
            if sub_q_lists is not None:
                for image_id, pred_answer, question, ques_id, gt_answer, sub_q_list, sub_a_list in zip(image_ids, pred_answers, questions, question_ids, gt_answers, sub_q_lists, sub_a_lists):
                    pred_qa_pairs.append(
                        {
                            "image_id": image_id,
                            "question_id": ques_id,
                            "question": question,
                            "pred_ans": pred_answer,
                            "gt_ans": gt_answer,
                            "sub_q_list": sub_q_list,
                            "sub_a_list": sub_a_list,
                        }
                    )
            else:
                for image_id, pred_answer, question, ques_id, gt_answer in zip(image_ids, pred_answers, questions, question_ids, gt_answers):
                    pred_qa_pairs.append(
                        {
                            "image_id": image_id,
                            "question_id": ques_id,
                            "question": question,
                            "pred_ans": pred_answer,
                            "gt_ans": gt_answer,
                        }
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
            # mc_acc = []
            
            for res in results:
                if res["gt_ans"] is None:
                    # prepare test results for leaderboard evaluation
                    self._save_result_leaderboard(results)
                    return
                
                pred = res["pred_ans"]
                gt_ans = res["gt_ans"]
                
                vqa_acc = 1.0 if pred == gt_ans else 0.0
                
                acc.append(vqa_acc)
                
            accuracy = sum(acc) / len(acc) * 100
            metrics = {"agg_metrics": accuracy, "da_acc": accuracy},  # "acc": accuracy,
            
            with open(
                    os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
            ) as f:
                f.write(json.dumps(metrics) + "\n")
            
            logging.info(metrics)
            print('metrics: ', metrics)
            
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


@registry.register_task("vqa_introspect")
class VQAIntrospectTask(VQATask):
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

        # TODO: gt_answerws from sample['what']?
        question_id = samples["question_id"]
        gt_answers = samples["sub_answer"]
        # print('in valid_step() in class VQAIntrospectTask')
        # print('question_id: ', question_id)
        # print('gt_answers: ', gt_answers)

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
        # TODO everything of _report_metrics() in VQAIntrospectTask class

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
