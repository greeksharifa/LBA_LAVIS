"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

from lavis.common.dist_utils import main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("captioning")
class CaptionTask(BaseTask):
    def __init__(self, num_beams, max_len, min_len, evaluate, report_metric=True):
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

        self.report_metric = report_metric

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len = run_cfg.min_len
        evaluate = run_cfg.evaluate

        report_metric = run_cfg.get("report_metric", True)

        return cls(
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            evaluate=evaluate,
            report_metric=report_metric,
        )

    def valid_step(self, model, samples):   # samples(batch)에 대해 모델의 inference 결과를 반환
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        img_ids = samples["image_id"]
        main_question_ids = samples["main_question_id"]
        for caption, img_id, main_question_id in zip(captions, img_ids, main_question_ids):
            results.append({"caption": caption, "image_id": int(img_id), "main_question_id": int(main_question_id)})

        return results

    def after_evaluation(self, val_result, split_name, epoch, **kwargs):
        eval_result_file = self.save_result(
            result=val_result,
            result_dir=registry.get_path("result_dir"),
            filename="{}_epoch{}".format(split_name, epoch),
            remove_duplicate="image_id",
        )

        if self.report_metric:
            metrics = self._report_metrics(
                eval_result_file=eval_result_file, split_name=split_name
            )
        else:
            metrics = {"agg_metrics": 0.0}

        return metrics

    @main_process
    def _report_metrics(self, eval_result_file, split_name):

        # TODO better way to define this
        coco_gt_root = os.path.join(registry.get_path("cache_root"), "coco_gt")
        coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)

        agg_metrics = coco_val.eval["CIDEr"] + coco_val.eval["Bleu_4"]
        log_stats = {split_name: {k: v for k, v in coco_val.eval.items()}}

        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(log_stats) + "\n")

        coco_res = {k: v for k, v in coco_val.eval.items()}
        coco_res["agg_metrics"] = agg_metrics

        return coco_res


@registry.register_task("vqa_introspect_captioning")
class VQAIntrospectCaptionTask(CaptionTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @main_process
    def _report_metrics(self, eval_result_file, split_name):
        gt_file = os.path.join(registry.get_path("cache_root"), "VQA-Introspect/VQAIntrospect_valv1.0.json")
        raw_gt_data = json.load(open(gt_file))              # gt data
        """
        >>> raw_gt_data['565248001']
        {'reasoning_answer_most_common': 'yes', 'introspect': [
                {'sub_qa': [{'sub_question': 'are the woman on the horses near to the sea?', 'sub_answer': 'yea'}], 'pred_q_type': 'perception'},
                {'sub_qa': [{'sub_question': 'is there sand and whitewater?', 'sub_answer': 'yes'}], 'pred_q_type': 'reasoning'},
                {'sub_qa': [{'sub_question': 'are the woman on the horses near to the sea?', 'sub_answer': 'yea'}], 'pred_q_type': 'perception'},
                {'sub_qa': [{'sub_question': 'is there sand and whitewater?', 'sub_answer': 'yes'}], 'pred_q_type': 'reasoning'}
            ],
            'reasoning_question': 'are they at the beach?', 'image_id': 565248}
        """
        gt_data = dict()
        for main_Q_id, value in raw_gt_data.items():
            if main_Q_id not in gt_data:
                gt_data[main_Q_id] = []
            for introspect in value['introspect']:
                for sub_qa in introspect['sub_qa']:
                    gt_data[main_Q_id].append(sub_qa['sub_question'])
        for main_Q_id, value in gt_data.items():
            gt_data[main_Q_id] = list(set(value))
        """
        gt_data: dict. key=main_Q_id, value=list of sub_Q
        gt_data['565248001']: [
            'are the woman on the horses near to the sea?',
            'is there sand and whitewater?'
        ]
        """
        import warnings;    warnings.filterwarnings("ignore")
        blue_scores = []
        res_data = json.load(open(eval_result_file))    # result data
        # res example: {'caption': 'stabilityout land guerre...', 'image_id': 565248, 'main_question_id': 565248001}
        for res in res_data:
            reference = [ref_sentence.split() for ref_sentence in gt_data[str(res['main_question_id'])]]
            candidate = res['caption'].split()
            blue_score = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
            # print('blue_score:', blue_score)
            blue_scores.append(blue_score)
        
        return {"agg_metrics": np.mean(blue_scores)}


# TODO better structure for this.
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from torchvision.datasets.utils import download_url


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {
        "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
        "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval
