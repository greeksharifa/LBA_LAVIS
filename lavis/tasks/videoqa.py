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

from collections import OrderedDict

from lavis.tasks.vqa import VQATask


@registry.register_task("videoqa")
class VideoQATask(VQATask):
    def valid_step(self, model, samples):
        # print('|', end='')
        return super().valid_step(model, samples)
        return self.valid_step_lba(model, samples, "answer")
        
    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        print("_report_metrics: result_file:", result_file, "split:", split)
        return super()._report_metrics(result_file, split)
        return self._report_metrics_lba(result_file, split, vqa_acc=False, use_vqa_tool=False)

