"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
import copy
from typing import Dict

from omegaconf import OmegaConf


class Config:
    def __init__(self, args):
        self.config = {}

        self.args = args

        config = OmegaConf.load(self.args.cfg_path)
        user_config = OmegaConf.merge(
            OmegaConf.create({
                "runner": {},
                "dataset": {},
                "model": {},
            }),
            self._build_opt_list(self.args.options)
        )        

        # runner_config = self.build_runner_config(config)
        model_config = self.build_model_config(config, user_config.model)
        dataset_config = self.build_dataset_config(config, user_config.dataset)
        

        self.config = OmegaConf.merge(config, model_config, dataset_config, user_config)
        if self.config.model.get("cache_dir", None) is None:
            self.config.model.cache_dir = os.path.join(self.config.model.HF_HOME, self.config.model.model_name.split("/")[0].split("-")[0])
            

        # check consistency
        # if self.args.visualize_only and self.runner_cfg.output_dir == "output":
        #     raise ValueError("visualize_only is True, but output_dir is default directory ('output'). Please specify the output directory.")
        
        # if self.runner_cfg.get("early_stop_K", False):
        #     logger = logging.getLogger("C2R")
        #     logger.warning("batch size is set to 1 when early_stop_K is True")
        #     self.model_cfg.batch_size = 1
            # assert self.model_cfg.batch_size == 1, "batch size must be 1 when early_stop_K is True"

        
        if self.runner_cfg.get("CoT", False):
            self.model_cfg.max_new_tokens_maina = 4096
        
    @staticmethod
    def build_model_config(config, user_model_config):
        model_name = config.model.get("model_name", None)
        model_name = user_model_config.get("model_name", model_name)
        if model_name is None:
            raise KeyError(
                "Expecting 'model_name' as the root key for model configuration."
            )
        
        model_config = OmegaConf.merge(
            OmegaConf.load(f'config/models/{model_name}.yaml'),
            user_model_config
        )
        return model_config
        

    @staticmethod
    def build_dataset_config(config, user_dataset_config):
        dataset_name = config.dataset.get("dataset_name", None)
        dataset_name = user_dataset_config.get("dataset_name", dataset_name)
        if dataset_name is None:
            raise KeyError(
                "Expecting 'dataset_name' as the root key for dataset configuration."
            )
        
        dataset_config = OmegaConf.merge(
            OmegaConf.load(f'config/datasets/{dataset_name}.yaml'),
            user_dataset_config
        )
        return dataset_config

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def get_config(self):
        return self.config

    def for_stage(self, mode):
        """Return an independent config whose runner mode targets one stage."""
        stage_config = self.__class__.__new__(self.__class__)
        stage_config.__dict__ = {
            key: value for key, value in self.__dict__.items() if key != "config"
        }
        stage_config.config = copy.deepcopy(self.config)
        stage_config.config.runner.mode = mode
        return stage_config

    @property
    def runner_cfg(self):
        return self.config.runner
    
    @property
    def dataset_cfg(self):
        return self.config.dataset

    @property
    def model_cfg(self):
        return self.config.model
    
    def set_logger(self, logger):
        self.logger = logger

    def pretty_print(self):
        pretty_print_str = "\n======  Running Parameters  ======\n" + self._convert_node_to_json(self.config.runner)
        pretty_print_str += "\n======  Dataset Attributes  ======\n" + self._convert_node_to_json(self.config.dataset)
        pretty_print_str += "\n======   Model Attributes   ======\n" + self._convert_node_to_json(self.config.model)
        self.logger.info(pretty_print_str)

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)
