"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
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
                "datasets": {},
                "model": {},
            }),
            self._build_opt_list(self.args.options)
        )

        # runner_config = self.build_runner_config(config)
        # model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config, user_config.datasets)
        
        # LBA TODO: dataset_config랑 user_config 순서를 바꿔야 함

        self.config = OmegaConf.merge(config, dataset_config, user_config)
        # self.config = OmegaConf.merge(
        #     runner_config, model_config, dataset_config, user_config
        # )
        
    @staticmethod
    def build_dataset_config(config, user_config):
        dataset_name = config.datasets.get("dataset_name", None)
        dataset_name = user_config.get("dataset_name", dataset_name)
        if dataset_name is None:
            raise KeyError(
                "Expecting 'dataset_name' as the root key for dataset configuration."
            )
        
        dataset_config = OmegaConf.merge(
            OmegaConf.load(f'dataset/configs/{dataset_name}.yaml'),
            user_config
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

    @property
    def runner_cfg(self):
        return self.config.runner
    # @property
    # def run_cfg(self):
    #     return self.config.run

    @property
    def datasets_cfg(self):
        return self.config.datasets

    # @property
    # def model_cfg(self):
    #     return self.config.model

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.runner))

        logging.info("\n======  Dataset Attributes  ======")
        datasets = self.config.datasets

        for dataset in datasets:
            if dataset in self.config.datasets:
                logging.info(f"\n======== {dataset} =======")
                dataset_config = self.config.datasets[dataset]
                logging.info(self._convert_node_to_json(dataset_config))
            else:
                logging.warning(f"No dataset named '{dataset}' in config. Skipping")

        logging.info(f"\n======  Model Attributes  ======")
        # logging.info(self._convert_node_to_json(self.config.model))

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)

