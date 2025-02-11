import argparse
import random
import itertools
import logging

from typing import Any, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from util.colors import Colors


def setup_seeds(config):
    seed = config.runner_cfg.seed # + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description='LBA method')
    parser.add_argument("--cfg-path", default='config/default.yaml', help="path to configuration file.")

    # for open-ended, evaluate by GPT-3.5
    # parser.add_argument('--eval_chatgpt', action='store_true', help='for open-ended, evaluate by GPT-3.5. only available in visualize mode')

    parser.add_argument('--ignore_error', action='store_true', help='ignore error')
    
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    # check consistency
    # pass

    return args


def transpose_list(lists: List[List[Any]]) -> List[List[Any]]:
    # [N, bsz] -> [bsz, N]
    result = list(zip(*lists))
    return [list(sub_list) for sub_list in result]



class IndexSampler:
    def __init__(self, N, M, K):
        self.N = N
        self.M = M
        self.K = K
        self._indices = []

        for i in range(min(N, K)):
            self._indices.append(sorted([(i+j) % N for j in range(M)]))
        
        if N < K:
            nCrs = itertools.combinations(list(range(N)), M)
            nCrs = list(nCrs)
            random.shuffle(nCrs)

            for nCr in nCrs:
                nCr = list(nCr)
                # print(nCr, self._indices)
                if len(self._indices) < K and nCr not in self._indices:
                    self._indices.append(sorted(nCr))
        logger = logging.getLogger("C2R")
        logger.info(f"Sampled indices: {self._indices}")

    @property
    def indices(self):
        """
        return list of list: [K, M]
        """
        return self._indices
    
    @indices.setter
    def indices(self, new_indices):
        self._indices = new_indices
