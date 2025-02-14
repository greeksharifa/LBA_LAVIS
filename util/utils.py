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


def create_answer_mapping(num=26):
    # e.g.
    # answer_mapping = {
    #     0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i", 9: "j",
    #     '0': "a", '1': "b", '2': "c", '3': "d", '4': "e", '5': "f", '6': "g", '7': "h", '8': "i", '9': "j",
    #     'A': "a", 'B': "b", 'C': "c", 'D': "d", 'E': "e", 'F': "f", 'G': "g", 'H': "h", 'I': "i", 'J': "j",
    # }
    answer_mapping = {}
    
    # 알파벳 소문자 리스트 생성 (a부터 j까지)
    lower_letters = [chr(ord('a') + i) for i in range(num)]
    
    # 정수 숫자(0-9)에 대한 매핑
    for i in range(num):
        answer_mapping[i] = lower_letters[i]
    
    # 문자열 숫자('0'-'9')에 대한 매핑
    for i in range(num):
        answer_mapping[str(i)] = lower_letters[i]
    
    # 대문자 알파벳('A'-'J')에 대한 매핑
    for i in range(num):
        answer_mapping[chr(ord('A') + i)] = lower_letters[i]
    
    return answer_mapping


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
