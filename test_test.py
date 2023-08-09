import json
import os
import glob

from tqdm import tqdm
from PIL import Image

import torch

from collections import Counter

ROOT_DIR = 'data/'     # local
# ROOT_DIR = '/home/ywjang/data/'     # 3090ti
# ROOT_DIR = '/data/'  # BIVI
# ROOT_DIR = '/data/dataset/'     # VTT
result = {}

splits = ['train', 'val']
for split in splits:
    counter = Counter()
    print(f'Generate in {split} split...')
    
    with open(os.path.join(ROOT_DIR, f'VQA-Introspect/VQAIntrospect_{split}v1.0.json'), 'r') as f_in:
        json_data = json.load(f_in)
        
        for idx, (main_question_id, value) in enumerate(json_data.items()):
            # if idx >= 3:
            #     break
            sub_qa_set = set()
            introspects = value['introspect']
            for introspect in introspects:
                sub_qa_list = introspect['sub_qa']
                for sub_qa in sub_qa_list:
                    sub_qa_set.add(sub_qa['sub_question'])
            counter[len(sub_qa_set)] += 1
            # if len(sub_qa_set) == 0:
            #     print(f'No sub question in {main_question_id}')
    print('counter: ', sorted(list(counter.items())))
    