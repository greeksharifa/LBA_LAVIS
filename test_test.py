import json
import os
import glob

from tqdm import tqdm
from PIL import Image

import torch


# ROOT_DIR = '/home/ywjang/data/'     # 3090ti
ROOT_DIR = '/data/'  # BIVI
# ROOT_DIR = '/data/dataset/'     # VTT
result = {}

splits = ['train', 'val']
for split in splits:
    print(f'Generate in {split} split...')
    with open(os.path.join(ROOT_DIR, f'VQA-Introspect/VQAIntrospect_{split}v1.0.json'), 'r') as f_in:
        json_data = json.load(f_in)
        
        image_id_list = []
        for idx, (main_question_id, value) in enumerate(json_data.items()):
            # if idx >= 3:
            #     break
            image_id_list.append(value["image_id"])
        image_id_list = list(set(image_id_list))
        print(f'{split} image_id_list:', len(image_id_list))
        