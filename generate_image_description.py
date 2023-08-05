import json
import os
import glob

from tqdm import tqdm
from PIL import Image

import torch

import warnings
warnings.filterwarnings(action="ignore")

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open("docs/_static/Confusing-Pictures.jpg").convert("RGB")

from lavis.models import load_model_and_preprocess
# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

print('model.prompt:', model.prompt)
print(model.generate({"image": image, "prompt": "Write a detailed description."}))

ROOT_DIR = '/home/ywjang/data/'     # 3090ti
# ROOT_DIR = '/data/'     # BIVI
# ROOT_DIR = '/data/dataset/'     # VTT
result = {}

# splits = ['train', 'val']
# splits = ['train']
splits = ['val']
for split in splits:
    print(f'Generate in {split} split...')
    f_out = open(os.path.join(ROOT_DIR, f'VQA-Introspect/instructBLIP_Description_{split}v1.0.jsonl'), 'a')
    with open(os.path.join(ROOT_DIR, f'VQA-Introspect/VQAIntrospect_{split}v1.0.json'), 'r') as f_in:
        json_data = json.load(f_in)
        
        image_id_list = []
        for main_question_id, value in json_data.items():
            image_id_list.append(value["image_id"])
        image_id_list = list(set(image_id_list))
        print(f'{split} image_id_list:', len(image_id_list))
        
        not_exist_image_id_list = []
        for idx, image_id in enumerate(tqdm(image_id_list)):
            if idx >= 10:
                break
            image_path = os.path.join(ROOT_DIR, f"coco/images/{split}2014/COCO_{split}2014_{image_id:012}.jpg")
            
            if os.path.exists(image_path) is False:
                not_exist_image_id_list.append(image_path.split('/')[-1])
                print('not exist image_path:', image_path)
                with open(os.path.join(ROOT_DIR, f'VQA-Introspect/not_exist_image_id_list_{split}v1.0.txt'), 'a') as f_out2:
                    f_out2.write('/'.join(image_path.split('/')[-2:]) + '\n')
                continue
                
            raw_image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            print('image_id:', image_id, '\t', "image_path:", image_path)
            description = model.generate({"image": image, "prompt": "Write a detailed description."})
            print(description)
            new_data = {
                "image_path": '/'.join(image_path.split('/')[-2:]),     # image_path.split('/')[-1],
                "description": description,
            }
            result[image_id] = new_data
            with open(os.path.join(ROOT_DIR, f'VQA-Introspect/instructBLIP_Description_{split}v1.0.jsonl'), 'a') as f_out:
                f_out.write(str(new_data) + '\n')
            
    print(f'split {split} Done.')
    print('not_exist_image_id_list:', not_exist_image_id_list)
    print('len(not_exist_image_id_list):', len(not_exist_image_id_list))
    print('out file:', os.path.join(ROOT_DIR, f'VQA-Introspect/instructBLIP_Description_{split}v1.0.jsonl'))
    print('not exist image file:', os.path.join(ROOT_DIR, f'VQA-Introspect/not_exist_image_id_list_{split}v1.0.txt'))
    