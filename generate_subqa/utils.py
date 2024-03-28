import json
import os
import glob
import numpy as np


def get_scripts(args):
    speech_path = os.path.join(args.root_dir, args.speech_path, 'DramaCap_train_script.json')
    script_list = json.load(open(speech_path, 'r'))
    scripts = {}
    for sample in script_list:
        try:
            vid = sample["vid"]
            # if not vid.endswith('0000'):    continue
            description = sample["desc"]
            subs = ''
            if sample["subtitle"] == ".":
                continue
            else:
                for sub in sample["subtitle"]["contained_subs"]:
                    if subs != '':
                        subs += '\n'
                    subs += f'{sub["speaker"]}: {sub["utter"].strip()}'
                
            scripts[vid] = {
                "subs:": subs,
                "description": description,
            }
        except:
            from pprint import pprint
            pprint(sample, width=200)
    
    return scripts


def get_image_path(args, sample):
    # shot이면 그 가운데 frame 1장 선택, scene이면 그 가운데 shot 1개 선택
    # 단, 현재는 scene에 대해서만 sub_qa를 만들 계획이므로 애초에 qa(sample)에는 scene만 존재함
    # TODO: scene이면 shot별로 하나씩 선택하도록 수정
    vid = sample["vid"]
    
    if vid.endswith('0000'):
        scene_dir_path = os.path.join(args.root_dir, f"AnotherMissOh_images/{vid.replace('_', '/')}")[:-4] # ex. /data1/AnotherMissOh/AnotherMissOh_images/AnotherMissOh01/001/0078
        dir_paths = sorted(glob.glob(os.path.join(scene_dir_path, '*/')))
        # print('dir_path: len =', len(dir_paths), '\tex)', dir_paths[0])
        
        if args.max_vision_num < len(dir_paths):
            idxs = np.linspace(-1, len(dir_paths), args.max_vision_num+2, dtype=int)
            idxs = idxs[1:-1]
            dir_paths = [dir_paths[idx] for idx in idxs]

        # print('dir_path: len =', len(dir_paths), dir_paths)
        # shot_contained = sample["shot_contained"]
    else:
        dir_paths = [os.path.join(args.root_dir, f"AnotherMissOh_images/{vid.replace('_', '/')}/")]
        
        
    image_paths = []
    for dir_path in dir_paths:
        images = glob.glob(dir_path + '*.jpg')
        image_paths.append(sorted(images)[len(images) // 2]) # shot 중 가운데 frame만 선택
    print('image_paths:', image_paths)
    # assert False

    return image_paths
