import json
import os
import glob
import numpy as np


def get_info_from_vid(vid):

    episode_id = vid[13:15]
    scene_id = vid[16:19]
    shot_id = vid[20:]

    return (episode_id, scene_id, shot_id)


def load_and_merge_jsons(path_format, splits=['train', 'val', 'test']):
    result = []
    for split in splits:
        path = path_format.format(split=split)
        data = json.load(open(path, 'r'))
        # if "episode" in data:
        #     episode_id = data["episode"]
        #     result[episode_id] = data
        # else:
        #     scene_id = data["scene_id"]
        #     result[scene_id] = data
        result.append(data)
        
    return result

def json_rows2json(path):
    result = {}
    lines = open(path, 'r', encoding='utf8').readlines()
    for i, line in enumerate(lines):
        data = json.loads(line.strip())
        if "episode" in data:
            episode_id = data["episode"]
            result[episode_id] = data
        else:
            scene_id = data["scene_id"]
            result[scene_id] = data
        # print('one line processed')
        if i >= 3:
            break
    return result
    
    
    result = {}
    with open(path, 'r') as file:
        for line in file:
            print('one line processed')
            # 각 줄을 JSON으로 파싱
            data = json.loads(line)
            
            # 'episode' 값을 키로 사용하여 결과 딕셔너리에 추가
            episode = data['episode']
            result[episode] = data

    return result

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
                "subs": subs,
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
