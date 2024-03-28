import os
import json
import datetime
from pprint import pprint
import argparse

from api_chatgpt import *


def main(args):
    prompt_format = open(args.prompt_path, 'r').read().strip()

    episodes = open(args.root_dir + 'Descriptions_CharacterBackground/Episode/AnotherMissOh_integrated_train_episode.json.rows', 'r', encoding='utf8').readlines() 
    print(type(episodes[0].strip()))
    episode_0 = json.loads(episodes[0].strip())
    # print(episode_0)
    # print('*' * 200)
    # episode_0 = {'Haeyoung1': ["daughter of Deogi", "daughter of Kyungsu", "niece of Jeongsuk", "ex-girlfriend of Taejin", "friend of Haeyoung2", "best friend of Heeran"], 'Deogi': ["mother of Haeyoung1", "sister-in-law with Jeongsuk", "wife of Kyungsu"], 'Kyungsu': ["father of Haeyoung1", "husband of Haeyoung1", "brother of Jeongsuk"], "Sukyung": ["sister of Dokyung", "sister of Hun", "daughter of Jiya", "superior of Haeyoung1"], 'Dokyung': ["brother of Sukyung", "brother of Hun", "son of Jiya", "ex-boyfriend of Haeyoung2"], 'Hun': ["brother of Dokyung", "brother of Sukyung", "son of Jiya", "work with Dokyung"], 'Jinsang': ["best friend of Dokyung", "lawyer"], 'Taejin': ["ex-boyfriend of Haeyoung1", "CEO"], 'Haeyoung2': ["ex-girlfriend of Dokyung"], 'Chairman': ["investor of Taejin", "in a relationship with Jiya"], 'Anna': ["girlfriend of Hun"], 'Heeran': ["best friend of Haeyoung1", "program director", "work with Dokyung"], 'Gitae': ["work with Dokyung"], 'Sangseok': ["work with Dokyung"], 'Yijoon': ["work with Dokyung"]}

    qas = json.load(open(args.root_dir + 'AnotherMissOhQA_train_set.json', 'r'))
    qas = list(filter(lambda x: x["vid"].endswith('0000'), qas))
    qas = sorted(qas, key=lambda x: x["vid"])
    # for i in range(21):
        # print(qas[i]["vid"])

    scene_f = open(os.path.join(args.root_dir, 'Descriptions_CharacterBackground/Scene/AnotherMissOh_integrated_train_scene.json.rows'), 'r', encoding='utf8') 
    for i, data in enumerate(scene_f):
        # print(scene.strip())
        scene = json.loads(data.strip())
        scene_description = scene["scene_description"]
        knowledge_graph = scene["knowledge_graph"]
        character_information = episode_0
        main_Q = qas[i]["que"]
        main_A = qas[i]["answers"][qas[0]["correct_idx"]]
        prompt = prompt_format.format(scene_description=scene_description, knowledge_graph=knowledge_graph, character_information=character_information, 
                                    main_Q=main_Q, main_A=main_A)
        print('que:', qas[i]["que"], '\n')
        print('prompt:', prompt, sep='\n')
        
        break


    if args.debug:
        response_data = {'id': 'chatcmpl-92brhs39ADx4sqOkypr7BE0F45OnA', 'object': 'chat.completion', 'created': 1710409237, 'model': 'gpt-4-0125-preview', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': "Given the answer and the details provided, let's create 10 supporting questions that logically lead to the understanding of the scene, step by step, focusing on the characters, their relationships, and the events:\n\n1. Who is Haeyoung1 in relation to Deogi and Kyungsu?\n2. What significant event does Haeyoung1 announce to Deogi and Kyungsu?\n3. How is Deogi related to Kyungsu?\n4. Can you list the roles or titles Haeyoung1 holds in relation to other characters mentioned?\n5. What action did Haeyoung1 take before delivering the news to Deogi and Kyungsu?\n6. Who were the recipients of Haeyoung1's news?\n7. What was the content of Haeyoung1's announcement?\n8. How might Deogi's role as the mother influence her reaction to Haeyoung1's announcement?\n9. Considering the relationships and roles, how could Kyungsu's position as Haeyoung1's father affect his response to the news?\n10. Why is the information about Haeyoung1's announcement significant to understanding the scene's context and the characters' reactions?\n\nThese questions guide through the characters' relationships, their roles, and the events leading up to the scene to grasp the complexity and the emotional weight of Haeyoung1's announcement, alongside providing context to Deogi's assumed responsibilities, which led to the given answer."}, 'logprobs': None, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 523, 'completion_tokens': 301, 'total_tokens': 824}, 'system_fingerprint': 'fp_31c0f205d1'}
    else:
        if args.vision:
            image_paths = get_image_path(args, qas[0])
            response_data = call_vision_api(args, prompt, image_paths)
        else:
            response_data = call_chat_api(args.model, prompt)

    print('-' * 120)
    print('content:', response_data['choices'][0]['message']['content'], sep='\n')

    pprint(response_data, width=200)

    print('-' * 120)
    print('response_data:', response_data, sep='\n')

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d_%H:%M:%S")

    if not args.debug:
        dump_data = response_data
        dump_data.update(vars(args))
        dump_data.update({'prompt': prompt})
        dump_data.update({'used_frames': image_paths})
        
        filename = args.output_dir + f'{time_str}_{args.model}.json'
        
        json.dump(response_data, open(filename, 'w'), indent=4)
        with open(filename.replace('.json', '.txt'), 'w') as f:
            f.write(response_data['choices'][0]['message']['content'])


def get_args():
    OPENAI_API_KEY = json.load(open('api_key.json', 'r'))["LBA"]

    parser = argparse.ArgumentParser(description='OpenAI ChatGPT')

    parser.add_argument('--openai_api_key', type=str, default=OPENAI_API_KEY)

    parser.add_argument('--root_dir', type=str, default="/data1/AnotherMissOh/")
    parser.add_argument('--prompt_path', type=str, default="path_to_your_prompt_path") #"prompts/subqa_240325.txt")
    parser.add_argument('--output_dir', type=str, default="output_chatgpt/")
    
    parser.add_argument('--max_tokens', type=int, default=1500) # max output token
    parser.add_argument('--temperature', type=float, default=0.)
    
    # vision
    parser.add_argument('--vision', action='store_true', default=False)
    parser.add_argument('--vision_detail', type=str, default="low", choices=["low", "high"]) 
    parser.add_argument('--max_vision_num', type=int, default=1)
    
    # debug
    parser.add_argument('--debug', action='store_true', default=False)
    
    args = parser.parse_args()

    args.model = "gpt-4-vision-preview" if args.vision else "gpt-4-turbo-preview" # "gpt-3.5-turbo" 
    
    return args


if __name__ == "__main__":
    args = get_args()
    print("args:", args)
    print("args.model:", args.model)
    main(args)
