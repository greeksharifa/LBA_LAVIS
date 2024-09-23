import os, json, glob
from pprint import pprint
import re
from tqdm import tqdm
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)

args = parser.parse_args()


"""
python merge_subqa.py --model=xl
"""
# for dataset_name in ['NExT_QA', 'STAR', 'TVQA', 'VLEP', 'DramaQA', 'IntentQA', 'EgoSchema']:
for dataset_name in ['VQA_Introspect', 'AOKVQA', 'OKVQA']:
    beam_sub_qas_path = f'/data/{dataset_name}/sub_qas_val_{args.model}_fewshot_vqaintrospect.json'
    greedy_sub_qas_path = f'/data/{dataset_name}/sub_qas_val_{args.model}_beam_and_greedy.json'
    beam_sub_qas_data = json.load(open(beam_sub_qas_path))
    greedy_sub_qas_data = json.load(open(greedy_sub_qas_path))
    print(dataset_name, "beam\t", next(iter(beam_sub_qas_data.items())))
    print(dataset_name, "greedy\t", next(iter(greedy_sub_qas_data.items())))
    
    results = {}
    # import pdb; pdb.set_trace()
    for k, v in tqdm(beam_sub_qas_data.items()):
        # Use a dictionary to keep track of unique items
        unique_dict = OrderedDict()

        # Iterate through the list and add items to the dictionary
        for item in v:
            key = item[0].capitalize().rstrip('?') + '?'
            value = item[1].rstrip('.')
            unique_dict[key] = value

        greedy_dict = greedy_sub_qas_data[k][1:]
        # Convert the dictionary values back to a list
        for gk, gv in greedy_dict:
            if len(unique_dict) >= 5:
                break
            key = gk.capitalize().rstrip('?') + '?'
            value = gv.rstrip('.')
            if key in unique_dict and unique_dict[key] == value:
                pass
            else:
                unique_dict[key] = value
                
        
        result = []
        for uk, uv in unique_dict.items():
            result.append([uk, uv])

        while len(result) < 5:
            result.append(result[-1])
        
        results[k] = result
        
            
        # print(result)
        # break
    json.dump(results, open(f'/data/{dataset_name}/sub_qas_val_{args.model}_fewshot_vqaintrospect_unique.json', 'w'), indent=4)
