import os, json, glob
from pprint import pprint
import re
from tqdm import tqdm
from collections import OrderedDict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--N', type=str, default="")

args = parser.parse_args()


"""
python merge_subqa.py --model=xl [--N=10] [--dataset="PathVQA,ArtVQA,SLAKE"]
"""
# dataset_names = ['NExT_QA', 'STAR', 'TVQA', 'VLEP', 'DramaQA', 'IntentQA', 'EgoSchema']
# dataset_names = ['VQA_Introspect', 'AOKVQA', 'OKVQA']
dataset_names = ['PathVQA', 'ArtVQA', 'SLAKE']

if args.dataset:
    dataset_names = args.dataset.split(',')
    
if args.N:
    N = int(args.N)
    N_tag = f"_N{N}"
else:
    N = 5
    N_tag = ""
    
for dataset_name in dataset_names:
    beam_sub_qas_path = f'/data/{dataset_name}/sub_qas_val_{args.model}_fewshot_vqaintrospect.json'
    greedy_sub_qas_path = f'/data/{dataset_name}/sub_qas_val_{args.model}_beam_and_greedy{N_tag}.json'
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
            if len(unique_dict) >= N:
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

        while len(result) < N:
            result.append(result[-1])
        
        results[k] = result
        
            
        # print(result)
        # break
    out_path = f'/data/{dataset_name}/sub_qas_val_{args.model}_fewshot_vqaintrospect_unique{N_tag}.json'
    json.dump(results, open(out_path, 'w'), indent=4)
    
    # verify
    data = json.load(open(out_path))
    for k, v in data.items():
        if len(v) != N:
            print("Error: len(v) != N")
            print(k, len(v))
            print(v)
            break
