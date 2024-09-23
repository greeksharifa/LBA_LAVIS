import os, shutil
import argparse
import json
import nltk
from tqdm import tqdm
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration

from models.model import VideoBlip2ForConditionalGeneration
from dataset.base_dataset import load_dataset
from configs.config import Config
from main_multi_subqa import parse_args, setup_seeds
from dataset.VQA_Introspect import VQAIntrospectDataset


    
def get_input(data_type, processor, device, vision_batch, text_inputs):
    if data_type == "images":
        inputs = processor(text=text_inputs, images=vision_batch, return_tensors="pt", padding=True)
    else:
        inputs = processor(text=text_inputs, return_tensors="pt", padding=True)
        pixel_values = []
        for video in vision_batch: # video: [n_frms, 640, 480]
            pixel_values.append(processor(images=video, return_tensors="pt", padding=True)['pixel_values'])  # [n_frms, 3, 224, 224]
        inputs["pixel_values"] = torch.stack(pixel_values, dim=0)#.to(device)
        # print("input_ids:", inputs["input_ids"].shape, inputs["input_ids"].device, "\tpixel_values:", inputs["pixel_values"].shape, inputs["pixel_values"].device)
        
    return inputs.to(device)


"""
pre-generate sub-qa pairs for each question in the dataset
usage:
CUDA_VISIBLE_DEVICES=4 python generate_subqa.py --options runner.sub_mode="beam_and_greedy" datasets.dataset_name="DramaQA" runner.batch_size=12 runner.num_sub_qa_generate=5 runner.recomposer_name="Salesforce/blip2-flan-t5-xl"
CUDA_VISIBLE_DEVICES=4 python generate_subqa.py --options runner.sub_mode="fewshot_vqaintrospect" datasets.dataset_name="NExTQA" runner.batch_size=12 runner.num_sub_qa_generate=5 runner.recomposer_name="Salesforce/blip2-flan-t5-xl"
CUDA_VISIBLE_DEVICES=4 python generate_subqa.py --options runner.sub_mode="Ktype" datasets.dataset_name="DramaQA" runner.batch_size=2 datasets.num_data=5 runner.num_sub_qa_generate=6 runner.recomposer_name="Salesforce/blip2-flan-t5-xl"
"""
def main():
    N_SUPPLE = 1
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)
    model_name = cfg.runner_cfg.recomposer_name
    processor_name = model_name
    # processor_name = "Salesforce/instructblip-flan-t5-xl"
    # model_name = processor_name
    cache_dir = os.path.join("/model/", model_name.split("/")[0])
    device = "cuda"
    
    if cfg.datasets_cfg.data_type == "images": # dataset_name in ["VQA_Introspect", "AOKVQA", "OKVQA"]:
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(device)#, device_map="auto")
        processor = Blip2Processor.from_pretrained(processor_name, cache_dir=cache_dir)
    else: # "videos"
        model = VideoBlip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(device)#, device_map="auto")
        processor = Blip2Processor.from_pretrained(processor_name, cache_dir=cache_dir)
    # model = InstructBlipVideoForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(device)#, device_map="auto")
    # processor = InstructBlipVideoProcessor.from_pretrained(processor_name, cache_dir=cache_dir)
    
    dataset = load_dataset(cfg.datasets_cfg, n_supple=N_SUPPLE)
    dataloader = DataLoader(dataset, batch_size=cfg.runner_cfg.batch_size,
                            shuffle=False, collate_fn=dataset.collater)

    prompt_Ktype = {
        # What is a missing information about ...
        "Ktype_0": "What or who is {entity}?", # Identity
        "Ktype_1": "What is the inclusion relationships of {entity}?", # Class
        "Ktype_2": "What is the properties or feature of {entity}?", # Attributes
        "Ktype_3": "What is the number of {entity}?", # Quantities
        "Ktype_4": "What is the spatial relations among {entity}s?", # Spatial
        "Ktype_5": "What is the detailed information of {entity}?", # Contents, 원래는 K7
    }
    vqa_introspect_dataset = VQAIntrospectDataset(None, None, '/data/coco/images/', 
                                                  ['/data/VQA_Introspect/VQAIntrospect_valv1.0.json', '/data/VQA/v2/v2_mscoco_val2014_annotations.json'])

    prompt_subqa_vqaintrospect = []
    idx_example = 0
    for i in range(cfg.runner_cfg.num_sub_qa_generate):
        prompt = ""
        fewshot_num = cfg.runner_cfg.get("fewshot_num", 5)
        for _ in range(fewshot_num):
            while True:
                example = vqa_introspect_dataset[idx_example]
                idx_example += 1
                if len(example["gt_sub_qas"]) > 0:
                    break
            
            main_q = example["text_input"].capitalize()
            sub_q = example["gt_sub_qas"][0][0].capitalize()
            prompt += f"Reasoning Question: {main_q} Perception Question: {sub_q}\n"
        
        prompt += "Reasoning Question: {main_question}? Perception Question:"
        prompt_subqa_vqaintrospect.append(prompt)
    pprint(prompt_subqa_vqaintrospect, width=300)
    
    temp_dir = f"temp/files/{cfg.datasets_cfg.dataset_name}/{cfg.runner_cfg.sub_mode}/"
    os.makedirs(temp_dir, exist_ok=True)
    
    results = {}
    for data_iter_step, batch in enumerate(tqdm(dataloader)):
        if os.path.exists(os.path.join(temp_dir, f"{cfg.datasets_cfg.dataset_name}_{data_iter_step}.json")):
            batch_result = json.load(open(os.path.join(temp_dir, f"{cfg.datasets_cfg.dataset_name}_{data_iter_step}.json"), "r"))
            results.update(batch_result)
            continue
        
        question_ids = batch["question_id"]
        bsz = len(batch["question_id"])
        
        batch_result = {}
        for i in range(cfg.runner_cfg.num_sub_qa_generate):
            if cfg.runner_cfg.sub_mode == "beam_and_greedy":   # Generate Sub-Questions by Huggingface model
                prompt = "Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {main_question}? Perception Question:"
                text_inputs = [prompt.format(main_question=main_question.rstrip('?')) for main_question in batch["text_input"]]
                
                inputs = get_input(cfg.datasets_cfg.data_type, processor, device, batch["vision"], text_inputs)
                
                generation_params = {
                    "do_sample": True,
                    "min_new_tokens": 1,
                    "max_new_tokens": 100,
                }
                beam_search = i==0
                if beam_search:
                    generation_params["num_beams"] = 5
                    generation_params["length_penalty"] = -1
                else:
                    generation_params["top_p"] = 0.8
                
                outputs = model.generate(**inputs, **generation_params)
                sub_questions = processor.batch_decode(outputs, skip_special_tokens=True)

            elif cfg.runner_cfg.sub_mode == "fewshot_vqaintrospect":
                text_inputs = [prompt_subqa_vqaintrospect[i].format(main_question=main_question.rstrip('?')) for main_question in batch["text_input"]]
                inputs = get_input(cfg.datasets_cfg.data_type, processor, device, batch["vision"], text_inputs)
                    
                generation_params = {
                    "do_sample": True,
                    "min_new_tokens": 1,
                    "max_new_tokens": 100,
                    "num_beams" : i+1,
                }
                if i != 0:
                    generation_params["length_penalty"] = -1
                    
                outputs = model.generate(**inputs, **generation_params)
                sub_questions = processor.batch_decode(outputs, skip_special_tokens=True)

            elif cfg.runner_cfg.sub_mode == "Ktype": # Generate Sub-Questions by Ktype
                sub_questions = []
                for question in batch["text_input"]:
                    tokens = nltk.word_tokenize(question)
                    tagged = nltk.pos_tag(tokens)
                    # Perform named entity recognition
                    entities = nltk.ne_chunk(tagged)
                    
                    entity_name = None
                    for subtree in entities:
                        if isinstance(subtree, nltk.Tree):
                            entity_name = " ".join([token for token, pos in subtree.leaves()])
                            entity_type = subtree.label()
                            break
                    else: # get the last noun or last word token in case of no named entity in question
                        nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
                        entity_name = nouns[0] if len(nouns) > 0 else tokens[-2]
                    
                    assert isinstance(entity_name, str), f"entity_name is not str but {type(entity_name)}."
                    sub_questions.append(prompt_Ktype[f"Ktype_{i}"].format(entity=entity_name))
            else:
                raise ValueError(f"Invalid sub_mode: {cfg.runner_cfg.sub_mode}")
                        
            # Generate Sub-Answers
            if cfg.runner_cfg.sub_mode in ["subqa", "fewshot_vqaintrospect"]:
                prompt = "Question: {sub_question}? Short answer:"
            else:
                prompt = "{sub_question}?"

            text_inputs = [prompt.format(sub_question=sub_question.rstrip('?')) for sub_question in sub_questions]
            inputs = get_input(cfg.datasets_cfg.data_type, processor, device, batch["vision"], text_inputs)
            
            generation_params = {
                "do_sample": False,
                "min_new_tokens": 1,
                "max_new_tokens": 10 if cfg.runner_cfg.sub_mode in ["subqa", "fewshot_vqaintrospect"] else 100,
                "num_beams": 5,
                "length_penalty": -1
            }
            outputs = model.generate(**inputs, **generation_params)
            sub_answers = processor.batch_decode(outputs, skip_special_tokens=True)

            # store to results    
            for b in range(bsz):
                if question_ids[b] not in results:
                    results[question_ids[b]] = []
                    batch_result[question_ids[b]] = []
                
                # if sub_questions[b].endswith('?'):
                results[question_ids[b]].append((sub_questions[b], sub_answers[b]))
                batch_result[question_ids[b]].append((sub_questions[b], sub_answers[b]))
            
        json.dump(batch_result, open(os.path.join(temp_dir, f"{cfg.datasets_cfg.dataset_name}_{data_iter_step}.json"), "w"), indent=4)
        
        if data_iter_step < 1:
            for main_question, (qid, sub_qas) in zip(batch["text_input"], results.items()):
                print(f'Question ID: {qid}')
                print(f'Main Question: {main_question}\nSub QAs:')
                pprint(sub_qas, width=300)
            # for main_question, sub_question, sub_answer in zip(batch["text_input"], sub_questions, sub_answers):
            #     print(f'Main Question: {main_question}\nSub Question: {sub_question}\nSub Answer: {sub_answer}\n')
            
    out_path = f"temp/subqa/sub_qas_val_{model_name.split('-')[-1]}_{cfg.runner_cfg.sub_mode}_{cfg.datasets_cfg.dataset_name}.json"
    json.dump(results, open(out_path, "w"), indent=4)
    print(f"Results saved to {out_path}")
    
    out_path = f"/data/{cfg.datasets_cfg.dataset_name}/sub_qas_val_{model_name.split('-')[-1]}_{cfg.runner_cfg.sub_mode}.json"
    json.dump(results, open(out_path, "w"), indent=4)
    print(f"Results saved to {out_path}")
    
    shutil.rmtree(temp_dir)
    print(f"Temp files removed.")

if __name__ == '__main__':
    main()
    