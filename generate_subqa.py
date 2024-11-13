import os, shutil
import argparse
import json
import nltk
from tqdm import tqdm
from pprint import pprint
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration

from models.model import VideoBlip2ForConditionalGeneration
from dataset.base_dataset import load_dataset
from configs.config import Config
from main_multi_subqa import setup_seeds
from dataset.VQA_Introspect import VQAIntrospectDataset



def parse_args():
    parser = argparse.ArgumentParser(description='LBA method')
    parser.add_argument("--cfg-path", default='configs/runner.yaml', help="path to configuration file.")
    # verbose
    parser.add_argument('--verbose', action='store_true', help='verbose')
    # remove temp files
    parser.add_argument('--save_temp', action='store_true', help='save temp files')
    parser.add_argument('--start', type=int, default=0, help='start index')
    parser.add_argument('--end', type=int, default=1000000, help='end index')
    
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    args = parser.parse_args()
    return args



def get_input(model_name, data_type, processor, device, vision_batch, text_inputs):
    if data_type == "images":
        inputs = processor(text=text_inputs, images=vision_batch, return_tensors="pt", padding=True)
    else:
        # import pdb; pdb.set_trace()
        if "Qwen" in model_name:# processor.__class__.__name__:
            from qwen_vl_utils import process_vision_info
            from utils.utils import ndarrays_to_base64
            
            messages_batch = []
            # texts_batch = []
            
            # import pdb; pdb.set_trace()
            for video, text_input in zip(vision_batch, text_inputs):
                base64_frames = ndarrays_to_base64(video, add_prefix=True)
                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": base64_frames,
                        },
                        {"type": "text", "text": text_input},
                    ],
                }]
                # base64_images = ndarrays_to_base64(video)
                # image_content = [{"type": "image", "image": "data:image;base64," + base64_images[i]} for i in range(len(video))]
                # messages = [
                #     # {"role": "system", "content": "You are a helpful assistant."},
                #     {
                #         "role": "user",
                #         "content": image_content + [{"type": "text", "text": text_input}],
                #     }
                # ]
                messages_batch.append(messages)
                # texts_batch.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                
            
            texts_batch = [
                processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in messages_batch
            ]
            
            image_inputs, video_inputs = process_vision_info(messages_batch)
            # import pdb; pdb.set_trace()
            inputs = processor(
                text=texts_batch,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
        elif "LLaVA-NeXT-Video" in model_name:
            conversations = []
            for video, text_input in zip(vision_batch, text_inputs):
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video"},
                            {"type": "text", "text": text_input},
                        ],
                    }
                ]
                conversations.append(conversation)
            
            prompt = processor.apply_chat_template(conversations, add_generation_prompt=True)
            
            video_batch = [np.array(v) for v in vision_batch]
            
            inputs = processor(text=prompt, videos=video_batch, padding=True, return_tensors="pt")
            
            
        elif "llama" in model_name:# processor.__class__.__name__:
            messages = []
            for video, txt in zip(vision_batch, text_inputs):
                message = {
                    "role": "user",
                    "content": [{"type": "image"} for _ in range(len(video))] + [{"type": "text", "text": txt}],
                }
                messages.append(message)
            
            processed_txt = processor.apply_chat_template(messages, add_generation_prompt=True)
            
            inputs = processor(
                text=processed_txt,
                images=vision_batch,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(text=text_inputs, return_tensors="pt", padding=True)
            pixel_values = []
            for video in vision_batch: # video: [n_frms, 640, 480]
                pixel_values.append(processor(images=video, return_tensors="pt", padding=True)['pixel_values'])  # [n_frms, 3, 224, 224]
            inputs["pixel_values"] = torch.stack(pixel_values, dim=0)#.to(device)
        # print("input_ids:", inputs["input_ids"].shape, inputs["input_ids"].device, "\tpixel_values:", inputs["pixel_values"].shape, inputs["pixel_values"].device)
        
    if device == "auto":
        return inputs
    else:
        return inputs.to(device)


"""
pre-generate sub-qa pairs for each question in the dataset
usage:
CUDA_VISIBLE_DEVICES=4 python generate_subqa.py --options runner.sub_mode="beam_and_greedy" datasets.dataset_name="DramaQA" runner.batch_size=12 runner.num_sub_qa_generate=5 runner.recomposer_name="Salesforce/blip2-flan-t5-xl"
CUDA_VISIBLE_DEVICES=4 python generate_subqa.py --options runner.sub_mode="fewshot_vqaintrospect" datasets.dataset_name="NExTQA" runner.batch_size=12 runner.num_sub_qa_generate=5 runner.recomposer_name="Salesforce/blip2-flan-t5-xl"
CUDA_VISIBLE_DEVICES=4 python generate_subqa.py --options runner.sub_mode="Ktype" datasets.dataset_name="DramaQA" runner.batch_size=2 datasets.num_data=5 runner.num_sub_qa_generate=6 runner.recomposer_name="Salesforce/blip2-flan-t5-xl"
CUDA_VISIBLE_DEVICES=4 python generate_subqa.py --options runner.sub_mode="beam" datasets.dataset_name="DramaQA" runner.batch_size=12 runner.num_sub_qa_generate=10 runner.recomposer_name="Salesforce/blip2-flan-t5-xl"

# Qwen/Qwen2-VL-7B-Instruct
CUDA_VISIBLE_DEVICES=2 python generate_subqa.py --options runner.sub_mode="beam_and_greedy" datasets.dataset_name="DramaQA" runner.batch_size=1 runner.num_sub_qa_generate=5 runner.recomposer_name="Qwen/Qwen2-VL-7B-Instruct"
CUDA_VISIBLE_DEVICES=3 python generate_subqa.py --options runner.sub_mode="fewshot_vqaintrospect" datasets.dataset_name="DramaQA" runner.batch_size=1 runner.num_sub_qa_generate=5 runner.recomposer_name="Qwen/Qwen2-VL-7B-Instruct"

# Llama-3.2-11B-Vision-Instruct
CUDA_VISIBLE_DEVICES=4 python generate_subqa.py --options runner.sub_mode="beam_and_greedy" datasets.dataset_name="DramaQA" runner.batch_size=1 runner.num_sub_qa_generate=5 runner.recomposer_name="meta-llama/Llama-3.2-11B-Vision-Instruct"
CUDA_VISIBLE_DEVICES=5 python generate_subqa.py --options runner.sub_mode="fewshot_vqaintrospect" datasets.dataset_name="DramaQA" runner.batch_size=1 runner.num_sub_qa_generate=5 runner.recomposer_name="meta-llama/Llama-3.2-11B-Vision-Instruct"

# "llava-hf/LLaVA-NeXT-Video-7B-hf"

"""
def main():
    N_SUPPLE = 0
    args = parse_args()
    if args.start != 0 or args.end != 1000000:
        print(f"Start: {args.start}, End: {args.end}")
        args.save_temp = True
    cfg = Config(args)
    setup_seeds(cfg)
    model_name = cfg.runner_cfg.recomposer_name
    processor_name = model_name
    # processor_name = "Salesforce/instructblip-flan-t5-xl"
    # model_name = processor_name
    cache_dir = os.path.join(cfg.model_cfg.cache_dir, model_name.split("/")[0])
    if torch.cuda.device_count() > 1:
        device = "auto"
    else:
        device = "cuda"
    N = cfg.runner_cfg.num_sub_qa_generate
    if N != 5:
        N_tag = f"_N{N}"
    else:
        N_tag = ""
    
    if "VideoLLaMA" in model_name:
        # if cfg.runner_cfg.batch_size != 1:
        #     raise ValueError("batch_size should be 1 for VideoLLaMA.")
        import sys
        sys.path.append('./VideoLLaMA2/')
        from VideoLLaMA2.videollama2 import model_init
        from VideoLLaMA2.videollama2.utils import disable_torch_init
        disable_torch_init()
        model, processor, tokenizer = model_init(
            model_name, 
            cache_dir=cache_dir, 
            # device_map=device_map,
        )
        processor = processor[cfg.datasets_cfg.data_type[:-1]]
        pass

    elif "LLaVA-NeXT-Video" in model_name:
        from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
        processor = LlavaNextVideoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
            cache_dir=cache_dir, 
            device_map=cfg.runner_cfg.device_map,
        )
    
    elif "Qwen" in model_name:
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from qwen_vl_utils import process_vision_info

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, # "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            cache_dir="/model/Qwen/"
        )

        # default processer
        processor = AutoProcessor.from_pretrained(
            model_name, # "Qwen/Qwen2-VL-7B-Instruct",
            cache_dir="/model/Qwen/",
            min_pixels = 256 * 28 * 28,
            max_pixels = 1280 * 28 * 28,
        )
        
    elif "llama" in model_name:
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        # model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

        model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="/model/llama/",
        )
        processor = AutoProcessor.from_pretrained(model_name, cache_dir="/model/llama/",)
        
    elif cfg.datasets_cfg.data_type == "images": # dataset_name in ["VQA_Introspect", "AOKVQA", "OKVQA"]:
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(device)#, device_map="auto")
        processor = Blip2Processor.from_pretrained(processor_name, cache_dir=cache_dir)
        
    else: # "videos"
        model = VideoBlip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(device)#, device_map="auto")
        processor = Blip2Processor.from_pretrained(processor_name, cache_dir=cache_dir)
    # model = InstructBlipVideoForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(device)#, device_map="auto")
    # processor = InstructBlipVideoProcessor.from_pretrained(processor_name, cache_dir=cache_dir)
    
    
    xl_or_xxl = "xl" if "-xl" in cfg.runner_cfg.recomposer_name or "7b" in cfg.runner_cfg.recomposer_name.lower() else "xxl"
    if "Qwen" in cfg.runner_cfg.recomposer_name:
        model_tag = cfg.runner_cfg.recomposer_name.split('/')[-1].replace('-', '_')
    else:
        model_tag = cfg.runner_cfg.recomposer_name.split('-')[-1]
    print('xl_or_xxl:', xl_or_xxl)
    print('model_tag:', model_tag)

    ann_paths = [os.path.join(cfg.datasets_cfg.root_dir, path) for path in cfg.datasets_cfg.ann_paths.get(cfg.datasets_cfg.split, 'val')]
    if len(ann_paths) >= 2:
        if os.path.exists(ann_paths[1].replace("xl", model_tag)):
            cfg.datasets_cfg.ann_paths.get(cfg.datasets_cfg.split, 'val')[-1] = ann_paths[-1] = ann_paths[-1].replace("xl", model_tag)
        else:
            cfg.datasets_cfg.ann_paths.get(cfg.datasets_cfg.split, 'val')[-1] = ann_paths[-1] = ann_paths[-1].replace("xl", xl_or_xxl)
    
    dataset = load_dataset(cfg.datasets_cfg, n_supple=N_SUPPLE, ann_paths=ann_paths)#model_tag=model_name.split("/")[-1])
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
                                                  ['/data/VQA_Introspect/VQAIntrospect_valv1.0.json', '/data/VQA/v2/v2_mscoco_val2014_annotations.json'],
                                                  num_data=-1, split='val')

    prompt_subqa_vqaintrospect = []
    idx_example = 0
    for i in range(N):
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
    # pprint(prompt_subqa_vqaintrospect, width=300)
    
    if "Qwen" in cfg.runner_cfg.recomposer_name:
        model_tag = "Qwen2VL" # cfg.runner_cfg.recomposer_name.split('/')[-1].replace('-', '_')
    else:
        model_tag = cfg.runner_cfg.recomposer_name.split('-')[-1]
    
    temp_dir = f"temp/files/{cfg.datasets_cfg.dataset_name}/{model_tag}/{cfg.runner_cfg.sub_mode}/"
    os.makedirs(temp_dir, exist_ok=True)
    
    results = {}
    for data_iter_step, batch in enumerate(tqdm(dataloader)):
        if data_iter_step < args.start or data_iter_step >= args.end:
            continue
        
        # if all question_id saved in output_dir/files/questionid.json, skip
        for question_id in batch['question_id']:
            saved_path = os.path.join(temp_dir, f"{cfg.datasets_cfg.dataset_name}_{question_id}.json")
            if not os.path.exists(saved_path):
                break
            result = json.load(open(saved_path))
            results[question_id] = result
        else:
            continue
    
        
        if os.path.exists(os.path.join(temp_dir, f"{cfg.datasets_cfg.dataset_name}_{data_iter_step}.json")):
            batch_result = json.load(open(os.path.join(temp_dir, f"{cfg.datasets_cfg.dataset_name}_{data_iter_step}.json"), "r"))
            results.update(batch_result)
            continue
        
        question_ids = batch["question_id"]
        bsz = len(batch["question_id"])
        
        if all(qid in results for qid in question_ids):
            continue
        
        batch_result = {}
        if cfg.runner_cfg.sub_mode == "beam": # TODO
            # Generate Sub-Questions 
            prompt = "Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {main_question}? Perception Question:"
            text_inputs = [prompt.format(main_question=main_question.rstrip('?')) for main_question in batch["text_input"]]
            
            inputs = get_input(model_name, cfg.datasets_cfg.data_type, processor, device, batch["vision"], text_inputs)
            
            outputs = model.generate(
                **inputs,
                min_new_tokens=1,
                max_new_tokens=100,
                num_beams=N,
                num_return_sequences=N,
                length_penalty=-1,
                # return_dict_in_generate=True,
                # output_scores=True,
            )
            pprint(processor.batch_decode(model.generate(**inputs, min_new_tokens=1, max_new_tokens=100, num_beams=10, num_return_sequences=10, length_penalty=-1, no_repeat_ngram_size=2), skip_special_tokens=True), width=300)
            # if "Qwen" in model_name:
            #     outputs = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
            sub_questions = processor.batch_decode(outputs, skip_special_tokens=True)
            import pdb; pdb.set_trace()
            
            for i in range(N):
                # TODO
                
                # Generate Sub-Answers
                if cfg.runner_cfg.sub_mode in ["beam_and_greedy", "fewshot_vqaintrospect"]:
                    prompt = "Question: {sub_question}? Short answer:"
                else:
                    prompt = "{sub_question}?"

                text_inputs = [prompt.format(sub_question=sub_question.rstrip('?')) for sub_question in sub_questions]
                inputs = get_input(model_name, cfg.datasets_cfg.data_type, processor, device, batch["vision"], text_inputs)
                
                generation_params = {
                    "do_sample": False,
                    "min_new_tokens": 1,
                    "max_new_tokens": 10 if cfg.runner_cfg.sub_mode in ["beam_and_greedy", "fewshot_vqaintrospect"] else 100,
                    "num_beams": 5,
                    "length_penalty": -1
                }
                outputs = model.generate(**inputs, **generation_params)
                if "Qwen" in model_name:
                    outputs = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
                sub_answers = processor.batch_decode(outputs, skip_special_tokens=True)

                # store to results    
                for b in range(bsz):
                    # if question_ids[b] not in results:
                    #     results[question_ids[b]] = []
                    if question_ids[b] not in batch_result:
                        batch_result[question_ids[b]] = []
                    # if sub_questions[b].endswith('?'):
                    # if len(results[question_ids[b]]) < cfg.runner_cfg.num_sub_qa_generate:
                    #     results[question_ids[b]].append((sub_questions[b], sub_answers[b]))
                    batch_result[question_ids[b]].append((sub_questions[b], sub_answers[b]))
                    
        else:
            for i in range(N):
                if cfg.runner_cfg.sub_mode == "beam_and_greedy":   # Generate Sub-Questions by Huggingface model
                    prompt = "Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {main_question}? Perception Question:"
                    text_inputs = [prompt.format(main_question=main_question.rstrip('?')) for main_question in batch["text_input"]]
                    
                    if "VideoLLaMA" in model_name:
                        from VideoLLaMA2.videollama2 import mm_infer_batch
                        image_or_videos = []
                        for v in batch["vision"]:
                            image_or_videos.append(processor(v))
                        image_or_videos = torch.stack(image_or_videos)
                        # image_or_video = processor(batch["vision"][0])
                    else:
                        inputs = get_input(model_name, cfg.datasets_cfg.data_type, processor, device, batch["vision"], text_inputs)
                        
                    generation_params = {
                        "do_sample": True,
                        "min_new_tokens": 1,
                        "max_new_tokens": 200,
                    }
                    beam_search = i==0
                    if beam_search:
                        if "Qwen" in model_name:
                            model.generation_config.temperature=None
                            model.generation_config.top_p=None
                            model.generation_config.top_k=None
                            generation_params["do_sample"] = False
                        generation_params["num_beams"] = 5
                        generation_params["length_penalty"] = -1
                        if N == 1:
                            generation_params["return_dict_in_generate"] = True
                            generation_params["output_scores"] = True
                    else:
                        generation_params["top_p"] = 0.8
                        
                    if "VideoLLaMA" in model_name:
                        sub_questions, o_score = mm_infer_batch(
                            image_or_videos, text_inputs, model, tokenizer, modal=cfg.datasets_cfg.data_type[:-1],
                            **generation_params
                        )
                        # sub_questions, o_score = mm_infer(
                        #     image_or_video, text_inputs[0], model, tokenizer, modal=cfg.datasets_cfg.data_type[:-1], 
                        #     **generation_params
                        # )
                    else:
                        outputs = model.generate(**inputs, **generation_params)
                        if N == 1: 
                            sub_questions_scores = outputs.sequences_scores.tolist() # torch.exp(outputs.sequences_scores).tolist()
                            outputs = outputs.sequences
                            
                        if "Qwen" in model_name:
                            outputs = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
                            
                        sub_questions = processor.batch_decode(outputs, skip_special_tokens=True)

                elif cfg.runner_cfg.sub_mode == "fewshot_vqaintrospect":
                    text_inputs = [prompt_subqa_vqaintrospect[i].format(main_question=main_question.rstrip('?')) for main_question in batch["text_input"]]
                    if "VideoLLaMA" in model_name:
                        from VideoLLaMA2.videollama2 import mm_infer_batch
                        image_or_videos = []
                        for v in batch["vision"]:
                            image_or_videos.append(processor(v))
                        image_or_videos = torch.stack(image_or_videos)
                        # image_or_video = processor(batch["vision"][0])
                    else:
                        inputs = get_input(model_name, cfg.datasets_cfg.data_type, processor, device, batch["vision"], text_inputs)
                        
                    generation_params = {
                        "do_sample": True,
                        "min_new_tokens": 1,
                        "max_new_tokens": 200,
                        "num_beams" : i+1,
                    }
                    if i != 0:
                        if "Qwen" in model_name:
                            model.generation_config.temperature=None
                            model.generation_config.top_p=None
                            model.generation_config.top_k=None
                            generation_params["do_sample"] = False
                        generation_params["length_penalty"] = -1
                        
                    if "VideoLLaMA" in model_name:
                        sub_questions, o_score = mm_infer_batch(
                            image_or_videos, text_inputs, model, tokenizer, modal=cfg.datasets_cfg.data_type[:-1],
                            **generation_params
                        )
                    else:
                        outputs = model.generate(**inputs, **generation_params)
                        if "Qwen" in model_name:
                            outputs = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
                            
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
                    
                if "LLaVA-NeXT-Video" in model_name:
                    sub_questions = [sub_question.split("ASSISTANT: ")[-1] for sub_question in sub_questions]

                text_inputs = [prompt.format(sub_question=sub_question.rstrip('?')) for sub_question in sub_questions]
                if "VideoLLaMA" in model_name:
                        from VideoLLaMA2.videollama2 import mm_infer_batch
                        image_or_videos = []
                        for v in batch["vision"]:
                            image_or_videos.append(processor(v))
                        image_or_videos = torch.stack(image_or_videos)
                        # image_or_video = processor(batch["vision"][0])
                else:
                    inputs = get_input(model_name, cfg.datasets_cfg.data_type, processor, device, batch["vision"], text_inputs)
                
                generation_params = {
                    "do_sample": False,
                    "min_new_tokens": 1,
                    "max_new_tokens": 200,# if cfg.runner_cfg.sub_mode in ["subqa", "fewshot_vqaintrospect"] else 100,
                    "num_beams": 5,
                    "length_penalty": -1
                }
                if N == 1:
                    generation_params["return_dict_in_generate"] = True
                    generation_params["output_scores"] = True
                    
                if "Qwen" in model_name:
                    model.generation_config.temperature=None
                    model.generation_config.top_p=None
                    model.generation_config.top_k=None
                    generation_params["do_sample"] = False
                
                if "VideoLLaMA" in model_name:
                    sub_answers, o_score = mm_infer_batch(
                        image_or_videos, text_inputs, model, tokenizer, modal=cfg.datasets_cfg.data_type[:-1],
                        **generation_params
                    )
                else:
                    outputs = model.generate(**inputs, **generation_params)
                    if N == 1: 
                        sub_answers_scores = outputs.sequences_scores.tolist() # torch.exp(outputs.sequences_scores).tolist()
                        outputs = outputs.sequences
                        
                    if "Qwen" in model_name:
                        outputs = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
                        
                    sub_answers = processor.batch_decode(outputs, skip_special_tokens=True)
                    if "LLaVA-NeXT-Video" in model_name:
                        sub_answers = [sub_answer.split("ASSISTANT: ")[-1] for sub_answer in sub_answers]

                # store to results    
                for b in range(bsz):
                    # if question_ids[b] not in results:
                    #     results[question_ids[b]] = []
                    if question_ids[b] not in batch_result:
                        batch_result[question_ids[b]] = []
                    # if sub_questions[b].endswith('?'):
                    # if len(results[question_ids[b]]) < cfg.runner_cfg.num_sub_qa_generate:
                    #     results[question_ids[b]].append((sub_questions[b], sub_answers[b]))
                    # import pdb; pdb.set_trace()
                    if N == 1:
                        batch_result[question_ids[b]].append((sub_questions[b], sub_answers[b], sub_questions_scores[b], sub_answers_scores[b]))
                    else:
                        batch_result[question_ids[b]].append((sub_questions[b], sub_answers[b]))
                
        
        for k, v in batch_result.items():
            json.dump(v, open(os.path.join(temp_dir, f"{cfg.datasets_cfg.dataset_name}_{k}.json"), "w"), indent=4)
        results.update(batch_result)
        
        if data_iter_step < 1:
            for main_question, (qid, sub_qas) in zip(batch["text_input"], results.items()):
                print(f'Question ID: {qid}')
                print(f'Main Question: {main_question}\nSub QAs:')
                pprint(sub_qas, width=300)
            # for main_question, sub_question, sub_answer in zip(batch["text_input"], sub_questions, sub_answers):
            #     print(f'Main Question: {main_question}\nSub Question: {sub_question}\nSub Answer: {sub_answer}\n')
        
    
    if args.save_temp:
        print(f"Temp files saved to {temp_dir}")
    else:
        if "Qwen" in model_name:
            tag = model_name.split('/')[-1].replace('-', '_')
        else:
            tag = model_name.split('-')[-1]
        out_path = f"temp/subqa/sub_qas_val_{tag}_{cfg.runner_cfg.sub_mode}_{cfg.datasets_cfg.dataset_name}{N_tag}.json"
        json.dump(results, open(out_path, "w"), indent=4)
        print(f"Results saved to {out_path}")
        
        out_path = os.path.join(cfg.datasets_cfg.root_dir, f"{cfg.datasets_cfg.dataset_name}/sub_qas_val_{tag}_{cfg.runner_cfg.sub_mode}{N_tag}.json")
        json.dump(results, open(out_path, "w"), indent=4)
        print(f"Results saved to {out_path}")
        
        shutil.rmtree(temp_dir)
        print(f"Temp files removed.")

if __name__ == '__main__':
    main()
    