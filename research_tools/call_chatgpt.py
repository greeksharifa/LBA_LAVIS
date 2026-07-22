import os
import re
import argparse
import json
import datetime

from tqdm import tqdm
from pprint import pprint
from time import sleep


import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration

from openai import OpenAI

from models.model import VideoBlip2ForConditionalGeneration, Recomposer
from dataset.base_dataset import load_dataset
from configs.config import Config
from main_multi_subqa import setup_seeds
# from utils.api_chatgpt import call_vision_api, call_chat_api
from call_api import *
# from temp.line_message import line_notify


def api_result_to_json(api_result, verbose=True):
    result_json = json.loads(api_result.json())

    retry = result_json["request_counts"]["failed"] > 0 or result_json["status"] == "expired"
    for k, v in result_json.items():
        if k.endswith("_at") and v:
            # if retry:
            if verbose:
                print(f'{k:20s}: {datetime.datetime.fromtimestamp(v)}')
            result_json[k] = datetime.datetime.fromtimestamp(v)
        elif v is not None or "errors" in k:
            # if retry:
            if verbose:
                print(f'{k:20s}: {v}')
        else:
            pass

    return result_json, retry


def main():
    """
    pre-generate sub-qa pairs for each question in the dataset
    usage:
    python call_chatgpt.py --chatgpt_mode=subq --options datasets.dataset_name="NExTQA" runner.batch_size=1
    CUDA_VISIBLE_DEVICES=4 python call_chatgpt.py --chatgpt_mode=suba --options datasets.dataset_name="NExTQA" runner.batch_size=1
    python call_chatgpt.py --chatgpt_mode=maina_before --options datasets.dataset_name="NExTQA" runner.batch_size=1 datasets.num_data=10
    python call_chatgpt.py --chatgpt_mode=maina_after --options datasets.dataset_name="NExTQA" runner.batch_size=1 runner.num_sub_qa_select=1 datasets.num_data=3
    # verify
    CUDA_VISIBLE_DEVICES=4 python call_chatgpt.py --chatgpt_verify --chatgpt_mode=suba --options datasets.dataset_name="NExTQA" runner.batch_size=1
    """
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)

    # dataset
    if cfg.runner_cfg.batch_size != 1:
        print("runner.batch_size must be 1. Automatically set to 1.")
    # import pdb; pdb.set_trace()
    ann_paths = [os.path.join(cfg.datasets_cfg.root_dir, path) for path in cfg.datasets_cfg.ann_paths.get(cfg.datasets_cfg.split, 'val')]
    dataset = load_dataset(cfg.datasets_cfg, ann_paths=ann_paths, n_supple=0)
    open_ended = cfg.datasets_cfg.open_ended
    # dataloader = DataLoader(dataset, batch_size=cfg.runner_cfg.batch_size,
    #                         shuffle=False, collate_fn=dataset.collater)

    # verifying model
    if args.chatgpt_verify and args.chatgpt_mode == "suba":
        model = None#InstructBlipVideoForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", cache_dir="/model/Salesforce/")#.to("cuda")
        processor = None#InstructBlipVideoProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl", cache_dir="/model/Salesforce/")

        # model = Recomposer(cfg, device="cuda:0", model_type="recomposer")
        # model = VideoBlip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir="/model/Salesforce/")#.to("cuda")
        # processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", cache_dir="/model/Salesforce/")
        # if cfg.runner_cfg.device_map != "auto":
        #     model = model.to(cfg.runner_cfg.device_map)


    # openai
    api_key = json.load(open("temp/api_key.json", "r"))["openai_jsj"]
    client = OpenAI(api_key=api_key)

    # generate result
    format_error_list = []
    fatal_error_list = []
    file_not_exist_list = []
    question_set = set()


    dataset_name = cfg.datasets_cfg.dataset_name
    num_sub_qa_select = cfg.runner_cfg.num_sub_qa_select
    root_dir = f"/data/ywjang/chatgpt_result/{dataset_name}"
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(os.path.join(root_dir, f"{args.chatgpt_mode}"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, f"{args.chatgpt_mode}/select{num_sub_qa_select}/batch_inputs"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, f"{args.chatgpt_mode}/select{num_sub_qa_select}/batch_results"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, f"{args.chatgpt_mode}/select{num_sub_qa_select}/error_results"), exist_ok=True)


    # batch_job_ids
    batch_job_ids_path = os.path.join(root_dir, f"{args.chatgpt_mode}/select{num_sub_qa_select}/batch_job_ids.json")
    if os.path.exists(batch_job_ids_path):
        batch_job_ids = json.load(open(batch_job_ids_path, "r"))
    else:
        batch_job_ids = {}


    sub_questions_path = os.path.join(root_dir, f"chatgpt_result_{dataset_name}_subq.json")
    sub_answers_path = os.path.join(root_dir, f"chatgpt_result_{dataset_name}_suba.json")
    main_answers_path = os.path.join(root_dir, f"chatgpt_result_{dataset_name}_{args.chatgpt_mode}_select{num_sub_qa_select}.json")

    results = {}
    # load sub_q when mode==sub_a, load sub_q&sub_a when mode==maina_after

    if args.chatgpt_mode == "subq":
        if os.path.exists(sub_questions_path):
            results = json.load(open(sub_questions_path, "r"))
    if args.chatgpt_mode == "suba":
        sub_questions = json.load(open(sub_questions_path, "r"))
        if os.path.exists(sub_answers_path):
            results = json.load(open(sub_answers_path, "r"))

    elif args.chatgpt_mode == "maina_after":
        sub_questions = json.load(open(sub_questions_path, "r"))
        sub_answers = json.load(open(sub_answers_path, "r"))
        if os.path.exists(main_answers_path):
            results = json.load(open(main_answers_path, "r"))

    # if not args.chatgpt_verify and len(results) == len(dataset):
    #     print(f"results already exist. len(results) == len(dataset) == {len(dataset)}")
    #     return

    errors = {}
    completed, failed, expired, total = 0, 0, 0, 0

    for data_iter_step, data in enumerate(tqdm(dataset)):
        success = False
        while not success:
            try:
                question_id = str(data["question_id"])
                question_set.add(question_id)

                # print("*" * 120 + f"\ndata_iter_step: {data_iter_step:5d} | question_id: {question_id} processing...")

                if args.chatgpt_verify:
                    # if data_iter_step >= 30: break

                    # if already processed, skip
                    if args.chatgpt_mode == "subq":
                        # import pdb; pdb.set_trace()
                        if question_id in results and type(results[question_id]) == list and len(results[question_id]) == 5:
                            results[question_id] = [r.strip() for r in results[question_id]]
                            print(f"question_id: {question_id} sub_answer already exists.")
                            success = True
                            continue
                        elif question_id in results:
                            try:
                                # questions = '[' + results[question_id].split('[')[1].split(']')[0] + ']'
                                start = results[question_id].index('[') + 1
                                end = len(results[question_id]) - results[question_id][::-1].index(']') - 1
                                questions = '[' + results[question_id][start:end] + ']'
                                import re
                                questions = re.sub(r'\s*\n\s*', '', questions)
                                questions = json.loads(questions)
                                if len(questions) == 5:
                                    results[question_id] = questions
                                    print(f"question_id: {question_id} sub_question already exists.")
                                    success = True
                                    continue
                            except:
                                import traceback
                                traceback.print_exc()
                                print('question_id:', question_id)
                                print('results[question_id]:', results[question_id])
                                print('questions:', questions)
                                import pdb; pdb.set_trace()
                    if args.chatgpt_mode == "suba":
                        if question_id in results and type(results[question_id]) == list and len(results[question_id]) == 5:
                            results[question_id] = [r.strip() for r in results[question_id]]
                            print(f"question_id: {question_id} sub_answer already exists.")
                            success = True
                            continue
                        elif question_id in results:
                            answer = results[question_id].replace('\n\n', '\n').lstrip('Answers:').strip()
                            print('*' * 40 + f' question_id: {question_id} ' + '*' * 40)
                            print(f'ANSWER:\n{answer}')
                            import re
                            revised = re.findall(r'\b(.*.)[ \n]*', answer)
                            revised = [r.strip() for r in revised]
                            print(f'REVISED:')
                            pprint(revised, width=300)
                            if len(revised) == 5:
                                pass
                            elif len(revised) > 5:
                                revised = revised[-5:]
                            else:
                                revised += [revised[-1]] * (5 - len(revised))

                            results[question_id] = revised

                            print(f"question_id: {question_id} sub_answer already exists.")
                            success = True
                            continue

                            # if "I'm" not in answer and "I don't" not in answer and "I can't" not in answer:
                            #     import pdb; pdb.set_trace()
                            # success = True
                            # continue

                    elif args.chatgpt_mode in ["maina_before", "maina_after"]:
                        if question_id in results:
                            print(f"question_id: {question_id} main_answer already exists.")
                            continue

                    # not processed
                    if args.chatgpt_mode == "subq":
                        raise NotImplementedError("verifying subq is not implemented.")
                    elif args.chatgpt_mode == "suba":
                        # continue
                        text_input = sub_questions[question_id]
                        import pdb; pdb.set_trace()
                        video = [np.array(data["vision"]) for _ in range(len(text_input))]

                        # inputs = processor(text=text_input, images=video, return_tensors="pt").to(model.device)
                        inputs = get_input(processor, device="cuda", video_batch=video, text_inputs=text_input)

                        outputs = model.generate(
                            **inputs,
                            do_sample=False,
                            num_beams=5,
                            max_length=256,
                            repetition_penalty=1.5,
                            length_penalty=1.0,
                        )
                        sub_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

                        # inputs = processor(text=text_input, images=video, return_tensors="pt", padding=True)#.to("cuda")
                        # if cfg.runner_cfg.device_map != "auto":
                        #     inputs = inputs.to(cfg.runner_cfg.device_map)
                        # outputs = model.generate(**inputs, max_new_tokens=50)
                        # outputs = model(vision=video, text_inputs=text_input)
                        # sub_answer = processor.batch_decode(outputs, skip_special_tokens=True)
                        print('instructblip sub_answer:', sub_answer)
                        # sub_answer, _ = outputs
                        results[question_id] = sub_answer

                    elif args.chatgpt_mode == "maina_before":
                        raise NotImplementedError("verifying maina_before is not implemented.")
                    elif args.chatgpt_mode == "maina_after":
                        raise NotImplementedError("verifying maina_after is not implemented.")

                    success = True
                else:
                    if cfg.datasets_cfg.num_data != -1 and data_iter_step >= cfg.datasets_cfg.num_data:
                        break

                    batch_inputs_path = os.path.join(root_dir, f"{args.chatgpt_mode}/select{num_sub_qa_select}/batch_inputs/{question_id}.jsonl")

                    verbose = True
                    if question_id in batch_job_ids:
                        verbose = False
                        batch_job_id = batch_job_ids[question_id]
                        # print(f"question_id: {question_id} is already in batch_job_ids with batch_job_id: {batch_job_id}")
                        batch_job = client.batches.retrieve(batch_job_id)
                        result_json, create_batch = api_result_to_json(batch_job, verbose=verbose)
                        # import pdb; pdb.set_trace()
                        # answer = results[question_id]
                        # if args.chatgpt_mode == "suba" and ("I'm" in answer or "I don't" in answer or "I can't" in answer):
                        #     import pdb; pdb.set_trace()
                            # create_batch = True
                        if create_batch:
                            print(f"question_id: {question_id} is already in batch_job_ids with batch_job_id: {batch_job_id} but failed.")
                    else:
                        create_batch = True
                        print(f"question_id: {question_id} is not in batch_job_ids. creating...")

                    # import pdb; pdb.set_trace()

                    if create_batch:
                        if True:#not os.path.exists(batch_inputs_path):
                            video = data["vision"]
                            # print(type(video[0]))
                            if isinstance(video[0], np.ndarray):
                                base64_frames = ndarrays_to_base64(video)
                            elif isinstance(video[0], Image.Image):
                                base64_frames = images_to_base64(video)
                            else:
                                raise NotImplementedError(f"video type {type(video[0])} is not supported.")

                            content = []
                            for base64_frame in base64_frames:
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_frame}",
                                        "detail": args.chatgpt_vision_detail,
                                    }
                                })

                            candidate_list = data["candidate_list"] if "candidate_list" in data else None
                            if args.chatgpt_mode == "subq":
                                prompts = [get_subq_prompt(data["text_input"])]
                            elif args.chatgpt_mode == "suba":
                                prompts = [get_suba_prompt(sub_questions[question_id])]
                            elif args.chatgpt_mode == "maina_before":
                                prompts = [get_maina_before_prompt(data["text_input"], open_ended, candidate_list)]
                            elif args.chatgpt_mode == "maina_after":
                                if num_sub_qa_select == 5:
                                    prompts = [get_maina_after_prompt(data["text_input"], data['sub_question_list'], data['sub_answer_list'], open_ended, candidate_list)]
                                else:
                                    prompts = []
                                    for i in range(5):
                                        sq_list, sa_list = [], []
                                        for j in range(num_sub_qa_select):
                                            s_idx = (i+j) % 5
                                            sq_list.append(data['sub_question_list'][s_idx])
                                            sa_list.append(data['sub_answer_list'][s_idx])
                                        prompts.append(get_maina_after_prompt(data["text_input"], sq_list, sa_list, open_ended, candidate_list))

                            for p_idx, prompt in enumerate(prompts):
                                # print(f"question_id: {question_id} p_idx: {p_idx} prompt:\n{prompt}\n")
                                content.append({
                                    "type": "text",
                                    "text": prompt
                                })
                                batch_input = {
                                    "custom_id": f'{question_id}_{p_idx}',
                                    "method": "POST",
                                    "url": "/v1/chat/completions",
                                    "body": {
                                        "model": "gpt-4o-mini",
                                        "messages": [
                                            {"role": "system", "content": "You are an AI assistant who has rich video understanding abilities. You will be provided with video frames."},
                                            {"role": "user", "content": content}
                                        ],
                                        "max_tokens": 500,
                                        "logprobs":True,
                                    }
                                }
                                file_mode = 'w' if p_idx == 0 else 'a'
                                with open(batch_inputs_path, file_mode) as f:
                                    f.write(json.dumps(batch_input) + '\n')

                        # openai batch
                        batch_file = client.files.create(
                            file=open(batch_inputs_path, "rb"),
                            purpose="batch"
                        )
                        print('batch_file:', batch_file)
                        print('batch_file.id:', batch_file.id)

                        sleep(2)
                        batch_job = client.batches.create(
                            input_file_id=batch_file.id,
                            endpoint="/v1/chat/completions",
                            completion_window="24h",
                            metadata={
                                "dataset_name": dataset_name,
                                "question_id": question_id
                            }
                        )
                        batch_job_id = batch_job.id
                        batch_job_ids[question_id] = batch_job_id
                        json.dump(batch_job_ids, open(batch_job_ids_path, "w"), indent=4)

                    batch_job = client.batches.retrieve(batch_job_id)
                    result_json, create_batch = api_result_to_json(batch_job, verbose=verbose)
                    if create_batch:
                        import pdb; pdb.set_trace()

                    c = result_json["request_counts"]["completed"]
                    f = result_json["request_counts"]["failed"]
                    e = result_json["status"] == "expired" and (c == 0 or f == 0)
                    t = result_json["request_counts"]["total"]
                    completed += c
                    failed += f
                    expired += e
                    total += t
                    if f or e:
                        result_json, create_batch = api_result_to_json(batch_job, verbose=True)
                        print(f"question_id: {question_id} | completed: {completed:5d} | failed: {f:5d} | expired: {e:5d} | total: {t:5d}")

                    # if t - (c+f+e) != 0:
                    #     print(f"question_id: {question_id} t - (c+f+e) != 0")
                    #     import pdb; pdb.set_trace()
                    #     fatal_error_list.append(question_id)

                    # print(batch_job)
                    print(f"completed: {completed:5d} | failed: {failed:5d} | expired: {expired:5d} | total: {total:5d}")
                    success = True

                    result_file_id = batch_job.output_file_id
                    if result_file_id is None:
                        print(f"result_file_id is None (maybe validating or in progress). question_id={question_id}")
                        continue
                    else:
                        print(f"result_file_id: {result_file_id}")
                        api_result = client.files.content(result_file_id).content

                        result_file_name = os.path.join(root_dir, f"{args.chatgpt_mode}/select{num_sub_qa_select}/batch_results/{question_id}.jsonl")

                        with open(result_file_name, 'wb') as file:
                            file.write(api_result)

                        # Loading data from saved file
                        with open(result_file_name, 'r') as file:
                            if args.chatgpt_mode == "maina_after":
                                results[question_id] = {
                                    "question_id": question_id,
                                    "text_input": [],
                                    "main_question": data["text_input"],
                                    "gt_ans": dataset.answer_mapping(data["gt_ans"]),
                                    "confidence_lba": [],
                                    "text_output_lba": [],
                                    "api_result_text_lba": [],
                                    "logprobs_contents_lba": [],
                                }


                            for line in file:
                                # Parsing the JSON string into a dict and appending to the list of results
                                json_object = json.loads(line.strip())
                                api_result_text = json_object['response']['body']['choices'][0]['message']['content']
                                # if data_iter_step == len(dataset)-1:
                                #     import pdb; pdb.set_trace()

                                if args.chatgpt_mode == "subq":
                                    formatted_subq = reformat_subq(api_result_text)
                                    if formatted_subq:
                                        results[question_id] = formatted_subq
                                    else:
                                        results[question_id] = api_result_text
                                elif args.chatgpt_mode == "suba":
                                    formatted_suba = reformat_suba(api_result_text)
                                    if question_id in results:
                                        answer = results[question_id]
                                        if "I'm" in answer or "I don't" in answer or "I can't" in answer:
                                            print('before:', api_result_text)
                                            if formatted_suba:
                                                print('after, formatted_suba:', formatted_suba)
                                            else:
                                                print('after, api_result_text:', api_result_text)
                                    if formatted_suba:
                                        results[question_id] = formatted_suba
                                    else:
                                        results[question_id] = api_result_text
                                elif args.chatgpt_mode == "maina_before":
                                    # maina_before = api_result['choices'][0]['message']['content']
                                    # print('maina_before:', maina_before)
                                    logprobs_contents = json_object['response']['body']['choices'][0]['logprobs']["content"]
                                    maina_before = ""
                                    confidence_score = 0.
                                    for logprobs_content in logprobs_contents:
                                        logprob = logprobs_content["logprob"]
                                        confidence_score += logprob
                                        maina_before += logprobs_content["token"]
                                        print(logprobs_content["token"], '\t', logprobs_content["logprob"])
                                        if maina_before in ["A", "B", "C", "D", "E", "(A)", "(B)", "(C)", "(D)", "(E)"]:
                                            break
                                    if maina_before in "ABCDE":
                                        maina_before = "(" + maina_before + ")"

                                    candidate_list = data["candidate_list"] if "candidate_list" in data else None
                                    results[question_id] = {
                                        "question_id": question_id,
                                        "text_input": get_maina_before_prompt(data["text_input"], open_ended, candidate_list),
                                        "main_question": data["text_input"],
                                        "gt_ans": dataset.answer_mapping(data["gt_ans"]),
                                        "confidence_base": confidence_score,
                                        "text_output_base": maina_before,
                                        "api_result_text_base": api_result_text,
                                        "logprobs_contents_base": [{"token": lc["token"], "logprob": lc["logprob"]} for lc in logprobs_contents],
                                    }

                                elif args.chatgpt_mode == "maina_after":
                                    logprobs_contents = json_object['response']['body']['choices'][0]['logprobs']["content"]
                                    maina_after = ""
                                    confidence_score = 0.
                                    for logprobs_content in logprobs_contents:
                                        logprob = logprobs_content["logprob"]
                                        confidence_score += logprob
                                        maina_after += logprobs_content["token"]
                                        # print(logprobs_content["token"], '\t', logprobs_content["logprob"])
                                        if maina_after in ["A", "B", "C", "D", "E", "(A)", "(B)", "(C)", "(D)", "(E)"]:
                                            break
                                    if maina_after in "ABCDE":
                                        maina_after = "(" + maina_after + ")"

                                    # text_input
                                    i = int(json_object['custom_id'].split('_')[-1])
                                    sq_list, sa_list = [], []
                                    for j in range(num_sub_qa_select):
                                        s_idx = (i+j) % 5
                                        sq_list.append(data['sub_question_list'][s_idx])
                                        sa_list.append(data['sub_answer_list'][s_idx])

                                    candidate_list = data["candidate_list"] if "candidate_list" in data else None
                                    results[question_id]["text_input"].append(get_maina_after_prompt(data["text_input"], sq_list, sa_list, open_ended, candidate_list))

                                    results[question_id]["confidence_lba"].append(confidence_score)
                                    results[question_id]["text_output_lba"].append(maina_after)
                                    results[question_id]["api_result_text_lba"].append(api_result_text)
                                    results[question_id]["logprobs_contents_lba"].append([{"token": lc["token"], "logprob": lc["logprob"]} for lc in logprobs_contents])

                                    # results[question_id] = {
                                    #     "question_id": question_id,
                                    #     "text_input": get_maina_after_prompt(data["text_input"], data["candidate_list"], data['sub_question_list'], data['sub_answer_list']),
                                    #     "main_question": data["text_input"],
                                    #     "gt_ans": dataset.answer_mapping(data["gt_ans"]),
                                    #     "confidence_lba": confidence_score,
                                    #     "text_output_lba": maina_after,
                                    #     "api_result_text_lba": api_result_text,
                                    #     "logprobs_contents_lba": [{"token": lc["token"], "logprob": lc["logprob"]} for lc in logprobs_contents],
                                    # }


                            #     custom_id = json_object['custom_id']
                            #     subq_i = int(custom_id.split('_')[-1])
                            #     temp_subqa.append((subq_i, (sub_questions[question_id][subq_i], api_result_text)))
                            # results_subqa[question_id] = [subqa for _, subqa in sorted(temp_subqa)]

                    # error results
                    error_file_id = batch_job.error_file_id
                    if error_file_id is not None:
                        print(f"error_file_id: {error_file_id}")
                        api_result = client.files.content(error_file_id).content
                        error_file_name = os.path.join(root_dir, f"{args.chatgpt_mode}/select{num_sub_qa_select}/error_results/{question_id}.jsonl")

                        with open(error_file_name, 'wb') as file:
                            file.write(api_result)

                        with open(error_file_name, 'r') as file:
                            for line in file:
                                # Parsing the JSON string into a dict and appending to the list of results
                                json_object = json.loads(line.strip())
                                pprint(json_object, width=300)

            except:
                print(f"question_id: {question_id} failed.")
                import traceback
                traceback.print_exc()
                sleep(60)
                success = False

    if args.chatgpt_mode == "maina_after":
        results_path = os.path.join(root_dir, f"chatgpt_result_{dataset_name}_{args.chatgpt_mode}_select{num_sub_qa_select}.json")
    else:
        results_path = os.path.join(root_dir, f"chatgpt_result_{dataset_name}_{args.chatgpt_mode}.json")

    json.dump(results, open(results_path, "w"), indent=4)
    print(f"\nresults saved to {results_path} . number of completed: {len(results)} / {len(dataset)}")
    print(f"completed: {completed}, failed: {failed}, expired: {expired}, total: {total}")
    print(f"fatal_error_list: {fatal_error_list}")


    # json.dump(results_subqa, open(f"{root_dir}/subqa_val_chatgpt.json", "w"), indent=4)
    # print(f"subqa_val_chatgpt.json saved to {root_dir}/subqa_val_chatgpt.json .")


if __name__ == '__main__':
    # import pdb; pdb.set_trace()

    main()
