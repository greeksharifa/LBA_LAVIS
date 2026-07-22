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

import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration

from openai import OpenAI

from models.model import VideoBlip2ForConditionalGeneration
from dataset.base_dataset import load_dataset
from configs.config import Config
from main_multi_subqa import setup_seeds
from utils.api_chatgpt import call_vision_api, call_chat_api
from temp.line_message import line_notify



def get_input(processor, device, video_batch, text_inputs):
    inputs = processor(text=text_inputs, return_tensors="pt", padding=True)

    pixel_values = []
    for video in video_batch: # video: [n_frms, 640, 480]
        pixel_values.append(processor(images=video, return_tensors="pt", padding=True)['pixel_values'])  # [n_frms, 3, 224, 224]
    inputs["pixel_values"] = torch.stack(pixel_values, dim=0)#.to(device)
    inputs = inputs.to(device)
    print("input_ids:", inputs["input_ids"].shape, inputs["input_ids"].device, "\tpixel_values:", inputs["pixel_values"].shape, inputs["pixel_values"].device)

    return inputs


def parse_args():
    parser = argparse.ArgumentParser(description='LBA method')
    parser.add_argument("--cfg-path", default='configs/runner.yaml', help="path to configuration file.")
    parser.add_argument('--verbose', action='store_true', help='verbose')

    parser.add_argument('--subq', action='store_true', help='generate sub-q')
    parser.add_argument('--suba', action='store_true', help='generate sub-a')

    parser.add_argument('--chatgpt_mode', type=str, choices=["subq", "suba", "maina_before", "maina_after"])

    parser.add_argument('--chatgpt_model', type=str, default="gpt-4o-mini", help='OpenAI model')
    parser.add_argument('--chatgpt_max_tokens', type=int, default=500, help='OpenAI max tokens')
    parser.add_argument('--chatgpt_temperature', type=float, default=0., help='OpenAI temperature')
    parser.add_argument('--chatgpt_vision_detail', type=str, default="low", choices=["low", "high"])
    parser.add_argument('--chatgpt_verify', action='store_true', help='verify the generated questions')


    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    return args


def ndarrays_to_base64(frame_list):
    base64_frames = []

    for frame in frame_list:
        # Ensure the frame is in uint8 format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)

        # Convert the buffer to base64
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        base64_frames.append(jpg_as_text)

    return base64_frames


def base64_to_ndarrays(base64_frames):
    decoded_frames = []

    for encoded in base64_frames:
        # Decode base64 string to bytes
        jpg_original = base64.b64decode(encoded)

        # Use numpy to create a 1D array of unsigned bytes
        jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)

        # Decode the JPEG to a numpy array
        frame = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)

        decoded_frames.append(frame)

    return decoded_frames


def get_prompt(main_question):
    #     prompt = """Generate 5 perception questions based on given images, like following 2 examples:
    # Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?
    # Reasoning Question: is it cold outside? Perception Question: are any people wearing jackets?
    # Reasoning Question: {main_question}? Perception Question:"""

    prompt = """You are an AI assistant who has rich video understanding abilities.
You will be provided with:
main_question: {main_question}
and video frames.

Your goal is:
To effectively analyze the video and answer the main_question, you should break down the main_question into several sub-questions that address the key aspects of the video.
You have to generate 5 sub_questions which those sub_questions can drive to the given main_question. sub_questions should be about the entire video.
Make sure that your sub_questions are based on the information you have.

Format: in list of dictionary format = ["sub_question_1", ..., "sub_question_5"]

sub_questions: """

    return prompt.format(main_question=main_question.rstrip('?'))


reObj1 = re.compile(r'"(.+\?\'?)"')
reObj2 = re.compile(r'\d\. (.+\?\'?)')
def reformat_subq(text):
    result = reObj1.findall(text)
    if len(result) == 5 and all([r[-1] in "?'" for r in result]):
        return result

    result = reObj2.findall(text)
    if len(result) == 5 and all([r[-1] in "?'" for r in result]):
        return result

    return None

"""
Here are the sub-questions based on the provided video frames:

```json
[
    "What is the context of the conversation between the characters in the first two frames?",
    "How do the characters' expressions and body language indicate their feelings or reactions?",
    "What is the significance of the setting in the third frame, and how does it relate to the characters' interactions?",
    "What actions or dialogue occur just before the character enters the room in the fourth frame?",
    "How do the characters' clothing and colors contribute to the overall mood of the scenes?"
]

# -----------------------------------------------------------------------------------------------------------------------------------------------
Here are the sub-questions based on the provided video frames:

1. What is the context or setting of the event taking place in the video?
2. How are the characters interacting with each other in the frames?
3. What emotions or reactions are displayed by the characters in the video?
4. Are there any significant objects or elements in the environment that could influence the next event?
5. What prior events or conversations might have led to the current situation depicted in the video?

These sub-questions can help analyze the video and determine which event is more likely to happen next.
"""

def reformat_subq_backup(text):
    pattern = ''#r'.*'
    for i in range(1, 6):
        pattern += rf'{i}\. \[?(.+\?)\]?.*'
        # pattern += f'{i}' + r'\. (?:\'\"\[)*(.+\?)(?:\'\"\])*.*'
        if i != 5:
            pattern += '\n'
    reObj = re.compile(pattern)
    import pdb; pdb.set_trace()

    # reObj = re.compile(r'1\. (.+\?).*\n2\. (.+\?)\n3\. (.+\?)\n4\. (.+\?)\n5\. (.+\?).*')
    matchObj = reObj.search(text)
    if matchObj:
        return [matchObj.group(i) for i in range(1, 6)]
    else:
        return None


def api_result_to_json(api_result):
    result_json = json.loads(api_result.json())
    for k, v in result_json.items():
        if k.endswith("_at") and v:
            print(f'{k:20s}: {datetime.datetime.fromtimestamp(v)}')
            result_json[k] = datetime.datetime.fromtimestamp(v)
        elif v is not None or "errors" in k:
            print(f'{k:20s}: {v}')
        else:
            pass

    return result_json


reObj3 = re.compile(r'"(.+\'?)"')
reObj4 = re.compile(r'\d[\.:] (.+)')
def reformat_suba(text):
    result = reObj3.findall(text)
    if len(result) == 5:
        return result

    result = reObj4.findall(text)
    if len(result) == 5:
        return result

    return None

def main():
    """
    pre-generate sub-qa pairs for each question in the dataset
    usage:
    CUDA_VISIBLE_DEVICES=4 python generate_subqa_chatgpt.py --subq --options datasets.dataset_name="NExTQA" runner.batch_size=1
    CUDA_VISIBLE_DEVICES=4 python generate_subqa_chatgpt.py --suba --options datasets.dataset_name="NExTQA" runner.batch_size=1
    # verify
    CUDA_VISIBLE_DEVICES=4 python generate_subqa_chatgpt.py --chatgpt_verify --options datasets.dataset_name="NExTQA" runner.batch_size=1
    """
    N_SUBQA = 5
    N_SUPPLE = 5
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)

    # dataset
    if cfg.runner_cfg.batch_size != 1:
        print("runner.batch_size must be 1. Automatically set to 1.")
    dataset = load_dataset(cfg.datasets_cfg, n_supple=N_SUPPLE)
    # dataloader = DataLoader(dataset, batch_size=cfg.runner_cfg.batch_size,
    #                         shuffle=False, collate_fn=dataset.collater)

    # openai
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    batch_input_file = client.files.create(
        file=open("temp/batchinput.jsonl", "rb"),
        purpose="batch"
    )

    # generate result
    api_results = {}
    results = {}
    root_dir = f"/data1/chatgpt_result/{cfg.datasets_cfg.dataset_name}"
    os.makedirs(root_dir, exist_ok=True)

    format_error_list = []
    fatal_error_list = []
    file_not_exist_list = []
    question_set = set()



    if args.chatgpt_mode == "subq":
        for data_iter_step, data in enumerate(dataset):
            question_id = data["question_id"]
            question_set.add(question_id)

            if args.chatgpt_verify:
                print(f"\r{data_iter_step+1:6d}/{len(dataset):6d} : question_id={str(question_id):10s} verifying...", end='\t')
                json_path = os.path.join(root_dir, f"{question_id}.json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        if os.path.getsize(json_path) == 0:
                            fatal_error_list.append(question_id)
                            print('file empty')
                        else:
                            try:
                                """
                                try:
                                    json_data = json.load(f)
                                except:
                                    json_data = f.read()
                                if type(json_data) == list and len(json_data) == 5:
                                    results[question_id] = json_data
                                elif reformat_subq(json_data):
                                    results[question_id] = reformat_subq(json_data)
                                    # json.dump(results[question_id], open(json_path, "w"), indent=4)
                                """
                                json_data = f.read()
                                if reformat_subq(json_data):
                                    results[question_id] = reformat_subq(json_data)
                                else:
                                    raise Exception(f'format error. question_id={question_id}, data_iter_step={data_iter_step}\njson_data={json_data}')
                                print('verified', end='')
                            except:
                                format_error_list.append(question_id)
                                print('\nformat error')
                                print('question_id:', question_id)
                                print('json_data:', )
                                pprint(json_data, width=300)
                                import pdb; pdb.set_trace()
                                results[question_id] = None
                else:
                    print(f'file not exist: {question_id}')
                    file_not_exist_list.append(question_id)
            else:
                json_path = os.path.join(root_dir, f"{question_id}.json")
                # if os.path.exists(json_path) and os.path.getsize(json_path) > 0 and json.load(open(json_path, "r")) != []:
                if os.path.exists(json_path) and reformat_subq(open(json_path, "r").read()):
                    print(f"\r{data_iter_step+1:6d}/{len(dataset):6d} : question_id={str(question_id):10s} already exists.", end='\t')
                    continue
                try:
                    print(f"\r{data_iter_step+1:6d}/{len(dataset):6d} : question_id={str(question_id):10s} creating...", end='\n')
                    if os.path.exists(json_path):
                        print('file data:', open(json_path, "r").read())
                    else:
                        print(f'json_data: file {json_path} is not exist')
                    # if data_iter_step < 1377: continue
                    # if question_id != 9933831: continue # 3226 9923067

                    video = data["vision"] # list of ndarray. [n_frms, H, W, 3]
                    text_input = get_prompt(data["text_input"])
                    base64_frames = ndarrays_to_base64(video)
                    # decoded_video = base64_to_ndarrays(base64_frames)

                    for _ in range(5):
                    # num_try = 5
                    # while num_try > 0:
                        api_result = call_vision_api(args, prompt=text_input, base64_frames=base64_frames)
                        if 'error' in api_result:
                            line_notify(f"vision api error, retrying...question_id={question_id} : {api_result['error']}")
                        else:
                            try:
                                # if api_result['choices'][0]['message']['content'] != []:
                                if reformat_subq(api_result['choices'][0]['message']['content']):
                                    break
                            except:
                                line_notify(f"result is ridiculous, retrying...question_id={question_id}")
                        sleep(5)
                        # num_try -= 1
                    print('content:', api_result['choices'][0]['message']['content'], sep='\n')

                    api_results[question_id] = api_result
                    results[question_id] = api_result['choices'][0]['message']['content']

                    with open(os.path.join(root_dir, f"{question_id}.json"), "w") as f:
                        sub_questions = reformat_subq(api_result['choices'][0]['message']['content'])
                        if sub_questions is None:
                            import pdb; pdb.set_trace()
                            format_error_list.append(question_id)
                            sub_questions = api_result['choices'][0]['message']['content']
                        json.dump(sub_questions, f, indent=4)
                        results[question_id] = sub_questions

                    with open(f"{root_dir}/api_results.txt", "a") as f:
                        f.write(f"{question_id}: {api_result}\n")


                    if data_iter_step % 100 == 0:
                        line_notify(f"generate_subqa_chatgpt.py: {data_iter_step+1:6d}/{len(dataset):6d} completed.")
                    # if data_iter_step >= 1: break
                    # if question_id == 9933831: break
                except Exception as e:
                    format_error_list.append(question_id)
                    print('\n', e)
                    import pdb; pdb.set_trace()
                    pprint(api_result, width=300)
                    line_notify(f"generate_subqa_chatgpt.py failed.\n{str(e)}")

        print("\n")
        if args.chatgpt_verify:
            print(f'len(results): {len(results):6d} | len(question_set): {len(question_set):6d}')
            print('format_error_list:', format_error_list)
            print('fatal_error_list:', fatal_error_list)
            json.dump(file_not_exist_list, open(f"{root_dir}/file_not_exist_list.json", "w"), indent=4)
            json.dump(results, open(f"{root_dir}/sub_questions_val_chatgpt.json", "w"), indent=4)
            print(f"sub_questions_val_chatgpt.json saved to {root_dir}/sub_questions_val_chatgpt.json .")
        else:
            json.dump(api_results, open(f"{root_dir}/api_results_subq.json", "a"), indent=4)
            # json.dump(results, open(f"{root_dir}/results.json", "w"), indent=4)
            json.dump(format_error_list, open(f"{root_dir}/format_error_list.json", "w"), indent=4)
            print("format_error_list:", format_error_list)
    elif args.chatgpt_mode == "suba":
        os.makedirs(os.path.join(root_dir, f"suba"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, f"suba/batch_inputs"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, f"suba/batch_results"), exist_ok=True)

        results_subqa = {}

        batch_job_ids_path = os.path.join(root_dir, f"suba/batch_job_ids.json")

        sub_questions = json.load(open(f"{root_dir}/sub_questions_val_chatgpt.json", "r"))

        # while True:

        if os.path.exists(batch_job_ids_path):
            batch_job_ids = json.load(open(batch_job_ids_path, "r"))
        else:
            batch_job_ids = {}

        sub_answers_path = os.path.join(root_dir, f"sub_answers_val_chatgpt_raw.json")
        if os.path.exists(sub_answers_path):
            results_suba = json.load(open(sub_answers_path, "r"))
        else:
            results_suba = {}

        for data_iter_step, data in enumerate(tqdm(dataset)):
            if args.chatgpt_verify:
                # if data_iter_step >= 30: break
                question_id = data["question_id"]

                if question_id in results_suba and type(results_suba[question_id]) == list and len(results_suba[question_id]) == 5:
                    print(f"question_id: {question_id} sub_answer already exists.")
                    continue

                print(f"\r{data_iter_step+1:6d}/{len(dataset):6d} : question_id={str(question_id):10s} sub_answer creating...", end='\t')

                video = data["vision"]
                base64_frames = ndarrays_to_base64(video)
                from utils.api_chatgpt import call_vision_api
                prompt = ""
                for subq_i, sub_question in enumerate(sub_questions[question_id]):
                    prompt += f"Question {subq_i+1}: {sub_question}\n"
                res = call_vision_api(args, prompt=prompt, base64_frames=base64_frames, system_prompt="You are a helpful video understanding assistant. Based on the video frames, answer the following questions. Important: Print answers only, one per one line. Do not refuse to answer.\n")
                # print(res)
                try:
                    sub_answer = res['choices'][0]['message']['content']
                    pprint(sub_answer, width=300)
                    formatted_suba = reformat_suba(sub_answer)
                    if type(formatted_suba) == list and len(formatted_suba) == 5:
                        results_suba[question_id] = formatted_suba
                    else:
                        results_suba[question_id] = sub_answer
                except:
                    print(f"question_id: {question_id} sub_answer is not valid.")
                    pass

            else:
                if data_iter_step >= 462: break

                question_id = data["question_id"]
                question_set.add(question_id)

                batch_inputs_path = os.path.join(root_dir, f"suba/batch_inputs/{question_id}.jsonl")

                if question_id in batch_job_ids:
                    batch_job_id = batch_job_ids[question_id]
                    print(f"question_id: {question_id} is already in batch_job_ids with batch_job_id: {batch_job_id}")
                else:
                    if not os.path.exists(batch_inputs_path):
                        video = data["vision"]
                        base64_frames = ndarrays_to_base64(video)

                        content = []
                        for base64_frame in base64_frames:
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_frame}",
                                    "detail": args.chatgpt_vision_detail,
                                }
                            })
                        prompt = ""
                        for subq_i, sub_question in enumerate(sub_questions[question_id]):
                            prompt += f"Question {subq_i+1}: {sub_question}\n"
                        content.append({
                            "type": "text",
                            "text": prompt
                        })
                        batch_input = {
                            "custom_id": question_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": "gpt-4o-mini",
                                "messages": [
                                    {"role": "system", "content": "You are a helpful video understanding assistant. Based on the video frames, answer the following questions. Important: Print answers only, one per one line.\n"},
                                    {"role": "user", "content": content}
                                ],
                                "max_tokens": 500
                            }
                        }
                        with open(batch_inputs_path, "w") as f:
                            f.write(json.dumps(batch_input) + '\n')

                    # openai batch
                    batch_file = client.files.create(
                        file=open(batch_inputs_path, "rb"),
                        purpose="batch"
                    )
                    print(batch_file)
                    print(batch_file.id)

                    sleep(5)
                    batch_job = client.batches.create(
                        input_file_id=batch_file.id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                        metadata={
                            "dataset_name": cfg.datasets_cfg.dataset_name,
                            "question_id": question_id
                        }
                    )
                    batch_job_id = batch_job.id
                    batch_job_ids[question_id] = batch_job_id
                    json.dump(batch_job_ids, open(batch_job_ids_path, "w"), indent=4)

                batch_job = client.batches.retrieve(batch_job_id)
                result_json = api_result_to_json(batch_job)
                # print(batch_job)

                result_file_id = batch_job.output_file_id
                if result_file_id is None:
                    print(f"result_file_id is None. question_id={question_id}")
                    continue
                else:
                    print(f"result_file_id: {result_file_id}")
                    api_result = client.files.content(result_file_id).content

                    result_file_name = os.path.join(root_dir, f"suba/batch_results/{question_id}.jsonl")

                    with open(result_file_name, 'wb') as file:
                        file.write(api_result)

                    # Loading data from saved file
                    with open(result_file_name, 'r') as file:
                        # results_suba[question_id] = []
                        # results_subqa[question_id] = []
                        # temp_subqa = []
                        for line in file:
                            # Parsing the JSON string into a dict and appending to the list of results
                            json_object = json.loads(line.strip())
                            sub_answer = json_object['response']['body']['choices'][0]['message']['content']
                            formatted_suba = reformat_suba(sub_answer)
                            if formatted_suba:
                                results_suba[question_id] = formatted_suba
                            else:
                                results_suba[question_id] = sub_answer
                        #     custom_id = json_object['custom_id']
                        #     subq_i = int(custom_id.split('_')[-1])
                        #     temp_subqa.append((subq_i, (sub_questions[question_id][subq_i], sub_answer)))
                        # results_subqa[question_id] = [subqa for _, subqa in sorted(temp_subqa)]


        json.dump(results_suba, open(f"{root_dir}/sub_answers_val_chatgpt_raw.json", "w"), indent=4)
        print(f"sub_answers_val_chatgpt.json saved to {root_dir}/sub_answers_val_chatgpt_raw.json .")
        # json.dump(results_subqa, open(f"{root_dir}/subqa_val_chatgpt.json", "w"), indent=4)
        # print(f"subqa_val_chatgpt.json saved to {root_dir}/subqa_val_chatgpt.json .")
    elif args.chatgpt_mode.startswith("maina"):
        os.makedirs(os.path.join(root_dir, f"{args.chatgpt_mode}"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, f"{args.chatgpt_mode}/batch_inputs"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, f"{args.chatgpt_mode}/batch_results"), exist_ok=True)

        results_subqa = {}

        batch_job_ids_path = os.path.join(root_dir, f"{args.chatgpt_mode}/batch_job_ids.json")

        for data_iter_step, data in enumerate(tqdm(dataset)):
            question_id = data["question_id"]
            question_set.add(question_id)



    else:
        raise ValueError("Please specify chatgpt_mode.")

if __name__ == '__main__':
    main()
