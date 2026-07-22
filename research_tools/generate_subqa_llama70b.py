import os
import argparse
import json
from tqdm import tqdm

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Blip2Processor, InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration


from models.model import VideoBlip2ForConditionalGeneration
from dataset.base_dataset import load_dataset
from configs.config import Config
from main_multi_subqa import parse_args, setup_seeds



def get_input(processor, device, video_batch, text_inputs):
    inputs = processor(text=text_inputs, return_tensors="pt", padding=True)

    pixel_values = []
    for video in video_batch: # video: [n_frms, 640, 480]
        pixel_values.append(processor(images=video, return_tensors="pt", padding=True)['pixel_values'])  # [n_frms, 3, 224, 224]
    inputs["pixel_values"] = torch.stack(pixel_values, dim=0)#.to(device)
    inputs = inputs.to(device)
    print("input_ids:", inputs["input_ids"].shape, inputs["input_ids"].device, "\tpixel_values:", inputs["pixel_values"].shape, inputs["pixel_values"].device)

    return inputs


"""
pre-generate sub-qa pairs for each question in the dataset
usage:
CUDA_VISIBLE_DEVICES=4 python generate_subqa.py --options datasets.dataset_name="DramaQA" runner.batch_size=12
"""

def main():
    N_SUBQA = 5
    N_SUPPLE = 5
    args = parse_args()
    cfg = Config(args)
    setup_seeds(cfg)
    # cache_dir = "/data4/Very_Large_Models/"
    # model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    if cfg.runner_cfg.llama_model == '8b':
        cache_dir = "/data2/llama/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/07eb05b21d191a58c577b4a45982fe0c049d0693/"#models--meta-llama--Meta-Llama-3.1-8B-Instruct"
    else:
        cache_dir = "/data2/llama/models--meta-llama--Meta-Llama-3.1-70B-Instruct/snapshots/168b5b277b07436c229bd69844a73872eb8b73a8/"#models--meta-llama--Meta-Llama-3.1-8B-Instruct"
    # model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    model_id = cache_dir #"/data2/llama/models--meta-llama--Meta-Llama-3.1-8B-Instruct/"

    # os.environ['HF_HOME'] = cache_dir

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        device_map="auto",
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        # quantization_config=quantization_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     model.resize_token_embeddings(len(tokenizer))

    input_text = "What are we having for dinner?"
    input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

    output = model.generate(**input_ids, max_new_tokens=100)

    print(tokenizer.decode(output[0], skip_special_tokens=True))

    # ------------------------------------------------------------------------------------------------

    dataset = load_dataset(cfg.datasets_cfg, n_supple=N_SUPPLE)
    dataloader = DataLoader(dataset, batch_size=cfg.runner_cfg.batch_size,
                            shuffle=False, collate_fn=dataset.collater)

    results = {}
    root_dir = f"temp/llama70b_result/{cfg.datasets_cfg.dataset_name}"
    os.makedirs(root_dir, exist_ok=True)

    for data_iter_step, batch in tqdm(enumerate(dataloader)):
        question_ids = batch["question_id"]
        # if question_ids[0] not in ['TVQA_4761', 'TVQA_4802', 'TVQA_5581', 'TVQA_12443', 'TVQA_12852']:
        #     print(question_ids[0], 'passed')
        #     continue
        # else:
        #     print(question_ids[0], 'processing')
        bsz = len(batch["question_id"])

        # prompt = "Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {main_question}? Perception Question:"
        prompt = """You are an AI assistant who has rich video understanding abilities.
You will be provided with:
main_question: {main_question}
and video frames.

Your goal is:
To effectively analyze the video and answer the main_question, you should break down the main_question into several sub-questions that address the key aspects of the video.
You have to generate 5 sub_questions which those sub_questions can drive to the given main_question. sub_questions should be about the entire video.
Make sure that your sub_questions are based on the information you have.

Format: in list of dictionary format = ["sub_question_1",..., "sub_question_5"]

Geenrate sub_questions only: """
        text_inputs = [prompt.format(main_question=main_question.rstrip('?')) for main_question in batch["text_input"]]

        # print("Generating sub-q...")

        input_ids = tokenizer(text_inputs, return_tensors="pt", padding=True).to("cuda")
        output = model.generate(**input_ids, max_new_tokens=500, pad_token_id=tokenizer.eos_token_id)
        sub_questions = tokenizer.batch_decode(output, skip_special_tokens=True)


        if data_iter_step == 0:
            for sub_quesiton, main_question in zip(sub_questions, batch["text_input"]):
                print(f'Main Question: {main_question:100s}, Sub Question: {sub_quesiton}')

        for b in range(bsz):
            results[question_ids[b]] = sub_questions[b].replace(text_inputs[b], '')
#             if question_ids[b] not in results:
#                 results[question_ids[b]] = set()

#             if sub_questions[b].endswith('?'):
#                 results[question_ids[b]].add(sub_questions[b])

#         if data_iter_step == 0:
#             import pdb; pdb.set_trace()

    json.dump(results, open(os.path.join(root_dir, "results_add.json"), "w"), indent=4)



if __name__ == '__main__':
    main()
