import os
import argparse
import json

import torch
from torch.utils.data import DataLoader
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
    model_name = "Salesforce/blip2-flan-t5-xl"
    processor_name = model_name
    # processor_name = "Salesforce/instructblip-flan-t5-xl"
    # model_name = processor_name
    cache_dir = os.path.join("/data2/", model_name.split("/")[0])
    device = "cuda"

    processor = Blip2Processor.from_pretrained(processor_name, cache_dir=cache_dir)
    model = VideoBlip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(device)#, device_map="auto")
    # processor = InstructBlipVideoProcessor.from_pretrained(processor_name, cache_dir=cache_dir)
    # model = InstructBlipVideoForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir).to(device)#, device_map="auto")

    dataset = load_dataset(cfg.datasets_cfg, n_supple=N_SUPPLE)
    dataloader = DataLoader(dataset, batch_size=cfg.runner_cfg.batch_size,
                            shuffle=False, collate_fn=dataset.collater)

    results = {}

    for data_iter_step, batch in enumerate(dataloader):
        question_ids = batch["question_id"]
        bsz = len(batch["question_id"])

        prompt = "Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {main_question}? Perception Question:"
        text_inputs = [prompt.format(main_question=main_question.rstrip('?')) for main_question in batch["text_input"]]

        inputs_list = [get_input(processor, device, batch["vision"], text_inputs)]
        for i in range(N_SUPPLE):
            video_batch = []
            for b in range(bsz):
                # print(i, b)
                # print(len(batch["vision_supple"]), len(batch["vision_supple"][i]))
                video_batch.append(batch["vision_supple"][b][i])

            inputs_list.append(get_input(processor, device, video_batch, text_inputs))

        # inputs = processor(text=text_inputs, images=batch["vision"], return_tensors="pt", padding=True).to(device)

        # import pdb; pdb.set_trace()
        generation_params = {
            "do_sample": True,
            "num_beams": 5,
            "top_p": 0.999,
            "max_new_tokens": 100,
            "min_length": 1,
            "length_penalty": -1,
            # "return_dict_in_generate": True,
            # "output_scores": True
        }

        while True:
            cnt = 0
            print("Generating sub-q...")
            min_n_subqa = N_SUBQA
            sum_n_subqa = 0
            # for i in range(N_SUBQA):
            outputs = model.generate(**inputs_list[cnt % len(inputs_list)], **generation_params)
            sub_questions = processor.batch_decode(outputs, skip_special_tokens=True)

            for sub_quesiton, main_question in zip(sub_questions, batch["text_input"]):
                print(f'Main Question: {main_question:100s}, Sub Question: {sub_quesiton}')

            for b in range(bsz):
                if question_ids[b] not in results:
                    results[question_ids[b]] = set()

                if sub_questions[b].endswith('?'):
                    results[question_ids[b]].add(sub_questions[b])

                min_n_subqa = min(min_n_subqa, len(results[question_ids[b]]))
                sum_n_subqa += len(results[question_ids[b]])

            print(results)
            print(f"Min number of sub-qa: {min_n_subqa}")
            print(f"Average number of sub-qa: {sum_n_subqa/bsz}")
            if min_n_subqa >= N_SUBQA:
                break

            results_json = results.copy()
            for k, v in results_json.items():
                results_json[k] = list(v)
            json.dump(results_json, open("temp/subqa.json", "w"), indent=4)


        # for k, v in results.items():
        #     results[k] = list(v)

        break

        for j in range(N_SUBQA):
            print(f"Question {i*N_SUBQA+j}: {processor.decode(outputs[j])}")




if __name__ == '__main__':
    main()
