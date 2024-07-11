import requests
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import torch
from torch import nn


class Decomposer(nn.Module):
    def __init__(self, model_name, device="cuda"):
        super().__init__()
        
        self.device = device
        self.decomposer_name = f'google/flan-t5-{model_name}'
        self.decomposer_tokenizer = T5Tokenizer.from_pretrained(self.decomposer_name)
        self.decomposer_model = T5ForConditionalGeneration.from_pretrained(
            self.decomposer_name, 
            torch_dtype=torch.bfloat16,
            # load_in_4bit=True,
            # device_map="auto",
        ).to(device)

    def forward(self, text_inputs):
        input_ids = self.decomposer_tokenizer(text_inputs, padding="longest", return_tensors="pt").input_ids.to(self.device)
        # outputs = self.decomposer_model.generate(input_ids)
        outputs = self.decomposer_model.generate(input_ids, num_beams=5, do_sample=True, top_p=0.95, temperature=1.0, length_penalty=1.0, repetition_penalty=1.0, max_new_tokens=50)
        sub_questions = self.decomposer_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return sub_questions


class Recomposer(nn.Module):
    def __init__(self, model_name, device="cuda"):
        super().__init__()
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def forward(self, images, text_inputs):
        inputs = self.processor(images, text_inputs, return_tensors="pt", padding=True).to(self.device)
        # out = self.model.generate(**inputs)
        # return self.processor.batch_decode(out, skip_special_tokens=True)

        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_new_tokens=10,
            min_length=1,
            length_penalty=-1,
            return_dict_in_generate=True,
            output_scores=True,
        )
        # <class 'transformers.generation.utils.BeamSearchEncoderDecoderOutput'>
        # odict_keys(['sequences', 'sequences_scores', 'scores', 'beam_indices'])

        output_text = self.processor.batch_decode(
            outputs.sequences, skip_special_tokens=True
        )
        output_scores = torch.exp(outputs.sequences_scores).tolist()

        return output_text, output_scores


def demo():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to("cuda")  # , device_map="auto")
    device = model.device

    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    questions = ["Question: How many dogs are in the picture?",
                 "Question: How many ships are in the picture?"]
    inputs = processor([raw_image, raw_image], questions,return_tensors="pt", padding=True).to("cuda")  # , torch.float16)

    out = model.generate(**inputs)
    print(out)
    print(processor.batch_decode(out, skip_special_tokens=True))
