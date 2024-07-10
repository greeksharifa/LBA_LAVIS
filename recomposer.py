import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import torch
from torch import nn


class Recomposer(nn.Module):
    def __init__(self, model_name, device="cuda"):
        super().__init__()
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def forward(self, images, questions):
        inputs = self.processor(images, questions, return_tensors="pt", padding=True).to(self.device)
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
        output_scores = torch.exp(outputs.sequences_scores)

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
