from typing import Optional
import requests
from PIL import Image
import os

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import InstructBlipVideoImageProcessor, InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration
from accelerate import infer_auto_device_map
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info

# from utils.api_chatgpt import call_vision_api, ndarrays_to_base64
from utils.utils import ndarrays_to_base64

import numpy as np
import torch
from torch import nn



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
