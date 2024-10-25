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


def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])



def video_llava_demo():
    import av
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration

    model_name = "LanguageBind/Video-LLaVA-7B-hf"
    processor = VideoLlavaProcessor.from_pretrained(model_name)
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        model_name, 
        device_map="auto",
        attn_implementation=None,
    )



    video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    videos = read_video_pyav(container, indices)

    inputs = processor(videos=videos, text="USER: <video>\nWhat do you see here? ASSISTANT:", return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        num_beams=5,
        max_new_tokens=40,
        min_length=1,
        length_penalty=-1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    output_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    print(output_text)
    output_scores = torch.exp(outputs.sequences_scores).tolist()


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
