import os

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import AutoTokenizer

# Download instruct_blip_vicuna7b_trimmed.pth
# wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth

ROOT_DIR = "/home/ywjang/models/"
# ROOT_DIR = "/Users/sharifa/models/"

def download_HF_model(model_ckpt):
    print('Downloading {}...'.format(model_ckpt))
    os.makedirs(f"{ROOT_DIR}{model_ckpt}/", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    tokenizer.save_pretrained(f"{ROOT_DIR}{model_ckpt}/")
    
    model = Blip2ForConditionalGeneration.from_pretrained(model_ckpt, device_map="auto")#, offload_folder="offload")
    model.save_pretrained(f"{ROOT_DIR}{model_ckpt}/")
    print('Done.')


# Download blip2-opt-2.7b
download_HF_model("Salesforce/blip2-opt-2.7b")

# Download vicuna-7b-v1.3
download_HF_model("lmsys/vicuna-7b-v1.3")
