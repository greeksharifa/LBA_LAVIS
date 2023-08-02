import os
import glob

from tqdm import tqdm
from PIL import Image

import torch

import warnings
warnings.filterwarnings(action="ignore")

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open("docs/_static/Confusing-Pictures.jpg").convert("RGB")

from lavis.models import load_model_and_preprocess
# loads InstructBLIP model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

print(model.generate({"image": image, "prompt": "Write a detailed description."}))

import pdb; pdb.set_trace()
