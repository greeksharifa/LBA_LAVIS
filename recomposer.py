# pip install accelerate
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to("cuda")#, device_map="auto")
device = model.device

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

questions = ["Question: How many dogs are in the picture?", "Question: How many ships are in the picture?"]
inputs = processor([raw_image, raw_image], questions, return_tensors="pt", padding=True).to("cuda")#, torch.float16)

out = model.generate(**inputs)
print(out)
print(processor.batch_decode(out, skip_special_tokens=True))
# print('device:', device)
# print('model.device:', model.device)
