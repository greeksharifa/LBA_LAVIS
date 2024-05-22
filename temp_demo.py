import torch
from PIL import Image
# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
# load sample image
raw_image = Image.open("docs/_static/Confusing-Pictures.jpg").convert("RGB")
# display(raw_image.resize((596, 437)))

from lavis.models import load_model_and_preprocess
# loads InstructBLIP model
# model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5_instruct", model_type="flant5xl", is_eval=True, device=device)
# prepare the image
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

# print(model.generate({"image": image, "prompt": "What is unusual about this image?"}))
print(model.generate({"image": image, "main_question": ["What is unusual about this image?"]}))
'''
print(model.generate({"image": image.expand(4, -1, -1, -1), "main_question": [
        "What is unusual about this image?",
        "What is the man doing on top of the car?",
        "What is the model name of car?",
        """ You have to answer the following instructions.
"Main-question" is `{main_Q}`.

1. What is the insufficent information to answer the `{main_Q}`?
2. Generate one "Sub-question" about insufficent information to get more information.
3. What is the answer to "Sub-question"?
4. What is the answer to `{main_Q}`?""".format(main_Q="What is unusual about this image?")
     ]}))
'''
# print(model.generate({"image": image, "prompt": "What is unusual about this image?"}))
# print(model.generate({"image": image, "prompt": "Write a short description for the image."}))
# print(model.generate({"image": image, "prompt": "Write a detailed description."}))
# print(model.generate({"image": image, "prompt":"Describe the image in details."}, use_nucleus_sampling=True, top_p=0.9, temperature=1))
# ['The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.']
# ['a man in a yellow shirt is standing on top of a car']
# ["A man in a yellow shirt is standing on the back of a yellow SUV parked on a busy city street. He is holding an ironing board and appears to be ironing clothes while standing on the vehicle's tailgate. There are several other cars and trucks visible in the background, adding to the bustling atmosphere of the scene. The man's presence on the back of the SUV creates a unique and creative way for him to do his laundry while commuting to work or running errands in the city."]
# ["In the image, a man in a yellow shirt is standing on top of a car parked on a busy city street. The car appears to be a large SUV or minivan, and the man is ironing clothes on the roof of the vehicle. He is wearing a yellow shirt and appears to be focused on his task. Around him, there are several other vehicles on the street, including a taxi, a truck, and a bus. The scene suggests that the man is using his car's roof as a makeshift laundry area while he waits for his clothes to dry."]
