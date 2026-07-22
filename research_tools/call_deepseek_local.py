import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

## single image conversation example
conversation = [
    {
        "role": "<|User|>",
        "content": "<image>\n<|ref|>The giraffe at the back.<|/ref|>.",
        "images": ["./images/visual_grounding.jpeg"],
    },
    {"role": "<|Assistant|>", "content": ""},
]

## multiple images (or in-context learning) conversation example
# conversation = [
#     {
#         "role": "User",
#         "content": "<image_placeholder>A dog wearing nothing in the foreground, "
#                    "<image_placeholder>a dog wearing a santa hat, "
#                    "<image_placeholder>a dog wearing a wizard outfit, and "
#                    "<image_placeholder>what's the dog wearing?",
#         "images": [
#             "images/dog_a.png",
#             "images/dog_b.png",
#             "images/dog_c.png",
#             "images/dog_d.png",
#         ],
#     },
#     {"role": "Assistant", "content": ""}
# ]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation,
    images=pil_images,
    force_batchify=True,
    system_prompt=""
).to(vl_gpt.device)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(f"{prepare_inputs['sft_format'][0]}", answer)

import sys
sys.exit()


import sys
from typing import List
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
import torch
import os

# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model_name = "deepseek-ai/deepseek-vl2-tiny" # small
llm = LLM(model_name, tensor_parallel_size=2, download_dir="/model/DeepSeek")


def get_input(question, image_urls):
    placeholder = "".join(f"image_{i}:<image>\n"
                          for i, _ in enumerate(image_urls, start=1))
    prompt = f"<|User|>: {placeholder}{question}\n\n<|Assistant|>:"
    images = [Image.open(image_path) for image_path in image_paths]
    return prompt, images


def run_generate(model, question: str, image_urls: List[str]):
    prompt, images = get_input(question, image_urls)

    sampling_params = SamplingParams(temperature=0.0,
                                     max_tokens=128,
                                     stop_token_ids=None)

    outputs = model.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": images
            },
        },
        sampling_params=sampling_params)

    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)

    return outputs


prompt = "Describe the interaction in the given images."
image_paths = [
    '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00001.jpg',
    '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00005.jpg',
    '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00009.jpg',
    '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00013.jpg',
]

response = run_generate(llm, prompt, image_paths)

# Print the response
print("Generated Response:")
print(response.text)




sys.exit()



from vllm import LLM, SamplingParams
#, BeamSearchParams

llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    tensor_parallel_size=2,
    download_dir="/model/DeepSeek"
)

params = SamplingParams(
    temperature=0,
    max_tokens=64,
    logprobs=0,
)

video_frames = [
    '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00001.jpg',
    '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00005.jpg',
    '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00009.jpg',
    '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/000013.jpg',
]
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this set of frames. Consider the frames to be a part of the same video."},
    ],
}
for i in range(len(video_frames)):
    base64_image = encode_image(video_frames[i]) # base64 encoding.
    new_image = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    message["content"].append(new_image)


outputs = llm.generate(["Hello, my name is", "Prove that 1+1=2."], params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated text: {generated_text!r}")
    import pdb; pdb.set_trace()

    cumulative_logprob = output.outputs[0].cumulative_logprob
    print(f"cumulative_logprob: {cumulative_logprob}")
    num_tokens = len(output.outputs[0].logprobs)
    print(f"num_tokens: {num_tokens}")
    ppl = 2 ** (-cumulative_logprob / num_tokens)
    print(f"ppl: {ppl}")



sys.exit()



params = SamplingParams(beam_width=5, max_tokens=50)
outputs = llm.generate("Hello, my name is", params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")




from vllm import VLLM

model = VLLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    download_dir="/model/DeepSeek"
)
output = model.generate(
    input_text="Hello, my name is",
    decoding_algorithm="beam_search",
    beam_width=5,
)
print(output)




# from vllm import LLM, SamplingParams

# llm = LLM(model="FuseAI/FuseO1-DeepSeekR1-QwQ-SkyT1-32B-Preview", tensor_parallel_size=8)
# sampling_params = SamplingParams(max_tokens=32768, temperature=0.7, stop=["<|im_end|>", "<｜end▁of▁sentence｜>"], stop_token_ids=[151645, 151643])

# conversations = [
#     [
#         {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{{}}."},
#         {"role": "user", "content": "Quadratic polynomials $P(x)$ and $Q(x)$ have leading coefficients $2$ and $-2,$ respectively. The graphs of both polynomials pass through the two points $(16,54)$ and $(20,53).$ Find $P(0) + Q(0).$."},
#     ],
# ]

# responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)

# for response in responses:
#     print(response.outputs[0].text.strip())
