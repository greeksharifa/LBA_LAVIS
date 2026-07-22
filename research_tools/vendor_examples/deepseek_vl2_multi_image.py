"""Local DeepSeek-VL2 multi-image smoke example.

Derived from deepseek-ai/DeepSeek-VL2 at
ff23960c5cf9e6874b44be38af930cfb0ccbb620. See the preserved license under
``vendor_patches/licenses/DeepSeek-VL2-LICENSE-CODE``.
"""

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# specify the path to the model
model_path = "deepseek-ai/deepseek-vl2-small"
vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

# multiple images/interleaved image-text
conversation = [
    {
        "role": "<|User|>",
        "content": "This is image_1: <image>\n"
                   "This is image_2: <image>\n"
                   "This is image_3: <image>\n Can you tell me what are in the images?",
        "images": [
            '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00001.jpg',
            '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00005.jpg',
            '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00009.jpg',
            # '/data/VLEP/vlep_frames/friends_s08e03_seg01_clip_00_ep/00013.jpg',
        ],
    },
    {"role": "<|Assistant|>", "content": ""}
]

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
outputs = vl_gpt.language.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=prepare_inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False)
print(f"{prepare_inputs['sft_format'][0]}", answer)
