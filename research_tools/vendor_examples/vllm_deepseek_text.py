"""Local vLLM DeepSeek text-generation smoke example.

Created against vllm-project/vllm at
0f465ab53303fbd3c8ad32163db161cdb0cf8dad. See the preserved license under
``vendor_patches/licenses/vllm-LICENSE``.
"""

from vllm import LLM, SamplingParams
#, BeamSearchParams

llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", cache_dir="/model/DeepSeek")
params = SamplingParams(beam_width=5, max_tokens=50)
outputs = llm.generate("Hello, my name is", params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")



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
