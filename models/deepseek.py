import sys
from vllm import LLM, SamplingParams
#, BeamSearchParams


class CustomDeepSeek:
    def __init__(self, model_name, tensor_parallel_size=2, download_dir="/model/DeepSeek", limit_mm_per_prompt={"image": 4}):
        self.llm = LLM(
            model=model_name, 
            tensor_parallel_size=tensor_parallel_size,
            download_dir=download_dir,
            limit_mm_per_prompt=limit_mm_per_prompt
        )
        
    def set_params(self, params):
        self.params = params
        
    def generate(self, input_text):
        outputs = self.llm.generate(input_text, self.params)
        return outputs
    
    
def demo():
    llm = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
        tensor_parallel_size=2,
        download_dir="/model/DeepSeek"
    )

    params = SamplingParams(
        temperature=0,
        max_tokens=1024,
        logprobs=0,
    )
    outputs = llm.generate("Hello, my name is", params)
