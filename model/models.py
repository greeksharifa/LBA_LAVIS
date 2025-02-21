import re
import json

from pprint import pprint
from pathlib import Path
from typing import Any, List, Tuple
from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from config.configs import Config
# from utils.logger import get_logger
# from utils.util import transpose_list, get_confidence_and_ppl
# from dataset.video_dataset import EXTs
# from model.prompt import get_subq_prompt



class C2RFramework(ABC):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.N = cfg.runner_cfg.N
        self.M = cfg.runner_cfg.M
        
        model_id = self.cfg.model_cfg.model_id
        cache_dir = self.cfg.model_cfg.cache_dir

        self.llm = self.load_model_and_processor(model_id, cache_dir)

        self.model.generation_config.do_sample=False
        self.model.generation_config.temperature=None
        self.model.generation_config.top_p=None
        self.model.generation_config.top_k=None
    
    # @abstractmethod
    def load_model_and_processor(self, model_id: str, cache_dir: str):
        llm = LLM(
            model=model_id, 
            trust_remote_code=True, 
            tensor_parallel_size=self.cfg.model_cfg.tensor_parallel_size, 
            gpu_memory_utilization=self.cfg.model_cfg.gpu_memory_utilization, 
            swap_space=self.cfg.model_cfg.swap_space,
        )
        return llm
        # pass

    # @abstractmethod
    # def get_inputs(self, prompts: List[str], visions: Any, vpaths: List[Path] = None) -> Any:
    #     """
    #     Apply template to the text inputs and visions.
    #     Args:
    #         prompts: List[str]
    #         visions: List[Any]
    #         vpaths: List[str]
    #     Returns: list of inputs for model
    #         transformers.BatchFeature
    #     """
    #     pass

    def generate(self, prompts: List[Any]) -> List[Any]:
        sampling_params = SamplingParams(
            max_num_seqs=self.cfg.model_cfg.max_num_seqs,
            max_new_tokens=self.cfg.model_cfg.max_new_tokens,
            temperature=self.cfg.model_cfg.temperature,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return outputs

    def _generate(self,
                  inputs: Any,
                  return_confidence: bool,
                  max_new_tokens: int,
                  **kwargs
                  ) -> dict:
        outputs = self.llm.generate(inputs, sampling_params)
        pass

    def postprocess_outputs(self, outputs: List[str]) -> List[str]:
        pass

    def postprocess_subqs(self, output_texts: List[str]) -> List[List[str]]:
        pass

    def generate_sub_qas(self, 
                         main_qs: List[str], 
                         visions: List[Any],
                         vpaths: List[Path] = None,
                         ) -> Tuple[List[List[str]], List[List[str]], List[List[float]], List[List[float]], List[List[float]]]:
        pass
    
    def generate_base_answers(self, 
                              main_qs          : List[str], 
                              candidate_lists  : List[List[str]],
                              visions          : List[Any],
                              question_types   : List[str],
                              **kwargs
                              ) -> Tuple[List[str], List[float], List[float], List[float]]:
        pass

    def generate_refined_answers(self, 
                                 main_qs: List[str], 
                                 candidate_lists: List[List[str]], 
                                 visions: List[Any],
                                 sub_qs: List[List[str]], 
                                 sub_as: List[List[str]],
                                 base_answers: List[str],
                                 question_types: List[str] = None,
                                 few_shot_samples: List[str] = None,
                                 ) -> Tuple[List[str], List[float], List[float], List[float], List[str]]:
        pass
    
class Qwen2_5VL(C2RFramework):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # self.model = self.load_model_and_processor(cfg.model_cfg.model_id, cfg.model_cfg.cache_dir)
