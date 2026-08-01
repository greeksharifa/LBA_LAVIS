import re
import json

from pprint import pprint
from pathlib import Path
from typing import Any, List, Tuple, Union, Optional
from abc import ABC, abstractmethod

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer

from model.vllm_config import (
    build_engine_kwargs,
    patch_zero_count_dummy_video_allocation,
    validate_tensor_parallel_size,
)

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

        self.engine_args, self.llm = self.load_model()

    @abstractmethod
    def load_model(self) -> Tuple[dict, LLM]:
        pass

    @abstractmethod
    def apply_chat_template(self, text_prompt: str, vision: Any = None, mm_uuids: str = None) -> dict:
        """
        Apply chat template to text prompts.
        Args:
            text_prompt: Text prompt
            vision: Vision data. images: PIL.Image or [PIL.Image]. videos: np_ndarrays or (np_ndarrays, metadata).
            mm_uuids: Multi-modal UUIDs.
        Returns: vllm prompt with chat template
        """
        pass

    '''
    @abstractmethod
    def get_inputs(self, prompts: List[str], visions: Any, vpaths: List[Path] = None) -> Any:
        """
        Apply template to the text inputs and visions.
        Args:
            prompts: List[str]
            visions: List[Any]
            vpaths: List[str]
        Returns: list of inputs for model
            transformers.BatchFeature
        """
        pass
    '''

    def generate(self, prompts: List[Any]) -> List[Any]:
        sampling_params = SamplingParams(
            n=1,
            max_tokens=self.cfg.model_cfg.max_tokens,
            temperature=self.cfg.model_cfg.temperature,
            logprobs=0,
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
    def load_model(self) -> Tuple[dict, LLM]:
        self.model_id = self.cfg.model_cfg.model_id
        self.modality = self.cfg.dataset_cfg.data_type

        engine_args = build_engine_kwargs(self.cfg)
        validate_tensor_parallel_size(engine_args["tensor_parallel_size"])
        patch_zero_count_dummy_video_allocation()
        llm = LLM(**engine_args)
        return engine_args, llm

    def apply_chat_template(self, text_prompt: str, vision: Any = None, mm_uuids: str = None) -> dict:
        if self.modality == "image":
            placeholder = "<|image_pad|>"
        elif self.modality == "video":
            placeholder = "<|video_pad|>"

        vision_placeholder = placeholder * len(vision)
        # make mm_uuids identical to vision, postfix with _<index>
        mm_uuids = [f"{mm_uuids}_{i}" for i in range(len(vision))]
        # mm_uuids = "_".join(mm_uuids)

        text_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{vision_placeholder}<|vision_end|>"
            f"{text_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        return {
            'multi_modal_data': {self.modality: vision},
            'multi_modal_uuids': {self.modality: mm_uuids},
            'prompt': text_prompt
        }
