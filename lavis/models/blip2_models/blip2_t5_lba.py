"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import T5TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
from lavis.models.blip2_models.blip2_t5 import Blip2T5

from confidence import calculate_sentence_confidence

@registry.register_model("blip2_t5_lba")
class Blip2T5LBA(Blip2T5):
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5xl": "configs/models/LBA/blip2_pretrain_flant5xl_LBA.yaml",
        # "pretrain_flant5xl_vitL": "configs/models/blip2/blip2_pretrain_flant5xl_vitL.yaml",
        # "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        # "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }
    
    LBA_PROMPT = {
        "recomposer": "Context: is the sky blue? no. are there clouds in the sky? yes. Question: what weather is likely? Short answer: rain.  Context: {sub_question}? {sub_answer}. Question: {main_question} Short answer: ",
        "decomposer": "Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {main_question} Perception Question: ",
        
        "K1": "What is a missing information about who or what a person or thing is?", # Identity
        "K2": "What is a missing information about inclusion relationships of {entity}?", # Class
        "K3": "What is a missing information about properties or feature of {entity}?", # Attributes
        "K4": "What is a missing information about the number of {entity}?", # Quantities
        "K5": "What is a missing information about spatial relations among {entity}?", # Spatial
        "K7": "What is a missing information about detailed information of {entity}?", # Contents
    }
    
    __cnt = 0
    
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        t5_model="google/flan-t5-xl",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        decomposition=True,
        decomposer_name="self",
        surprisal_threshold=1e-5,
        decompose_using_GT=False,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )

        assert decomposition in ["K-type", "zero-shot", False], f"decomposition should be one of ['K-type', 'zero-shot', False], but got {decomposition}."
        self.decomposition = decomposition
        
        self.decomposer_name = decomposer_name
        if decomposition and decomposer_name != "self":
            # LBA TODO: load decomposer model like flan_t5_base (not blip2)
            self.decomposer_tokenizer = T5Tokenizer.from_pretrained(self.decomposer_name)
            self.decomposer_model = T5ForConditionalGeneration.from_pretrained(
                self.decomposer_name, 
                torch_dtype=torch.bfloat16,
                # load_in_4bit=True,
                # device_map="auto",
            ).to(self.t5_model.device)
                
        
        self.surprisal_threshold = surprisal_threshold
        print('surprisal_threshold:', surprisal_threshold)
        
        self.decompose_using_GT = decompose_using_GT
        print('decompose_using_GT:', decompose_using_GT)
    
    
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)
        
        # LBA
        decomposition = cfg.get("decomposition", True)
        decomposer_name = cfg.get("decomposer_name", "self")
        decompose_using_GT = cfg.get("decompose_using_GT", False)
        surprisal_threshold = cfg.get("surprisal_threshold", 1e-5) # meaning of default value: almost always generate sub-q

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
            decomposition=decomposition,
            decomposer_name=decomposer_name,
            surprisal_threshold=surprisal_threshold,
            decompose_using_GT=decompose_using_GT,
        )
        model.load_checkpoint_from_config(cfg)

        return model
    
    def predict_answers_by_lba(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        **kwargs
    ):
        def _predict_answers(_samples, prompt_type="default", prompt=prompt):
            return self.predict_answers(
                samples=_samples,
                num_beams=num_beams,
                inference_method=inference_method,
                max_len=max_len,
                min_len=min_len,
                num_ans_candidates=num_ans_candidates,
                answer_list=answer_list,
                prompt=prompt,
                length_penalty=length_penalty,
                prompt_type=prompt_type,
            )
            
        output_texts_origin, confidences = _predict_answers(samples)

        if self.decomposition == False: # baseline
            return {
                'pred_answers': output_texts_origin
            }
        else:    
            if self.decomposition == "K-type":
                pass
            elif self.decomposition == "GT":
                # sub_qa 생성 생략
                # generate main_answer (recomposition)
                output_texts_lba, _ = _predict_answers(samples, prompt_type="recomposition")
            elif self.decomposition == "zero-shot":
                device = self.decomposer_model.device
                
                # generate sub_question (decomposition)
                decomposer_prompt = self.get_lba_prompt("decomposer")
                text_input = [decomposer_prompt.format(main_question=main_question) for main_question in samples["text_input"]]
                
                if self.decomposer_name == "self":  # Image+Text
                    sub_questions, _ = _predict_answers(samples, prompt_type="decomposition")
                else:                               # Only Text
                    input_ids = self.decomposer_tokenizer(text_input, padding="longest", return_tensors="pt").input_ids.to(device)
                    '''
                    # print('input_ids device:', input_ids.device)
                    # print('self.decomposer_model device:', self.decomposer_model.device)
                    # print('self.decomposer_model device:', next(self.decomposer_model.parameters()).device)
                    '''
                    outputs = self.decomposer_model.generate(input_ids)
                    sub_questions = self.decomposer_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # generate sub_answer
                samples_for_sub_answer = samples.copy()
                samples_for_sub_answer["text_input"] = sub_questions
                sub_answers, _ = _predict_answers(samples_for_sub_answer)
                
                # generate main_answer (recomposition)
                samples_for_main_answer = samples.copy()
                _sub_qas = []
                
                for sub_question, sub_answer in zip(sub_questions, sub_answers):
                    _sub_qas.append([(sub_question, sub_answer)])   # _sub_qas shape: [bs, 1, 2]
                
                samples_for_main_answer["sub_qas"] = _sub_qas 
                output_texts_lba, _ = _predict_answers(samples_for_main_answer, prompt_type="recomposition")
                    
                
            return {
                'output_texts_origin': output_texts_origin,
                'output_texts_lba': output_texts_lba,
                'confidences': confidences,
            }
            
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        prompt_type="default",
        **kwargs
    ):
        image = samples["image"] # torch.Size([bs, 3, 224, 224])
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if prompt:
            if prompt_type == "recomposition":
                recomposer_prompt = self.get_lba_prompt("recomposer")
                text_input = []
                for main_question, sub_qas in zip(samples["text_input"], samples["sub_qas"]):
                    if len(sub_qas) == 0: # GT에 sub_qa가 없는 경우가 있음
                        '''
# 중복 제거하고 하나의 MQ 당 SQ 개수
Generate in train split...
[(0, 720), (1, 2153), (2, 10558), (3, 35434), (4, 6313), (5, 1111), (6, 197), (7, 32), (8, 1)]
Generate in val split...
[(0, 1116), (1, 2761), (2, 5305), (3, 10938), (4, 2200), (5, 412), (6, 54), (7, 7)]
                        '''
                        text_input.append(prompt.format(main_question))
                    else:
                        # LBA TODO: 현재 첫번째 sub_qa만 고정적으로 사용함
                        assert not isinstance(sub_qas[0], str), f"type of sub_qas[0] is {type(sub_qas[0])}."
                        assert len(sub_qas[0]) == 2, f"len of sub_qas[0] is {len(sub_qas[0])}."
                        sub_question, sub_answer = sub_qas[0]
                        sub_question = sub_question.rstrip('?')
                        sub_answer = sub_answer.rstrip('.')
                        text_input.append(recomposer_prompt.format(sub_question=sub_question, sub_answer=sub_answer, main_question=main_question))
            elif prompt_type == "decomposition":
                decomposer_prompt = self.get_lba_prompt("decomposer")
                text_input = [decomposer_prompt.format(main_question=main_question) for main_question in samples["text_input"]]
            else: # prompt_type == "default"
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]
            
        if Blip2T5LBA.__cnt < 10:
            Blip2T5LBA.__cnt += 1
            print('text_input[:2]:', text_input[:2])

        input_tokens = self.t5_tokenizer(
            text_input, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
            )
            output_text = self.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if self._apply_lemmatizer:
            output_text = self._lemmatize(output_text)
            
        confidence = calculate_sentence_confidence(self.t5_model, self.t5_tokenizer, text_input, output_text)

        return output_text, confidence
