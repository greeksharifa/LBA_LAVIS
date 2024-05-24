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

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_t5 import T5Config, T5ForConditionalGeneration
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
    
    LBA_PROMPT = "Context : is the sky blue? no. are there clouds in the sky? yes. Question : what weather is likely? Short answer : rain  Context : {sub_question}? {sub_answer}. Question : {main_question}? Short answer : "
    
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
        decomposer_name="self",
        surprisal_threshold=1e-5,
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
        
        self.decomposer_name = decomposer_name
        if decomposer_name != "self":
            # LBA TODO: load decomposer model like flan_t5_base (not blip2)
            pass
        
        
        self.surprisal_threshold = surprisal_threshold
        print('surprisal_threshold:', surprisal_threshold)
    
    
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
        decomposer_name = cfg.get("decomposer_name", "self")
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
            decomposer_name=decomposer_name,
            surprisal_threshold=surprisal_threshold
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
        output_text, confidence_list = self.predict_answers(
            samples,
            num_beams=num_beams,
            inference_method=inference_method,
            max_len=max_len,
            min_len=min_len,
            num_ans_candidates=num_ans_candidates,
            answer_list=answer_list,
            prompt=prompt,
            length_penalty=length_penalty,
            generate_sub_q=False,
            **kwargs
        )
        
        if self.decomposer_name == "self":
            output_text_lba, confidence_lba_list = self.predict_answers(
                samples,
                num_beams=num_beams,
                inference_method=inference_method,
                max_len=max_len,
                min_len=min_len,
                num_ans_candidates=num_ans_candidates,
                answer_list=answer_list,
                prompt=prompt,
                length_penalty=length_penalty,
                generate_sub_q=True,
                **kwargs
            )
        else:
            # LBA TODO: decomposer model로 predict
            pass
        
        change_cnt = 0
        
        return_text = []
        for i, (text, confidence, text_lba, confidence_lba) in enumerate(zip(output_text, confidence_list, output_text_lba, confidence_lba_list)):
            if confidence < 1 / (2 ** self.surprisal_threshold):
                change_cnt += 1
                return_text.append(text_lba)
                # print(f'confidence change: {confidence:.6f} -> {confidence_lba:.6f}')
            else:
                return_text.append(text)
            if i < 3:
                print('text    :', text)
                print('text_lba:', text_lba)
        print('avg confidence:', sum(confidence_list) / len(confidence_list))
        print('avg confidence_lba:', sum(confidence_lba_list) / len(confidence_lba_list))
        print(f'change_cnt: {change_cnt} / {len(output_text)}')
        
        return return_text
        
    
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
            if kwargs["generate_sub_q"]:
                sub_qa_prompt = Blip2T5LBA.LBA_PROMPT
                text_input = []
                for main_question, sub_qas in zip(samples["text_input"], samples["sub_qas"]):
                    if len(sub_qas) == 0:
                        '''
# 중복 제거하고 하나의 MQ 당 SQ 개수
Generate in train split...
[(0, 720), (1, 2153), (2, 10558), (3, 35434), (4, 6313), (5, 1111), (6, 197), (7, 32), (8, 1)]
Generate in val split...
[(0, 1116), (1, 2761), (2, 5305), (3, 10938), (4, 2200), (5, 412), (6, 54), (7, 7)]
                        '''
                        text_input.append(prompt.format(main_question))
                    else:
                        text_input.append(sub_qa_prompt.format(sub_question=sub_qas[0][0], sub_answer=sub_qas[0][1], main_question=main_question))
            else:
                text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

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
            
        
        # print('outputs:', outputs.shape)
        # print('self.t5_tokenizer:', self.t5_tokenizer)
        confidence = calculate_sentence_confidence(self.t5_model, self.t5_tokenizer, text_input, output_text)
        # print('confidence:', sep='\t')
        # for c in confidence:
        #     print(f'{c:.6f}', sep=' ')
        # print()


        return output_text, confidence
