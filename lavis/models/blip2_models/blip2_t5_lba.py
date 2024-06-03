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
        "decomposer": "Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {main_question} Perception Question: "
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
        def _predict_answers(_samples, _recomposition):
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
                recomposition=_recomposition,
            )
            
        output_texts, confidences = _predict_answers(samples, _recomposition=False)
        '''
        output_texts, confidences = self.predict_answers(
            samples,
            num_beams=num_beams,
            inference_method=inference_method,
            max_len=max_len,
            min_len=min_len,
            num_ans_candidates=num_ans_candidates,
            answer_list=answer_list,
            prompt=prompt,
            length_penalty=length_penalty,
            recomposition=False,
            **kwargs
        )
        '''
        
        if self.decomposition:
            if self.decomposer_name == "self":
                output_lba_texts, lba_confidences = _predict_answers(samples, _recomposition=True)
                '''
                output_lba_texts, lba_confidences = self.predict_answers(
                    samples,
                    num_beams=num_beams,
                    inference_method=inference_method,
                    max_len=max_len,
                    min_len=min_len,
                    num_ans_candidates=num_ans_candidates,
                    answer_list=answer_list,
                    prompt=prompt,
                    length_penalty=length_penalty,
                    recomposition=True,
                    **kwargs
                )
                '''
            else:
                # LBA TODO: decomposer model로 predict
                # device = samples["image"].device
                device = self.decomposer_model.device
                
                # generate sub_question (decomposition)
                decomposer_prompt = self.get_lba_prompt("decomposer")
                # print('decomposer_prompt:', decomposer_prompt)
                text_input = [decomposer_prompt.format(main_question=main_question) for main_question in samples["text_input"]]
                # if prompt:
                #     text_input = [prompt.format(question) for question in samples["text_input"]]
                # else:
                #     text_input = samples["text_input"]
                
                input_ids = self.decomposer_tokenizer(text_input, padding="longest", return_tensors="pt").input_ids.to(device)
                # print('input_ids device:', input_ids.device)
                # print('self.decomposer_model device:', self.decomposer_model.device)
                # print('self.decomposer_model device:', next(self.decomposer_model.parameters()).device)
                outputs = self.decomposer_model.generate(input_ids)
                sub_questions = self.decomposer_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                assert len(sub_questions) == len(samples["text_input"]), "sub_questions length mismatch!"
                
                # generate sub_answer
                samples_for_sub_answer = samples.copy()
                samples_for_sub_answer["text_input"] = sub_questions
                sub_answers, _ = _predict_answers(samples_for_sub_answer, _recomposition=False)
                assert len(sub_answers) == len(samples["text_input"]), "sub_answers length mismatch!"
                '''
                sub_answers, _ = self.predict_answers(
                    samples_for_sub_answer,
                    num_beams=num_beams,
                    inference_method=inference_method,
                    max_len=max_len,
                    min_len=min_len,
                    num_ans_candidates=num_ans_candidates,
                    answer_list=answer_list,
                    prompt=prompt,
                    length_penalty=length_penalty,
                    recomposition=False,
                    **kwargs
                )
                '''
                
                # generate main_answer (recomposition) import pdb; pdb.set_trace()
                samples_for_main_answer = samples.copy()
                _sub_qas = []
                for sub_question, sub_answer in zip(sub_questions, sub_answers):
                    _sub_qas.append([(sub_question, sub_answer)])
                samples_for_main_answer["sub_qas"] = _sub_qas #list(zip(sub_questions, sub_answers))
                # [bs, 2] -> [bs, 1, 2]
                assert len(samples_for_main_answer["sub_qas"]) == len(samples["text_input"]), "samples_for_main_answer length mismatch!"
                # print('sub_questions                      len:', len(sub_questions))
                # print('sub_questions:', sub_questions)
                # print('sub_answers                        len:', len(sub_answers))
                # print('sub_answers:', sub_answers)
                # print('samples_for_main_answer["sub_qas"] len:', len(samples_for_main_answer["sub_qas"]))
                # print('samples_for_main_answer["sub_qas"][:2]:', samples_for_main_answer["sub_qas"][:2])
                output_lba_texts, lba_confidences = _predict_answers(samples_for_main_answer, _recomposition=True)
                assert len(output_lba_texts) == len(samples["text_input"]), "output_lba_texts length mismatch!"
                
            
            change_cnt = 0
            
            return_text = []
            for i, (output_text, confidence, output_lba_text, lba_confidence) in enumerate(zip(output_texts, confidences, output_lba_texts, lba_confidences)):
                if confidence < 1 / (2 ** self.surprisal_threshold):
                    change_cnt += 1
                    return_text.append(output_lba_text)
                    # print(f'confidence change: {confidence:.6f} -> {confidence_lba:.6f}')
                else:
                    return_text.append(output_text)
            #     if i < 2:
            #         print('self.decomposer_name:', self.decomposer_name)
            #         if self.decomposer_name != "self":
            #             print(f'main_question: {samples["text_input"][i]}')
            #             print(f'sub_question : {sub_questions[i]}')
            #             print(f'sub_answer   : {sub_answers[i]}')
            #         print(f'text: {output_text:15s}, \t text_lba: {output_lba_text:15s}')
            # print('avg confidence    :', sum(confidences) / len(confidences))
            # print('avg confidence_lba:', sum(lba_confidences) / len(lba_confidences))
            # print(f'change_cnt: {change_cnt} / {len(output_texts)}')
            
            return {
                'original_output_texts': output_texts,
                'output_lba_texts': output_lba_texts,
                'pred_answers': return_text,
                'confidences': confidences,
            }
        else:
            return {
                'pred_answers': output_texts
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
        recomposition=True,
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
            if recomposition:
                # sub_qa_prompt = Blip2T5LBA.LBA_PROMPT["recomposer"]
                recomposer_prompt = self.get_lba_prompt("recomposer")
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
                        # LBA TODO: 현재 첫번째 sub_qa만 고정적으로 사용함
                        assert isinstance(sub_qas[0], tuple), f"type of sub_qas[0] is {type(sub_qas[0])}."
                        assert len(sub_qas[0]) == 2, f"len of sub_qas[0] is {len(sub_qas[0])}."
                        sub_question, sub_answer = sub_qas[0]
                        sub_question = sub_question.rstrip('?')
                        sub_answer = sub_answer.rstrip('.')
                        text_input.append(recomposer_prompt.format(sub_question=sub_question, sub_answer=sub_answer, main_question=main_question))
                        
                        """if isinstance(sub_qas[0], str):
                            assert "sub_qas[0] is str. NOOO)"
                        try:
                            text_input.append(recomposer_prompt.format(sub_question=sub_qas[0][0], sub_answer=sub_qas[0][1], main_question=main_question))
                            print('main_question:', main_question)
                            print('sub_qas:', type(sub_qas), len(sub_qas), sub_qas)
                            print('sub_qas[0]:', type(sub_qas[0]), len(sub_qas[0]), sub_qas[0])
                        except IndexError as e:
                            print('>' * 400)
                            print('text_input:', text_input)
                            print('main_question:', main_question)
                            print('sub_qas:', type(sub_qas), len(sub_qas), sub_qas)
                            print('sub_qas[0]:', type(sub_qas[0]), len(sub_qas[0]), sub_qas[0])
                            print(e)
                            '''
                            sub_qas: <class 'tuple'> 2 ('0', '')
                            sub_qas[0]: <class 'str'> 1 0
                            string index out of range'''
                            print('<' * 400)
                            import json
                            json.dump(sub_qas, open('indexerror_samples.json', 'w'), indent=4)
                            assert False, "No!"
                        """
            else:
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
            
        
        # print('outputs:', outputs.shape)
        # print('self.t5_tokenizer:', self.t5_tokenizer)
        confidence = calculate_sentence_confidence(self.t5_model, self.t5_tokenizer, text_input, output_text)
        # print('confidence:', sep='\t')
        # for c in confidence:
        #     print(f'{c:.6f}', sep=' ')
        # print()


        return output_text, confidence
