"""
Requires Transformer 4.28 and above, implementation may change according the Llama implementation
"""
import logging
import json
import string
from packaging import version

from colors import Colors, print_sample

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers

from lavis.common import dist_utils
from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.blip2_vicuna_instruct import Blip2VicunaInstruct




@registry.register_model("blip2_vicuna_instruct_qar")
class Blip2VicunaInstructQAR(Blip2VicunaInstruct):
    """
    BLIP2 Vicuna QAR model.
    Supported model types:
        - vicuna7b
        - vicuna13b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_vicuna_instruct", "vicuna7b")
    """
    Questioner = None
    Answerer = None
    finetuned = None
    prompts = None
    _cnt = 0
    
    # def from_config(cls, cfg):
    #     vit_model = cfg.get("vit_model", "eva_clip_g")              # 'eva_clip_g'
    #     img_size = cfg.get("image_size")                            # 224
    #     num_query_token = cfg.get("num_query_token")                # 32
    #     llm_model = cfg.get("llm_model")                            # '/home/ywjang/models/lmsys/vicuna-7b-v1.3'
    #
    #     drop_path_rate = cfg.get("drop_path_rate", 0)               # 0
    #     use_grad_checkpoint = cfg.get("use_grad_checkpoint", False) # True
    #     vit_precision = cfg.get("vit_precision", "fp16")            # 'fp16'
    #     freeze_vit = cfg.get("freeze_vit", True)                    # True
    #
    #     prompt = cfg.get("prompt", "")                              # "Write a sub-question about image, when main-question is '{}'. sub-question:"
    #     max_txt_len = cfg.get("max_txt_len", 128)                   # 128
    #     max_output_txt_len = cfg.get("max_output_txt_len", 256)     # 256
    #
    #     apply_lemmatizer = cfg.get("apply_lemmatizer", False)       # False
    #
    #     qformer_text_input = cfg.get("qformer_text_input", True)    # True
    #
    #     model = cls(
    #         vit_model=vit_model,
    #         img_size=img_size,
    #         drop_path_rate=drop_path_rate,
    #         use_grad_checkpoint=use_grad_checkpoint,
    #         vit_precision=vit_precision,
    #         freeze_vit=freeze_vit,
    #         num_query_token=num_query_token,
    #         llm_model=llm_model,
    #         prompt=prompt,
    #         max_txt_len=max_txt_len,
    #         max_output_txt_len=max_output_txt_len,
    #         apply_lemmatizer=apply_lemmatizer,
    #         qformer_text_input=qformer_text_input,
    #     )
    #
    #     # if qformer_text_input:
    #     #     # Hard-coded to load from BLIP-2 stage-1 pre-trained model (not ideal)
    #     #     model.load_from_pretrained(
    #     #         url_or_filename="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
    #     #     )
    #
    #     model.load_checkpoint_from_config(cfg)
    #
    #     return model
    
    
    @staticmethod
    def set_questioner_and_answerer(questioner, answerer, finetuned):
        Blip2VicunaInstructQAR.Questioner = questioner
        Blip2VicunaInstructQAR.Answerer = answerer
        Blip2VicunaInstructQAR.finetuned = finetuned
    
    
    def set_prompts(self):
        _prompt_file_path = "prompts.json"
        self.prompts = json.load(open(_prompt_file_path, "r"))
        # Blip2VicunaInstructQAR.reasoner_prompt_done_cnt = 0


    def set_new_reasoner_prompt(self, new_reasoner_prompt):
        self.prompts["Reasoner"].update(new_reasoner_prompt)
        print('self.prompts["Reasoner"]:', self.prompts["Reasoner"])

    def update_reasoner_prompt(self):
        print(Colors.BRIGHT_RED + str(self.reasoner_prompt_done_cnt) + Colors.RESET)
        if dist_utils.is_main_process():
            _prompts = json.load(open("prompts.json", "r"))
            if len(_prompts["Reasoner_test"]) >= self.reasoner_prompt_done_cnt:
                self.prompts["Reasoner"].update(
                    _prompts["Reasoner_test"][self.reasoner_prompt_done_cnt]
                )
                self.reasoner_prompt_done_cnt += 1
                print(Colors.BRIGHT_YELLOW + '\nupdated! self.prompts["Reasoner"]:' + str(json.dumps(self.prompts["Reasoner"], indent=4)) + Colors.RESET)
                return True
            else:
                return False
            
        return False
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=True,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        # print('*' * 400)
        # print('samples: ', samples)
        """{
        'image': Tensor
        'text_input': ['what type vehicle does the person taking this picture own?'],
        'question_id': ['Cj9HH5YTfWDUZPses2sm8k'],
        'instance_id': ['968'],
        'choices': [['scooter', 'bicycle', 'truck', 'bus']],
        'correct_choice_idx': [None],
        'direct_answers': [None],
        'prompt': ['what type vehicle does the person taking this picture own?']
        }"""
        # logging.info("In QAR Generate:")
        # logging.info(Colors.BRIGHT_MAGENTA + f"Questioner device: {Blip2VicunaInstructQAR.Questioner.device}" + Colors.RESET)
        # logging.info(Colors.BRIGHT_MAGENTA + f"Answerer device: {Blip2VicunaInstructQAR.Answerer.device}" + Colors.RESET)
        # logging.info(Colors.BRIGHT_MAGENTA + f"Image device: {samples['image'].device}" + Colors.RESET)
        
        if samples["image"].device != Blip2VicunaInstructQAR.Questioner.device:
            Blip2VicunaInstructQAR.Questioner.to(samples["image"].device)
            logging.info(Colors.BRIGHT_MAGENTA + f"Questioner device move: {Blip2VicunaInstructQAR.Questioner.device}" + Colors.RESET)
        if samples["image"].device != Blip2VicunaInstructQAR.Answerer.device:
            Blip2VicunaInstructQAR.Answerer.to(samples["image"].device)
            logging.info(Colors.BRIGHT_MAGENTA + f"Answerer device move: {Blip2VicunaInstructQAR.Answerer.device}" + Colors.RESET)
        
        main_question = list(samples["prompt"])
        # logging.info(Colors.BRIGHT_MAGENTA + f"main_question: {main_question}" + Colors.RESET)
        if isinstance(main_question, list):
            main_question = main_question[0]
        # logging.info(Colors.BRIGHT_MAGENTA + f"main_question: {main_question}" + Colors.RESET)
        
        questioner_prompts = self.prompts["Questioner_MultipleSubQ"]
        answerer_prompts = self.prompts["Answerer"]
        reasoner_prompts = self.prompts["Reasoner"]
        
        questioner_prompt = questioner_prompts["init_prompt"].format(main_question)
        
        # sub_question_list = []
        # sub_answer_list = []
        
        reasoner_prompt = ""
        if reasoner_prompts["init_prompt"].count('{}') == 1:
            reasoner_prompt = reasoner_prompts["init_prompt"].format(main_question)    # reasoner_prompt
        
        for i in range(1, 2+1):
            # Questioner
            samples["prompt"] = questioner_prompt if i == 1 else questioner_prompt + '.'
            # print('i:', i, 'questioner samples["prompt"]:', samples["prompt"], sep='\t')
            sub_question = Blip2VicunaInstructQAR.Questioner.generate(samples)
            # print('i:', i, 'sub_question:', sub_question, sep='\t')
            if isinstance(sub_question, list):
                sub_question = sub_question[0]
            # sub_question_list.append(sub_question)
            if i == 1:
                questioner_prompt += questioner_prompts["after_prompt"]
            else:
                questioner_prompt += ', '
            questioner_prompt += questioner_prompts["pair_prompt"].format(i, sub_question)
            
            # Answerer
            answerer_prompt = answerer_prompts["init_prompt"].format(sub_question)
            samples["prompt"] = answerer_prompt
            # print('i:', i, 'answerer samples["prompt"]:', samples["prompt"], sep='\t')

            sub_answer = Blip2VicunaInstructQAR.Answerer.generate(samples)
            # print('i:', i, 'sub_answer:', sub_answer, sep='\t')
            if isinstance(sub_answer, list):
                sub_answer = sub_answer[0]
            # sub_answer_list.append(sub_answer)

            # Reasoner
            if i > 1 and not reasoner_prompts["pair_prompt"].endswith('. '):
                reasoner_prompt += ', '
            if reasoner_prompts["pair_prompt"].count('{}') == 4:
                reasoner_prompt += reasoner_prompts["pair_prompt"].format(i, sub_question, i, sub_answer)
            else:
                reasoner_prompt += reasoner_prompts["pair_prompt"].format(sub_question, sub_answer)
            
        reasoner_prompt += reasoner_prompts["final_prompt"].format(main_question)
        samples["prompt"] = reasoner_prompt
        # prompt = None
        # self.Questioner.generate
        # assert False, "Wow! " * 100
        # try:
        self.llm_tokenizer.padding_side = "left"
        
        # if "prompt" in samples.keys():
        #     prompt = samples["prompt"]
        # else:
        #     prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)

        if isinstance(reasoner_prompt, str):
            reasoner_prompt = [reasoner_prompt] * bs
        else:
            assert len(reasoner_prompt) == bs, "The number of prompts must be equal to the batch size."
            
        # logging.info(Colors.BLUE + f"Prompt: {prompt}" + Colors.RESET)
        
        
        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in reasoner_prompt[0]:
            reasoner_prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(reasoner_prompt)]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                reasoner_prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        # For video data
        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            with self.maybe_autocast():
                image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.llm_tokenizer(
            reasoner_prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)
        
        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)
            
            # self.llm_tokenizer.eos_token_id = 835
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                # eos_token_id=self.llm_tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        
        if Blip2VicunaInstructQAR._cnt < 3:
            Blip2VicunaInstructQAR._cnt += 1
            print_sample(samples, output_text=output_text, msg='in generate(), eval sample:', color=Colors.BLUE)
            # logging.info(Colors.CYAN + f"prompt: {prompt}" + Colors.RESET)
        elif Blip2VicunaInstructQAR._cnt == 3:
            Blip2VicunaInstructQAR._cnt += 1
            print(Colors.BRIGHT_GREEN + "finetuned: " + json.dumps(Blip2VicunaInstructQAR.finetuned, indent=4) + Colors.RESET)
            print(Colors.BRIGHT_YELLOW + "prompts: \n" + json.dumps(self.prompts, indent=4) + Colors.RESET)
        
        return output_text
        #
        # except Exception as e:
        #     print_sample(samples, msg='in QAR generate(), ERROR OCCUR!' + '!' * 170, color=Colors.RED)
        #     logging.info(Colors.RED + f"prompt: {prompt}" + Colors.RESET)
        #     logging.info(Colors.RED + f"error msg: {e}" + Colors.RESET)
        #     return ['ERROR message: ' + str(e)] * samples["image"].size(0)
        #     # raise Exception(e)
