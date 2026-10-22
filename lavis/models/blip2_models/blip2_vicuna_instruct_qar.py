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
    # prompts = None
    base = dict()
    _cnt = 0
    
    
    @staticmethod
    def set_questioner_and_answerer(questioner, answerer, finetuned):
        Blip2VicunaInstructQAR.Questioner = questioner
        Blip2VicunaInstructQAR.Answerer = answerer
        Blip2VicunaInstructQAR.finetuned = finetuned
    
        # base와 비교
        base_json = json.load(open("lavis/ywjang_output_qar_test/20230815093/result/val_vqa_result.json", "r"))
        
        for elem in base_json:
            question_id = elem["question_id"]
            question = elem["question"]
            gt_ans = elem["gt_ans"]
            pred_ans = elem["pred_ans"]
            
            Blip2VicunaInstructQAR.base[question_id] = {
                "question": question,
                "gt_ans": gt_ans,
                "base_pred_ans": pred_ans
            }
    
    
    # dummy
    # def set_new_reasoner_prompt(self, new_reasoner_prompt):
    #     self.sq_prompts["Reasoner"].update(new_reasoner_prompt)
    #     print('self.sq_prompts["Reasoner"]:', self.sq_prompts["Reasoner"])
    #
    # def update_reasoner_prompt(self):
    #     print(Colors.BRIGHT_RED + str(self.reasoner_prompt_done_cnt) + Colors.RESET)
    #     if dist_utils.is_main_process():
    #         _prompts = json.load(open("prompts.json", "r"))
    #         if len(_prompts["Reasoner_test"]) >= self.reasoner_prompt_done_cnt:
    #             self.sq_prompts["Reasoner"].update(
    #                 _prompts["Reasoner_test"][self.reasoner_prompt_done_cnt]
    #             )
    #             self.reasoner_prompt_done_cnt += 1
    #             print(Colors.BRIGHT_YELLOW + '\nupdated! self.sq_prompts["Reasoner"]:' + str(json.dumps(self.sq_prompts["Reasoner"], indent=4)) + Colors.RESET)
    #             return True
    #         else:
    #             return False
    #
    #     return False
    
    
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
        
        if hasattr(Blip2VicunaInstructQAR.Questioner, "device") and samples["image"].device != Blip2VicunaInstructQAR.Questioner.device:
            Blip2VicunaInstructQAR.Questioner.to(samples["image"].device)
            logging.info(Colors.BRIGHT_MAGENTA + f"Questioner device move: {Blip2VicunaInstructQAR.Questioner.device}" + Colors.RESET)
        if hasattr(Blip2VicunaInstructQAR.Answerer, "device") and samples["image"].device != Blip2VicunaInstructQAR.Answerer.device:
            Blip2VicunaInstructQAR.Answerer.to(samples["image"].device)
            logging.info(Colors.BRIGHT_MAGENTA + f"Answerer device move: {Blip2VicunaInstructQAR.Answerer.device}" + Colors.RESET)
        
        main_question = list(samples["text_input"])
        # logging.info(Colors.BRIGHT_MAGENTA + f"main_question: {main_question}" + Colors.RESET)
        if isinstance(main_question, list):
            main_question = main_question[0]
        # logging.info(Colors.BRIGHT_MAGENTA + f"main_question: {main_question}" + Colors.RESET)
        
        questioner_prompts = self.sq_prompts["Questioner_MultipleSubQ"]
        answerer_prompts = self.sq_prompts["Answerer"]
        reasoner_prompts = self.sq_prompts["Reasoner"]
        
        questioner_prompt = questioner_prompts["init_prompt"].format(main_question)
        
        sub_question_list = []
        sub_answer_list = []
        
        reasoner_prompt = reasoner_prompts["init_prompt"]
        if reasoner_prompts["init_prompt"].count('{}') == 1:
            reasoner_prompt = reasoner_prompts["init_prompt"].format(main_question)    # reasoner_prompt
            
        sub_qa_is_none = False
        
        for i in range(1, 1+1):
            # Questioner
            samples["prompt"] = questioner_prompt if i == 1 else questioner_prompt + '.'
            # print('i:', i, 'questioner samples["prompt"]:', samples["prompt"], sep='\t')
            sub_question = Blip2VicunaInstructQAR.Questioner.generate(samples)
            if sub_question is None:
                if i == 1:
                    sub_qa_is_none = True
                    # print('sub_question is None' + '!' * 400)
                break
            # print('i:', i, 'sub_question:', sub_question, sep='\t')
            if isinstance(sub_question, list):
                sub_question = sub_question[0]
            sub_question_list.append(sub_question)
            if i == 1:
                questioner_prompt += questioner_prompts["after_prompt"]
            else:
                questioner_prompt += ', '
            questioner_prompt += questioner_prompts["pair_prompt"].format(i, sub_question)
            
            # Answerer
            answerer_prompt = answerer_prompts["init_prompt"].format(sub_question)
            samples["prompt"] = answerer_prompt
            # print('i:', i, 'answerer samples["prompt"]:', samples["prompt"], sep='\t')

            sub_answer = Blip2VicunaInstructQAR.Answerer.generate(samples)#, answerer=True)
            # print('i:', i, 'sub_answer:', sub_answer, sep='\t')
            if isinstance(sub_answer, list):
                sub_answer = sub_answer[0]
            if sub_answer.endswith('0' * 10):
                sub_answer = ' '.join(sub_answer.split()[:-1])
            sub_answer_list.append(sub_answer)

            # Reasoner
            if i > 1 and not reasoner_prompts["pair_prompt"].endswith('. '):
                reasoner_prompt += ', '
            if reasoner_prompts["pair_prompt"].count('{}') == 4:
                reasoner_prompt += reasoner_prompts["pair_prompt"].format(i, sub_question, i, sub_answer)
            else:
                reasoner_prompt += reasoner_prompts["pair_prompt"].format(sub_question, sub_answer)
            
        if sub_qa_is_none:
            reasoner_prompt = ""
        
        reasoner_prompt += reasoner_prompts["final_prompt"].format(main_question)
        samples["prompt"] = reasoner_prompt
        # try:
        self.llm_tokenizer.padding_side = "left"
        
        image = samples["image"]

        bs = image.size(0)

        if isinstance(reasoner_prompt, str):
            reasoner_prompt = [reasoner_prompt] * bs
        else:
            assert len(reasoner_prompt) == bs, "The number of prompts must be equal to the batch size."
            
        # logging.info(Colors.BLUE + f"Prompt: {prompt}" + Colors.RESET)
        
        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
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

        outputs[outputs == 0] = 2   # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        
        if Blip2VicunaInstructQAR._cnt < 2000:
            print_sample(samples, output_text=output_text, msg=f'in generate(), eval sample: {Blip2VicunaInstructQAR._cnt}', color=Colors.BLUE)
            # logging.info(Colors.CYAN + f"prompt: {prompt}" + Colors.RESET)
        elif Blip2VicunaInstructQAR._cnt == 20:
            print(Colors.BRIGHT_GREEN + "finetuned: " + json.dumps(Blip2VicunaInstructQAR.finetuned, indent=4) + Colors.RESET)
            print(Colors.BRIGHT_YELLOW + "prompts: \n" + json.dumps(self.sq_prompts, indent=4) + Colors.RESET)
        
        Blip2VicunaInstructQAR._cnt += 1

        return output_text, ['\n'.join(sub_question_list)], ['\n'.join(sub_answer_list)]
        #
        # except Exception as e:
        #     print_sample(samples, msg='in QAR generate(), ERROR OCCUR!' + '!' * 170, color=Colors.RED)
        #     logging.info(Colors.RED + f"prompt: {prompt}" + Colors.RESET)
        #     logging.info(Colors.RED + f"error msg: {e}" + Colors.RESET)
        #     return ['ERROR message: ' + str(e)] * samples["image"].size(0)
        #     # raise Exception(e)
