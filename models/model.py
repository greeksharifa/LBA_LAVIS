from typing import Optional
import requests
from PIL import Image
import os

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from processors.alpro_processors import AlproVideoEvalProcessor
from transformers import InstructBlipVideoImageProcessor, InstructBlipVideoProcessor, InstructBlipVideoForConditionalGeneration
from accelerate import infer_auto_device_map
# from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info

# from utils.api_chatgpt import call_vision_api, ndarrays_to_base64
from utils.utils import ndarrays_to_base64

import numpy as np
import torch
from torch import nn


class Decomposer(nn.Module):
    def __init__(self, cfg, device="cuda"):
        super().__init__()
        
        self.device = device
        self.cfg = cfg
        
        self.model_name = f'google/flan-t5-{cfg.runner_cfg.decomposer_name}'
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16,
            # load_in_4bit=True,
            # device_map="auto",
            cache_dir=os.path.join(cfg.model_cfg.cache_dir, "google"),
            # local_files_only=True,
        ).to(device)

    def forward(self, text_inputs, generate_sub_q=True, beam_search=True):
        input_ids = self.tokenizer(text_inputs, padding="longest", return_tensors="pt").input_ids.to(self.device)
        # outputs = self.decomposer_model.generate(input_ids)
        outputs = self.model.generate(input_ids, num_beams=5, do_sample=True, top_p=0.95, temperature=1.0, length_penalty=1.0, repetition_penalty=1.0, max_new_tokens=100)
        sub_questions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return sub_questions #,None


class VideoBlip2ForConditionalGeneration(Blip2ForConditionalGeneration):
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                                              or shape (batch_size, num_channels, n_frms, height, width)):
                                                       (64,         3,            5,      224,    224  )
                Input images to be processed.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        # print('in VideoBlip2ForConditionalGeneration.generate(), pixel_values.shape:', pixel_values.shape)
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        # import pdb; pdb.set_trace()
        # For video data
        if pixel_values.dim() == 5:                         # [bsz, n_frms, 3, 224, 224]
            language_model_inputs, language_attention_mask = [], []
            for j in range(pixel_values.size(2)):
                # this_frame = pixel_values[:, :, j, :, :]    # [bsz, 3, 224, 224]
                this_frame = pixel_values[:, j, :, :, :]    # [bsz, 3, 224, 224]
                frame_embeds = self.vision_model(this_frame, return_dict=True).last_hidden_state
                frame_attention_mask = torch.ones(frame_embeds.size()[:-1], dtype=torch.long, device=frame_embeds.device)

                frame_query_tokens = self.query_tokens.expand(batch_size, -1, -1)
                frame_query_outputs = self.qformer(
                    query_embeds=frame_query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_attention_mask,
                    return_dict=True,
                )
                frame_query_output = frame_query_outputs.last_hidden_state
                
                language_model_inputs.append(self.language_projection(frame_query_output))
                language_attention_mask.append(torch.ones(
                    language_model_inputs[-1].size()[:-1], dtype=torch.long, device=language_model_inputs[-1].device
                ))
                
            language_model_inputs = torch.cat(language_model_inputs, dim=1)
            language_attention_mask = torch.cat(language_attention_mask, dim=1)
            
        else:
            image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

            query_tokens = self.query_tokens.expand(batch_size, -1, -1)
            query_outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            query_output = query_outputs.last_hidden_state

            language_model_inputs = self.language_projection(query_output)
            language_attention_mask = torch.ones(
                language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
            )
            
        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(frame_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        
        # import pdb; pdb.set_trace()

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs


class VideoInstructionBlipForConditionalGeneration(InstructBlipForConditionalGeneration):
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        if pixel_values.dim() == 5:
            language_model_inputs, language_attention_mask = [], []
            for j in range(pixel_values.size(2)):
                # this_frame = pixel_values[:, :, j, :, :]    # [bsz, 3, 224, 224]
                this_frame = pixel_values[:, j, :, :, :]    # [bsz, 3, 224, 224]
                frame_embeds = self.vision_model(this_frame, return_dict=True).last_hidden_state
                frame_attention_mask = torch.ones(frame_embeds.size()[:-1], dtype=torch.long, device=frame_embeds.device)
                
                frame_query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                frame_query_attention_mask = torch.ones(frame_query_tokens.size()[:-1], dtype=torch.long, device=frame_embeds.device)
                if qformer_attention_mask is None:
                    frame_qformer_attention_mask = torch.ones_like(qformer_input_ids)
                frame_qformer_attention_mask = torch.cat([frame_query_attention_mask, frame_qformer_attention_mask], dim=1)
                frame_query_outputs = self.qformer(
                    input_ids=qformer_input_ids,
                    attention_mask=frame_qformer_attention_mask,
                    query_embeds=frame_query_tokens,
                    encoder_hidden_states=frame_embeds,
                    encoder_attention_mask=frame_attention_mask,
                    return_dict=True,
                )
                frame_query_output = frame_query_outputs.last_hidden_state[:, : frame_query_tokens.size(1), :]
                
                language_model_inputs.append(self.language_projection(frame_query_output))
                language_attention_mask.append(torch.ones(
                    language_model_inputs[-1].size()[:-1], dtype=torch.long, device=language_model_inputs[-1].device
                ))
            
            language_model_inputs = torch.cat(language_model_inputs, dim=1)
            language_attention_mask = torch.cat(language_attention_mask, dim=1)
            
        else:
            image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state

            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
            if qformer_attention_mask is None:
                qformer_attention_mask = torch.ones_like(qformer_input_ids)
            qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
            query_outputs = self.qformer(
                input_ids=qformer_input_ids,
                attention_mask=qformer_attention_mask,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

            language_model_inputs = self.language_projection(query_output)
            language_attention_mask = torch.ones(
                language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
            )

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # the InstructBLIP authors used inconsistent tokenizer/model files during training,
        # with the tokenizer's bos token being set to </s> which has ID=2,
        # whereas the model's text config has bos token id = 0
        if self.config.text_config.architectures[0] == "LLaMAForCausalLM":
            if isinstance(outputs, torch.Tensor):
                outputs[outputs == 0] = 2
            else:
                outputs.sequences[outputs.sequences == 0] = 2

        return outputs


class Recomposer(nn.Module):
    def __init__(self, cfg, device, model_type):
        super().__init__()
        model_name = cfg.runner_cfg.get(f"{model_type}_name")
        cache_dir = os.path.join(cfg.model_cfg.cache_dir, model_name.split('/')[0])
        device_map = cfg.runner_cfg.device_map # if cfg.runner_cfg.device_map else device
        # self.processor = AlproVideoEvalProcessor(cfg.datasets_cfg.vis_processor.eval)
        # self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cfg.model_cfg.cache_dir).to(device)
        if model_type == "answerer":
            cache_dir = os.path.join(cfg.model_cfg.cache_dir, model_name.split('/')[0])
            self.processor = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = VideoBlip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir, device_map=device_map)#.to(device)
        elif "blip2" in model_name: # "flan-t5" in model_name or "blip2-opt-" in model_name:
            self.processor = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = VideoBlip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")
        elif "instructblip" in model_name:
            self.processor = InstructBlipVideoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = InstructBlipVideoForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir, device_map=device_map)#.to(device)
            # self.processor = InstructBlipProcessor.from_pretrained(model_name)
            # self.model = InstructBlipForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir, device_map=device_map)#.to(device)
        elif "Video-LLaVA" in model_name:
            from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration
            self.processor = VideoLlavaProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                model_name, 
                cache_dir=cache_dir, #os.path.join(cache_dir, "LanguageBind/"), 
                device_map="auto",
                attn_implementation=None,
            )
        elif "Qwen" in model_name:
            from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
            # default: Load the model on the available device(s)
            # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            #     model_name, 
            #     cache_dir=cache_dir, 
            #     torch_dtype="auto", 
            #     device_map=device_map
            # )
            # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                cache_dir=cache_dir,
                device_map=device_map,
            )

        else:
            raise NotImplementedError(f"Invalid Recomposer model name: {model_name}")
        
        if device_map == "auto":
            device_map = infer_auto_device_map(self.model)
            del self.model
            self.model = VideoBlip2ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir, device_map=device_map)

        self.model_name = model_name
        self.device = self.model.device
        self.cfg = cfg
        self.device_map = device_map
        print('device_map:', device_map)
        print('recomposer name: ',self.model.__class__.__name__)
        # print('self.processor:', self.processor)
        # print('self.processor.image_processor:', self.processor.image_processor)


    def forward(self, vision, text_inputs, generate_sub_q=False, beam_search=True):
        if self.model_name == "sevila":# in self.cfg.runner_cfg.recomposer_name and self.cfg.runner_cfg.answerer_name is None:
            samples = text_inputs

            pixel_values = []
            for video in vision: # video: [n_frms, 640, 480]
                # [n_frms, 640, 480] -> [n_frms, 3, 224, 224]
                pixel_values.append(self.processor(images=video, return_tensors="pt", padding=True)['pixel_values'])  # [n_frms, 3, 224, 224]
            # [n_frms, 3, 224, 224] -> [bsz, n_frms, 3, 224, 224]
            samples["video"] = torch.stack(pixel_values, dim=0)#.to(self.model.device)
            
            # video = self.processor(vision, return_tensors="pt", padding=True)['pixel_values'].to(self.model.device)
            # samples["video"] = video
            output_text, output_scores = self.model.generate(samples)
        elif "Qwen" in self.model_name:
            from qwen_vl_utils import process_vision_info
            
            messages = []
            # print('start', '#' * 100)
            # print(text_inputs[0])
            # print('end  ', '#' * 100)
            for vis, txt in zip(vision, text_inputs):
                # shape: such as [n_frms, 640, 480] or [640, 480]
                vis_type = "video" if type(vis) == list else "image"
                # vis_type = "video" if vis.ndim == 3 else "image" 
                if vis_type == "video":
                    base64_vis = ndarrays_to_base64(vis, add_prefix=True)
                else:
                    base64_vis = ndarrays_to_base64([vis], add_prefix=True)[0]
                    
                messages.append([{
                    "role": "user",
                    "content": [
                        {"type": vis_type, vis_type: base64_vis},
                        {"type": "text", "text": txt},
                    ],
                }])
                
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            if self.cfg.runner_cfg.device_map != "auto":
                inputs = inputs.to(self.model.device) # "cuda"
            
                
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            ret_c = []
            for output in output_text:
                try:
                    if "I am " in output and "% confident in my answer." in output:
                        ret_c.append(float(output.split('I am ')[-1].split('%')[0]))
                    else:
                        ret_c.append(float(output.split('2) ')[-1].split('\n')[-1][-5:]))
                except:
                    ret_c.append(0.000)
            # ret_c = [float(output.split('2) ')[-1].split('\n')[-1]) for output in output_text]
            
            
            ################ get output
            messages = []
            for vis, txt in zip(vision, text_inputs):
                txt = txt.replace(
                    "1) What is the answer?\n2) Print how confident you are in your answer, between 0.000 and 1.000.\nAnswer: ",
                    "Answer: The answer is "
                )
                # shape: such as [n_frms, 640, 480] or [640, 480]
                vis_type = "video" if type(vis) == list else "image"
                # vis_type = "video" if vis.ndim == 3 else "image" 
                if vis_type == "video":
                    base64_vis = ndarrays_to_base64(vis, add_prefix=True)
                else:
                    base64_vis = ndarrays_to_base64([vis], add_prefix=True)[0]
                    
                messages.append([{
                    "role": "user",
                    "content": [
                        {"type": vis_type, vis_type: base64_vis},
                        {"type": "text", "text": txt},
                    ],
                }])
                
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            if self.cfg.runner_cfg.device_map != "auto":
                inputs = inputs.to(self.model.device) # "cuda"
            
                
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            from utils.llava_answer_eval import map_prediction_to_answer
            
            print('output_text, ret_c:')
            from pprint import pprint
            pprint(output_text, width=300)
            pprint(ret_c, width=300)
            return output_text, ret_c
            

            # ret_o, ret_c = [], []
            # for output in output_text:
            #     if "1)" in output and "2)" in output:
            #         for option in ["A", "B", "C", "D", "E"]:
            #             if f"{option})" in output:
            #                 ret_o.append('(' + option + ')')
            #                 break
            #         else:
            #             ret_o.append(output)
                        
            #         ret_c.append(float(output.split('2) ')[-1]))
            #     else:
            #         for option in ["A", "B", "C", "D", "E"]:
            #             if f"{option}" in output:
            #                 ret_o.append('(' + option + ')')
            #                 ret_c.append(0.000)
            #                 break
            #         else:
            #             pass
            #         ret_o.append(output)
            #         ret_c.append(float(output.split('2) ')[-1]))
            
            # print('ret_o, ret_c:', ret_o, ret_c)
            
            # return ret_o, ret_c
            
            # if "1)" in output_text[0] and "2)" in output_text[0]:
            #     if "A)" in output_text[0]:
            print(output_text)
            # import pdb; pdb.set_trace()
            return output_text, [float(t.split('2) ')[-1]) for t in output_text]
            
        else:
            try:
                if "Video-LLaVA" in self.model_name:
                    # import pdb; pdb.set_trace()
                    # vision = [np.array(v) for v in vision] # len(vision), len(vision[0]), vision[0][0].shape
                    # video_llava_prompt = "USER: <video>\n{text_input}"
                    # video_llava_text_inputs = [video_llava_prompt.format(text_input=text_input.replace("Answer: The answer is ", "ASSISTANT: ")) for text_input in text_inputs]
                    video_llava_text_inputs = [f'USER: <video>\n{text_input.replace("Answer: The answer is ", "Answer with one of (A), (B), (C), (D), or (E). ASSISTANT: ")}' for text_input in text_inputs]
                    inputs = self.processor(videos=vision, text=video_llava_text_inputs, return_tensors="pt", padding=True)
                else:
                    inputs = self.processor(vision, text_inputs, return_tensors="pt", padding=True)
            except:
                
            # if isinstance(vision[0], Image.Image):
            #     # [bsz, W, H] -> [bsz, 3, 224, 224]     | [64, 640, 480] -> [64, 3, 224, 224]
            #     inputs = self.processor(vision, text_inputs, return_tensors="pt", padding=True)
            # elif isinstance(vision[0], np.ndarray): # video. type: List[np.ndarray]
            #     inputs = self.processor(vision, text=text_inputs, return_tensors="pt", padding=True)
            #     # inputs = self.processor(videos=images, text=text_inputs, return_tensors="pt", padding=True)
            # # elif isinstance(images[0], list): # video. type: List[List[np.ndarray]]
            #     # inputs = self.processor(images, text=text_inputs, return_tensors="pt", padding=True)
            # else: #isinstance(images[0], PIL.Image): # video. type: List[Image.Image]
                # images: [bsz, n_frms, W, H] = [8, 5, 1024, 768]
                inputs = self.processor(text=text_inputs, return_tensors="pt", padding=True) # [64, 29]

                pixel_values = []
                for video in vision: # video: [n_frms, 640, 480]
                    # [n_frms, 640, 480] -> [n_frms, 3, 224, 224]
                    pixel_values.append(self.processor(images=video, return_tensors="pt", padding=True)['pixel_values'])  # [n_frms, 3, 224, 224]
                # [n_frms, 3, 224, 224] -> [bsz, n_frms, 3, 224, 224]
                stacked = torch.stack(pixel_values, dim=0)#.to(self.model.device)
                # 미적용중.."""# [bsz, n_frms, 3, 224, 224] -> [bsz, 3, n_frms, 224, 224]"""
                inputs["pixel_values"] = stacked#.transpose(2, 1)

            # inputs = self.processor(images, text_inputs, return_tensors="pt", padding=True).to(self.device)
            # out = self.model.generate(**inputs)
            # return self.processor.batch_decode(out, skip_special_tokens=True)
            
            if self.cfg.runner_cfg.device_map != "auto":
                inputs = inputs.to(self.model.device) # "cuda"
            else:
                inputs = inputs.to(dtype=torch.float16)
                # inputs = inputs.to("cpu")
                
            # debug
            if self.cfg.runner_cfg.debug:
                print(self.model.device)
                for k, v in inputs.items():
                    print(k, type(v), v.device)
                import pdb; pdb.set_trace()
            
            generation_params = {
                "do_sample": generate_sub_q,
                "min_new_tokens": 1,
                "max_new_tokens": 100 if generate_sub_q else 10,
                "return_dict_in_generate": True,
                "output_scores": True,
                # "clean_up_tokenization_spaces": True,
            }
            if beam_search:
                generation_params["num_beams"] = 5
                generation_params["length_penalty"] = -1
            else:
                generation_params["top_p"] = 0.8
                # generation_params["temperature"] = 0.9
                # generation_params["top_k"] = 50
                
            # print('inputs.device:', inputs.device)
            # print('self.model.device:', self.model.device)
            # import pdb; pdb.set_trace()
            
            outputs = self.model.generate(
                **inputs,
                **generation_params
            )
            output_text = self.processor.batch_decode(
                outputs.sequences, skip_special_tokens=True
            )
            try:
                # <class 'transformers.generation.utils.BeamSearchEncoderDecoderOutput'>
                # odict_keys(['sequences', 'sequences_scores', 'scores', 'beam_indices'])
                output_scores = torch.exp(outputs.sequences_scores).tolist()
            except: # beam_search is False. GenerateEncoderDecoderOutput
                output_scores = None

        if "Video-LLaVA" in self.model_name:# and not generate_sub_q:
            _output_text = []
            for i, o in zip(text_inputs, output_text):
                _output_text.append(o.replace(i.replace("Answer: The answer is ", "Answer with one of (A), (B), (C), (D), or (E). ASSISTANT: "), "")[7:].strip())
                
            # print(_output_text)
            # import pdb; pdb.set_trace()
            output_text = _output_text

        return output_text, output_scores


def demo():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl").to("cuda")  # , device_map="auto")
    device = model.device

    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    questions = ["Question: How many dogs are in the picture?",
                 "Question: How many ships are in the picture?"]
    inputs = processor([raw_image, raw_image], questions,return_tensors="pt", padding=True).to("cuda")  # , torch.float16)

    out = model.generate(**inputs)
    print(out)
    print(processor.batch_decode(out, skip_special_tokens=True))
