import os
import ast
import json
from typing import List
import numpy as np
import pandas as pd
import torch
import pickle
import string
from pprint import pprint

from torch.utils.data import Dataset
from transformers import InstructBlipVideoProcessor
from utils.llava_answer_eval import map_prediction_to_answer

def load_dataset(datasets_cfg, split='val', n_supple=0, ann_paths=[], **kwargs):#xl_or_xxl="xl", model_tag=None):
    if datasets_cfg.dataset_name == "VQA_Introspect":
        from dataset.VQA_Introspect import VQAIntrospectDataset
        cls = VQAIntrospectDataset
    elif datasets_cfg.dataset_name == "AOKVQA":
        from dataset.AOKVQA import AOKVQADataset
        cls = AOKVQADataset
    elif datasets_cfg.dataset_name == "OKVQA":
        from dataset.OKVQA import OKVQADataset
        cls = OKVQADataset
    elif datasets_cfg.dataset_name == "PathVQA":
        from dataset.PathVQA import PathVQADataset
        cls = PathVQADataset
    elif datasets_cfg.dataset_name == "SLAKE":
        from dataset.SLAKE import SLAKEDataset
        cls = SLAKEDataset
    elif datasets_cfg.dataset_name == "ArtVQA":
        from dataset.ArtVQA import ArtVQADataset
        cls = ArtVQADataset
    elif datasets_cfg.dataset_name == "WinogroundVQA":
        from dataset.WinogroundVQA import WinogroundVQADataset
        cls = WinogroundVQADataset
    elif datasets_cfg.dataset_name == "GQA":
        from dataset.GQA import GQADataset
        cls = GQADataset
    elif datasets_cfg.dataset_name == "VQA_rad":
        from dataset.VQA_rad import VQA_radDataset
        cls = VQA_radDataset
    elif datasets_cfg.dataset_name == "DramaQA":
        from dataset.DramaQA import DramaQAEvalDataset
        cls = DramaQAEvalDataset
    elif datasets_cfg.dataset_name in ["NExTQA", "STAR"]:#, "VLEP"]:# "How2QA"]:
        from dataset.VideoQA import VideoEvalDataset
        cls = VideoEvalDataset
    elif datasets_cfg.dataset_name == "TVQA": 
        from dataset.TVQA import TVQAEvalDataset
        cls = TVQAEvalDataset
    elif datasets_cfg.dataset_name == "VLEP": # '/data/VLEP/sevila_style/vlep_frames/friends_s03e09_seg02_clip_07_ep.mp4'
        from dataset.VLEP import VLEPEvalDataset
        cls = VLEPEvalDataset
    elif datasets_cfg.dataset_name == "IntentQA":
        from dataset.IntentQA import IntentQAEvalDataset
        cls = IntentQAEvalDataset
    elif datasets_cfg.dataset_name == "EgoSchema":
        from dataset.EgoSchema import EgoSchemaEvalDataset
        cls = EgoSchemaEvalDataset
    elif datasets_cfg.dataset_name == "ActivityNetQA":
        from dataset.ActivityNetQA import ActivityNetQADataset
        cls = ActivityNetQADataset
    elif datasets_cfg.dataset_name == "MSVDQA":
        from dataset.MSVDQA import MSVDQADataset
        cls = MSVDQADataset
    elif datasets_cfg.dataset_name == "MSRVTTQA":
        from dataset.MSRVTTQA import MSRVTTQADataset
        cls = MSRVTTQADataset
    elif datasets_cfg.dataset_name == "MMMU":
        from dataset.MMMU import MMMUDataset
        cls = MMMUDataset
    elif datasets_cfg.dataset_name == "MME":
        from dataset.MME import MMEDataset
        cls = MMEDataset
    else:
        raise NotImplementedError(f"in dataset.base_dataset.py, load_dataset() | Invalid dataset name: {datasets_cfg.dataset_name}")

    # ann_paths = [os.path.join(datasets_cfg.root_dir, path) for path in datasets_cfg.ann_paths.get(datasets_cfg.split, split)]
    # # ann_paths = [path.replace("xl", xl_or_xxl) for path in ann_paths]
    # if len(ann_paths) >= 2:
    #     if os.path.exists(ann_paths[1].replace("xl", model_tag)):
    #         ann_paths[-1] = ann_paths[-1].replace("xl", model_tag)
    #     else:
    #         ann_paths[-1] = ann_paths[-1].replace("xl", xl_or_xxl)
        
    dataset = cls(
        vis_processor=None,
        text_processor=None,
        vis_root=os.path.join(datasets_cfg.root_dir, datasets_cfg.vis_root),
        ann_paths=ann_paths,
        num_data=datasets_cfg.num_data,
        vqa_acc=datasets_cfg.vqa_acc,
        n_frms=datasets_cfg.get("n_frms", 4),
        datasets_cfg=datasets_cfg,
        n_supple=n_supple, #datasets_cfg.get("n_supple"),
        data_type=datasets_cfg.data_type,
        open_ended=datasets_cfg.open_ended,
        split=split,
        **kwargs
    )
    
    return dataset
    

class BaseDataset(Dataset):
    def __init__(self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[], num_data=-1, **kwargs):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.annotation = []
                
        if len(ann_paths) == 1:
            ann_path = ann_paths[0]
        elif len(ann_paths) == 2:
            ann_path, sub_qas_path = ann_paths
            if os.path.exists(sub_qas_path):
                self.sub_qas = json.load(open(sub_qas_path, 'r'))
        else:
            raise ValueError(f"Invalid ann_paths: {ann_paths}")
        
        # ann
        # for ann_path in ann_paths:
        if any(ext in ann_path for ext in ['csv', 'tsv']):
            df = pd.read_csv(ann_path)
            self.annotation.extend(df.to_dict(orient="records"))
        elif 'jsonl' in ann_path:
            with open(ann_path, "r") as f:
                self.annotation.extend([json.loads(line) for line in f])
        elif 'pkl' in ann_path:
            with open(ann_path, 'rb') as f:
                self.annotation = pickle.load(f)
        else:
            with open(ann_path, "r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    self.annotation.extend(loaded)
                elif isinstance(loaded, dict):
                    self.annotation.extend([{"sample_id": k, **v} if isinstance(v, dict) else {"sample_id": k, "data": v} for k, v in loaded.items()])

        if num_data != -1:
            if num_data < len(self.annotation):
                # uniform_sampling
                idxs = np.linspace(0, len(self.annotation)-1, num_data, dtype=int)
                self.annotation = [self.annotation[i] for i in idxs]
                # self.annotation = self.annotation[:num_data]

        self.vis_processor = vis_processor
        self.text_processor = text_processor
        
        # self.vqa_acc = vqa_acc
        for k, v in kwargs.items():
            setattr(self, k, v)

        self._add_instance_ids()
        
        print("\n" + self.__class__.__name__)
        print('vis_processor : ', vis_processor)
        print('text_processor : ', text_processor)
        print('vis_root : ', vis_root)
        print('ann_paths : ', ann_paths)
        print('type(self.annotation), len(self.annotation):', type(self.annotation), len(self.annotation))
        self.cnt = 0

    def create_openai_client(self):
        api_key = json.load(open("temp/api_key.json", "r"))["LBA"]
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        
        """
        self.response_list:
        [
            {
                "main_question": ...,
                "outputs": ...,
                "targets": ...,
                "pred": "yes" or "no",
                "score": 0.0 ~ 5.0
            },
            {
                "main_question": "what are three people sitting on?",
                "outputs": "The three people in the video are sitting on a couch.",
                "targets": "couch",
                "pred": "yes",
                "score": 5
            }
        ]
        """
        path = os.path.join(self.output_dir, "response_df.csv")
        if os.path.exists(path):
            self.response_df = pd.read_csv(path) 
            # self.response_list = json.load(open(path, 'r'))
        else:
            required_columns = [
                "main_question",
                "outputs",
                "targets",
                "pred",
                "score"
            ]
            self.response_df = pd.DataFrame(columns=required_columns)
            # self.response_list = []
            
    def save_response_list(self):
        save_path = os.path.join(self.output_dir, "response_df.csv")
        self.response_df.to_csv(save_path, index=False)
        # json.dump(self.response_list, open(save_path, 'w'), indent=4)
        return save_path
    
    def get_openai_eval_response(self, pred_answer, gt_answer, main_question):
        message = [
            {
                "role": "system",
                "content":
                    "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                    "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                    "------"
                    "##INSTRUCTIONS: "
                    "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                    "- Consider synonyms or paraphrases as valid matches.\n"
                    "- Evaluate the correctness of the prediction compared to the answer."
            },
            {
                "role": "user",
                "content":
                    "Please evaluate the following video-based question-answer pair:\n\n"
                    f"Question: {main_question}\n"
                    f"Correct Answer: {gt_answer}\n"
                    f"Predicted Answer: {pred_answer}\n\n"
                    "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                    "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
            }
        ]
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message,
            temperature=0.7,
            max_tokens=800,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        # import pdb; pdb.set_trace()
        response_message = completion.choices[0].message.content
        response_dict = ast.literal_eval(response_message)
        
        return response_dict

    @staticmethod
    def answer_mapping(answer):
        return answer

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        result = {}
        for k, v in samples[0].items():
            # if isinstance(v, torch.Tensor): # no use, 240809
            #     result[k] = torch.stack([sample[k] for sample in samples], dim=0)
            # else:
            result[k] = [sample[k] for sample in samples]
        
        return result

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id", prefix=""):
        # for i in range(len(self.annotation)):
        #     self.annotation[i][key] = prefix + str(i)
        for idx, ann in enumerate(self.annotation):
            ann[key] = prefix + str(idx)
            
    def get_e_cr_e_ic(self, acc_origin_list, acc_lba_list):
        if self.vqa_acc:
            e_cr = sum([1 if acc_lba > acc_origin and acc_origin < 0.5 else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if acc < 0.5 else 0 for acc in acc_origin_list]) * 100
            e_ic = sum([1 if acc_lba < acc_origin and acc_origin > 0.5 else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if acc > 0.5 else 0 for acc in acc_origin_list]) * 100
        else:
            try:
                e_cr = sum([1 if acc_lba and not acc_origin else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if not acc_origin else 0 for acc_origin in acc_origin_list]) * 100
            except:
                e_cr = 0.
            try:
                e_ic = sum([1 if not acc_lba and acc_origin else 0 for acc_origin, acc_lba in zip(acc_origin_list, acc_lba_list)]) / sum([1 if acc_origin else 0 for acc_origin in acc_origin_list]) * 100
            except:
                e_ic = 0.
        return e_cr, e_ic
    
    def get_accuracy(self, outputs, targets, main_question=None):#, match1ok=False):
        """
        args
        - outputs: str          or list of str.         shape: [bsz]
        - targets: list of str  or list of list of str. shape: [bsz, 10]
        """
        # eval_chatgpt
        if hasattr(self, "eval_chatgpt") and hasattr(self, "client"):
            # Apply the condition to filter the DataFrame
            matching_row = self.response_df[(self.response_df["main_question"] == main_question) & 
                            (self.response_df["outputs"] == outputs) & 
                            (self.response_df["targets"] == targets)]
            if not matching_row.empty:
                response_dict = {
                    "pred": matching_row.iloc[0]["pred"],
                    "score": matching_row.iloc[0]["score"]
                }
            else:
                response_dict = self.get_openai_eval_response(pred_answer=outputs, gt_answer=targets, main_question=main_question)

                self.response_df = pd.concat([self.response_df, pd.DataFrame([{
                    "main_question": main_question,
                    "outputs": outputs,
                    "targets": targets,
                    "pred": response_dict["pred"],
                    "score": response_dict["score"]
                }])], ignore_index=True)
            
            
            '''
            for saved_response in self.response_list:
                if saved_response["main_question"] == main_question and saved_response["outputs"] == outputs and saved_response["targets"] == targets:
                    response_dict = {
                        "pred": saved_response["pred"],
                        "score": saved_response["score"]
                    }
                    break
                    # return response_dict["pred"]
            else:
                # print('start', '-' * 80)
                # pprint(saved_response, width=300)
                # print('end ', '-' * 80)
                response_dict = self.get_openai_eval_response(pred_answer=outputs, gt_answer=targets, main_question=main_question)
                self.response_list.append({
                    "main_question": main_question,
                    "outputs": outputs,
                    "targets": targets,
                    "pred": response_dict["pred"],
                    "score": response_dict["score"]
                })
                # return response_dict["pred"]
            '''
            
            if response_dict["pred"] == "yes":
                return 1.0
            else:
                return 0.0
        
        def _get_acc(out, target):
            if not self.open_ended: # True or False
                if type(out) == str:
                    out = map_prediction_to_answer(out)
                
            # convert to lower case string
            out = str(out).lower().rstrip('.').rstrip(',')
            
            if self.vqa_acc:
                assert isinstance(target, list), f"Invalid target type (expected list): {type(target)}, {target}"
                target = [str(t).lower().rstrip('.').rstrip(',') for t in target]
                return 1.0 if out in target else 0.0
            else:
                target = str(target).lower().rstrip('.').rstrip(',')
                if not self.open_ended and target in string.ascii_lowercase + string.ascii_uppercase:
                    target = '(' + target + ')'
                # if self.cnt < 3:
                #     print('out, target : ', out, target)
                #     self.cnt += 1
                return 1.0 if out == target else 0.0
            
            
        if not isinstance(outputs, list):# isinstance(outputs, (str, int)):
            acc = _get_acc(outputs, targets)
            if "no" in outputs.lower() and "no" in targets and acc < 0.5:
                import pdb; pdb.set_trace()
            return acc
        else:
            acc_list = []
            # import pdb; pdb.set_trace()
            for out, target_list in zip(outputs, targets):
                acc = _get_acc(out, target_list)
                acc_list.append(acc)
            return acc_list
    
    
def get_train_examplar(datasets_cfg):
    
    train_dataset = load_dataset(datasets_cfg, split='train')
    example = train_dataset[0]
    
    main_question = example["text_input"].strip().rstrip('?')
    candidate_list = example["candidate_list"]
    gt_ans = train_dataset.answer_mapping(example["gt_ans"])
    answer_sentence = example["answer_sentence"].strip().rstrip('.')
    
    prompt = """Context: {main_question}? {answer_sentence}.\nQuestion: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is {gt_ans}"""
    for candidate in candidate_list:
        choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
    
    return prompt.format(main_question=main_question, answer_sentence=answer_sentence, choices=choices, gt_ans=gt_ans)
            
def get_text_input(
    prompt_type:str="default",
    main_questions:List[str]=[],
    sub_questions:List[str]=[],
    sub_answers:List[str]=[],
    candidate_lists: List[List[str]]=[],
    gt_answers:List[str]=[],
    question_ids:List[str]=[],
    examplar: str="",
    model_name="",
    **kwargs,
):
    if "image" in prompt_type or "video" in prompt_type: 
        # print("default_image")
        # print(candidate_lists)
        '''
        Context:\n{sub_qas}                                                 | if recomposer
        {main_question}?                                                    | every case
        {choices}\n                                                         | if multi-choice
        Answer with the option's letter from the given choices directly.    | if multi-choice
        Answer the question using a single word or phrase.                  | if open-ended
        
        '''
        ret = []
        for i in range(len(main_questions)):            
            # context
            if "recomposer" in prompt_type: # recomposer_image
                sub_question = sub_questions[i]
                sub_answer = sub_answers[i]
                sub_qas = ""
                if isinstance(sub_question, str):
                    sub_question = [sub_question]
                    sub_answer = [sub_answer]
                for sq, sa in zip(sub_question, sub_answer):
                    sub_qas += f"{sq.rstrip('?')}? {sa.rstrip('.')}.\n"
                prompt = f"Context:\n{sub_qas}\n"
            else:                           # decomposer_image
                prompt = ""
                
            # main_question
            main_question = main_questions[i]
            prompt += f"{main_question.rstrip('?')}?\n"
            
            # choices and instructions
            if candidate_lists and candidate_lists[i] is not None:
                candidate_list = candidate_lists[i]
                # choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
                choices = '\n'.join([f"{chr(65+i)}. {c}" for i, c in enumerate(candidate_list)])
                prompt += f"{choices}\n"
                prompt += "Answer with the option's letter from the given choices directly."
            else:
                prompt += "Answer the question using a single word or phrase."

            ret.append(prompt)
        
        return ret
        
    elif prompt_type == "decomposer":
        prompt = "Reasoning Question: is the banana ripe enough to eat? Perception Question: is the banana yellow?\nReasoning Question: is it cold outside? Perception Question: are any people wearing jackets?\nReasoning Question: {main_question}? Perception Question:"
        return [prompt.format(main_question=main_question.rstrip('?')) for main_question in main_questions]
    
    elif prompt_type == "sub_answer":
        if "llava-hf/llava-v1.6" in model_name:
            prompt = "{sub_question}?\nAnswer the question using a single word or phrase."
        else:
            prompt = "Question: {sub_question}? Short answer:"
        return [prompt.format(sub_question=sub_question.rstrip('?')) for sub_question in sub_questions]
        
    if "video" in prompt_type:
        if prompt_type == "recomposer_video_description":
            prompt = "Video Description: {description}.\nQuestion: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is "
            
            ret = []
            for description, main_question, candidate_list in zip(kwargs.get('descriptions'), main_questions, candidate_lists):
                choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
                ret.append(prompt.format(description=description, main_question=main_question.rstrip('?'), choices=choices))
            return ret
        
        # elif prompt_type == "recomposer_video_irrelevant_info":
            
        else:
            pass # TODO
            prompt = examplar + "Context: {irr_info}.\nQuestion: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is "
            
            ret = []
            for irr_info, main_question, candidate_list in zip(kwargs.get('irr_info_list'), main_questions, candidate_lists):
                choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
                ret.append(prompt.format(irr_info=irr_info, main_question=main_question.rstrip('?'), choices=choices))
            return ret
    
    elif prompt_type == "default_video":
        if "Qwen" in model_name:
            prompt = "Question: {main_question}?\nChoices:\n{choices}\n"
            prompt += "1) What is the answer?\n"
            prompt += "2) Print how confident you are in your answer, between 0 and 100.\n"
            # prompt += "Example answer: (A), 0.857\n"
            prompt += "Answer: "
        else:
            prompt = examplar + "Question: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is "
            # prompt = "Question: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is "
        
        ret = []
        for main_question, candidate_list in zip(main_questions, candidate_lists):
            if candidate_list is None: # open-ended
                # prompt = prompt.replace("Choices:\n{choices}\nAnswer: The answer is ", "Short answer: ")
                prompt = prompt.replace("Choices:\n{choices}\n", "")
                ret.append(prompt.format(main_question=main_question.rstrip('?')))
            else:                      # multi-choice
                choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
                ret.append(prompt.format(main_question=main_question.rstrip('?'), choices=choices))
        return ret
    
    elif prompt_type == "recomposer_video":
        if "Qwen" in model_name:
            prompt = "Context:\n{sub_qas}Question: {main_question}?\nChoices:\n{choices}\n"
            prompt += "1) What is the answer?\n"
            prompt += "2) Print how confident you are in your answer, between 0 and 100.\n"
            # prompt += "Example answer: (A), 0.857\n"
            prompt += "Answer: "
        else:
            # prompt += "Context: {sub_question}? {sub_answer}.\nQuestion: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is "
            prompt = examplar + "Context:\n{sub_qas}Question: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is "
        
        ret = []
        for main_question, sub_question, sub_answer, candidate_list in zip(main_questions, sub_questions, sub_answers, candidate_lists):
            sub_qas = ""
            if isinstance(sub_question, str):
                sub_question = [sub_question]
                sub_answer = [sub_answer]
            for sq, sa in zip(sub_question, sub_answer):
                sub_qas += f"{sq.rstrip('?')}? {sa.rstrip('.')}.\n"
                
            if candidate_list is None: # open-ended
                # prompt = prompt.replace("Choices:\n{choices}\nAnswer: The answer is ", "Short answer: ")
                prompt = prompt.replace("Choices:\n{choices}\n", "")
                ret.append(prompt.format(main_question=main_question.rstrip('?'), sub_qas=sub_qas))
            else:                      # multi-choice
                choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
                ret.append(prompt.format(main_question=main_question.rstrip('?'), sub_qas=sub_qas, choices=choices))
            
        return ret
    
    else:
        raise NotImplementedError(f"Invalid prompt type: {prompt_type}")
    

def _backup():
    """
    if prompt_type == "default_image": # for default vqa or generating sub-answer
        # print("default_image")
        # print(candidate_lists)
        if candidate_lists:

            if "llava-hf/llava-v1.6" in model_name:
                '''
                <question>
                A. <option_1>
                B. <option_2>
                C. <option_3>
                D. <option_4>
                Answer with the option's letter from the given choices directly.
                '''
                prompt = "{main_question}?\n{choices}\nAnswer with the option's letter from the given choices directly."
            else:
                prompt = "{main_question}?\n{choices}\nAnswer with the option's letter from the given choices directly."
                # prompt = "Question: {main_question}?\nChoices:\n{choices}\nAnswer with the option's letter from the given choices directly."
            ret = []
            for main_question, candidate_list in zip(main_questions, candidate_lists):
                choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
                ret.append(prompt.format(main_question=main_question.rstrip('?'), choices=choices))
            return ret
                
        else:
            if "llava-hf/llava-v1.6" in model_name:
                prompt = "{main_question}?\nAnswer the question using a single word or phrase."
            else:
                prompt = "Question: {main_question}? Short answer:"
            return [prompt.format(main_question=main_question.rstrip('?')) for main_question in main_questions]
        
      
    elif prompt_type == "recomposer_image":
        examplar = "Context: is the sky blue? no. are there clouds in the sky? yes. Question: what weather is likely? Short answer: rain.\n"
        # print("recomposer_image")
        # print(candidate_lists)
        if candidate_lists:
            # prompt = examplar + "Context:\n{sub_qas}Question: {main_question}?\nChoices:\n{choices}\nAnswer: The answer is "
            prompt = examplar + "Context:\n{sub_qas}Question: {main_question}?\nChoices:\n{choices}\nAnswer with the option's letter from the given choices directly."
        else:
            prompt = examplar + "Context:\n{sub_qas}Question: {main_question}? Answer the question using a single word or phrase."
            # prompt = examplar + "Context:\n{sub_qas}Question: {main_question}? Short answer:"
            # prompt = examplar + "Context:\n{sub_qas}Question: {main_question}\nAnswer: The answer is "
        
        ret = []
        for i in range(len(main_questions)):
            main_question = main_questions[i]
            sub_question = sub_questions[i]
            sub_answer = sub_answers[i]
            sub_qas = ""
            if isinstance(sub_question, str):
                sub_question = [sub_question]
                sub_answer = [sub_answer]
            for sq, sa in zip(sub_question, sub_answer):
                sub_qas += f"{sq.rstrip('?')}? {sa.rstrip('.')}.\n"
            if candidate_lists:
                candidate_list = candidate_lists[i]
                choices = '\n'.join([f"({chr(65+i)}) {c}" for i, c in enumerate(candidate_list)])
                ret.append(prompt.format(main_question=main_question.rstrip('?'), sub_qas=sub_qas, choices=choices))
            else:
                ret.append(prompt.format(main_question=main_question.rstrip('?'), sub_qas=sub_qas))
        return ret
        # return [prompt.format(main_question=main_question.rstrip('?'), sub_question=sub_question.rstrip('?'), sub_answer=sub_answer.rstrip('.')) 
                # for main_question, sub_question, sub_answer in zip(main_questions, sub_questions, sub_answers)]
    
    """
    pass