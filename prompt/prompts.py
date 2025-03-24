from config.configs import Config
from typing import List

def get_base_prompt(sample: dict, cfg: Config) -> str:
    """
        Generate base answer for the main question.
        Args:
            sample     : dict
            cfg        : Config
        Returns:
            prompt     : str
        ================================================================
        ---------------- # open-ended     ---------------
        <main question>
        Answer the question using a single word or phrase.
        ---------------- # multiple-choice --------------
        <main question>
        A. <option 1>
        ...
        {Z}. <option {Z}>
        Answer with the option's letter from the given choices directly.
    """
    import pdb; pdb.set_trace()
    if sample["question_type"] == "open_ended":
        prompt = f"{sample['main_q']}"
        if cfg.runner_cfg.mode == "CoT":
            prompt += "\nYou should think about the problem-solving process step by step, and at the end, output the final answer in the format 'The answer is <final answer>.'"
            prompt += "\nThe <final answer> must be a single word or phrase."
            prompt += "\nLet's think step by step."
        # elif cfg.runner_cfg.mode == "StrategyQA":
        #     prompt += "\nAnswer with yes or no."
        if cfg.dataset_cfg.prompt.add_base_prompt:
            prompt += "\nAnswer the question using a single word or phrase."

    else: # multiple_choice
        prompt = ""
        if cfg.runner_cfg.mode == "CoT":
            prompt += "The following are multiple choice questions (with answers). Think step by step and then finish your answer with \"The answer is <X>\" where <X> is the correct letter choice."

        prompt += f"{sample['main_q']}\n"
        for i, candidate in enumerate(sample["candidate_list"]):
            prompt += f"{chr(65 + i)}. {candidate}\n"

        if cfg.runner_cfg.mode == "CoT":
            prompt += "\nLet's think step by step."
        elif cfg.dataset_cfg.prompt.add_base_prompt:
            prompt += "\nAnswer with the option's letter from the given choices directly."
            
    return prompt
    
def get_suba_prompt(sample: dict, cfg: Config) -> List[str]:
    """
        Generate N sub-answers for the N sub-questions.
        Args:
            sample     : dict
            cfg        : Config
        Returns:
            prompt     : List[str]
            
        sample (MMMU):
        {
            'candidate_list': ['$6', '$7', '$8', '$9'],
            'conf_subq_list': 0.4997720632523633,
            'data_type': 'image',
            'gt_ans': 'b',
            'main_q': '<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the '
                    'per unit manufacturing overhead costs incurred?',
            'qid': 'validation_Accounting_1',
            'question_type': 'multiple-choice',
            'subq_list': ['What are the fixed manufacturing overhead costs per unit at 25,000 units of production?',
                        'How does the fixed manufacturing overhead cost behave when production increases from 25,000 to 30,000 units?',
                        'What is the total fixed manufacturing overhead cost at 25,000 units of production?',
                        'What is the total fixed manufacturing overhead cost at 30,000 units of production?',
                        'What is the per unit manufacturing overhead cost at 30,000 units of production, considering both fixed and variable components?'],
            'vision': [<PIL.PngImagePlugin.PngImageFile image mode=RGBA size=733x237 at 0x7FC5C07BEFC0>],
            'vpath': ['/data/MMMU/mmmu_images/validation/validation_Accounting_1_1.png']
        }
    """
    sub_qs = sample["subq_list"]
    prompts = []

    for sub_q in sub_qs:
        prompt = f"{sub_q}\nAnswer in a maximum of one sentence."
        prompts.append(prompt)

    return prompts


# def get_subq_prompt(prompt_type: str, main_q: str, data_type: str, N: int) -> str:
def get_subq_prompt(sample: dict, cfg: Config) -> str:
    """
        Generate N sub-questions for the main question.
        Args:
            sample     : dict
            cfg        : Config
        Returns:
            prompt     : str
    """
    prompt_type = cfg.runner_cfg.subqa_mode
    main_q = sample["main_q"]
    data_type = sample["data_type"]
    N = cfg.runner_cfg.N

    if prompt_type == "self":
        prompt = """### Instruction
Your task is to decompose a given question (or instruction) Q into sub-questions.
You need to generate {N} sub-questions that will help you answer the given Q. 
"""
        if data_type == "video" or data_type == "image":
            prompt += f"Also, a single or multiple {data_type}(s) may be given. Given Q, you need to generate sub-questions considering what to focus on in the {data_type}(s).\n"

        prompt += """Please note that: You should output ONLY multiple sub-questions as shown in the following format.
### Format:
1. ...
2. ...
...
{N}. ...

### Input
The given question (or instruction) Q : '''{main_q}'''

The decomposed sub-questions for Q is:
"""
        
        return prompt.format(main_q=main_q, N=N)
    
    elif prompt_type == "QC_Q":
        question_categories = [
            {"id": 1,  "category": "Verification",             "example": "Which object was taken by the person?"},
            {"id": 2,  "category": "Case specification",       "example": "How did Phoebe feel in this scene?"},
            {"id": 3,  "category": "Concept completion",       "example": "What did Chandler just say?"},
            {"id": 4,  "category": "Feature specification",    "example": "What is the expression on House's face?"},
            {"id": 5,  "category": "Quantification",           "example": "How many objects were taken by the person?"},
            {"id": 6,  "category": "Definition",               "example": "What type of object is the closet/cabinet?"},
            {"id": 7,  "category": "Example",                  "example": "What is an example of a utility program?"},
            {"id": 8,  "category": "Comparison",               "example": "Which object was not involved in the action?"},
            {"id": 9,  "category": "Interpretation",           "example": "What is happening in the scene?"},
            {"id": 10, "category": "Causal antecedent",        "example": "What caused Phoebe to close her eyes?"},
            {"id": 11, "category": "Causal consequence",       "example": "What is the result of Monica moving a cabinet?"},
            {"id": 12, "category": "Goal orientation",         "example": "What action did the person perform with the blanket?"},
            {"id": 13, "category": "Instrumental/Procedural",  "example": "What method does Dokyung use to communicate?"},
            {"id": 14, "category": "Enablement",               "example": "What allows operating systems to translate things?"},
            {"id": 15, "category": "Expectational",            "example": "What might Chandler expect after refusing the call?"},
            {"id": 16, "category": "Judgemental",              "example": "What could be Deogi's opinion on Haeyoung1's choice?"},
            {"id": 17, "category": "Assertion",                "example": "Is a cup or tumbler better?"},
            {"id": 18, "category": "Request/Directive",        "example": "Can you hold a cup for me?"},
        ]
        question_categories_selected = [
            {"id": 4,  "category": "Feature specification",   "example": "What is the expression on House’s face?"},
            {"id": 9,  "category": "Interpretation",          "example": "What is happening in the scene?"},
            {"id": 10, "category": "Causal antecedent",       "example": "What caused Phoebe to close her eyes?"},
            {"id": 11, "category": "Causal consequence",      "example": "What is the result of Monica moving a cabinet?"},
            {"id": 12, "category": "Goal orientation",        "example": "What action did the person perform with the blanket?"},
        ]

        header = "category | example"
        lines = [f"{item['category']} | {item['example']}" for item in question_categories_selected] # question_categories
        question_category_type = header + "\n" + "\n".join(lines)

        prompt = """### Instruction
Your task is to decompose a given question (or instruction) Q into sub-questions.
Based on the information including question category types, you need to generate {N} sub-questions that will help you answer the given Q. 
"""
        if data_type == "videos" or data_type == "images":
            prompt += f"Also, a single or multiple {data_type[:-1]}(s) may be given. Given Q, you need to generate sub-questions considering what to focus on in the {data_type[:-1]}(s).\n"

        prompt += """### Input
Question Category type: '''
{question_category_type}
'''
The given question (or instruction) Q : '''{main_q}'''
"""
        prompt += """Please note that: You should output ONLY multiple sub-questions as shown in the following format.
### Format:
1. ...
2. ...
...
{N}. ...

The decomposed sub-questions for Q is:
"""
        
        return prompt.format(main_q=main_q, N=N, question_category_type=question_category_type)