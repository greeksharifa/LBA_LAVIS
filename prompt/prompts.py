# from configs import Config
from typing import List

def get_suba_prompt(prompt_type: str, subq_list: List[str], data_type: str, N: int) -> str:


def get_subq_prompt(prompt_type: str, main_q: str, data_type: str, N: int) -> str:
    """
    Generate sub-questions for the main question.
    Args:
        prompt_type: str # [self, QC_Q]
        main_q     : str
        data_type  : str # [texts|images|videos|features]
    Returns:
        prompt     : str
    """
    if prompt_type == "self":
        prompt = """### Instruction
Your task is to decompose a given question (or instruction) Q into sub-questions.
You need to generate {N} sub-questions that will help you answer the given Q. 
"""
        if data_type == "videos" or data_type == "images":
            prompt += f"Also, a single or multiple {data_type[:-1]}(s) may be given. Given Q, you need to generate sub-questions considering what to focus on in the {data_type[:-1]}(s).\n"

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