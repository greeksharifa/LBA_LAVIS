import re
import math
import copy

from collections import defaultdict

from typing import List, Any, Dict, Union

from model.protocol import generation_result_from_vllm


def postprocess_subqs(output_texts: str, N: int) -> List[str]:
    """
    Postprocess the sub-questions.
    Args:
        output_texts        : str
        N                   : int
    Returns:
        sub_questions       : List[str]
    """
    DEFAULT_SUBQS = [
        "Can you describe any observable characteristics or attributes related to this situation?",
        "What seems to be the general meaning or situation described here?",
        "What might have caused this event or statement to occur?",
        "What could be the possible result or effect of this event?",
        "What might the person's intention or goal be in this context?",
    ]
    '''default_sub_questions = [
        {
            "category": "Feature specification",
            "default_sub_q": "Can you describe any observable characteristics or attributes related to this situation?",
            "note": "시각적 정보(표정, 장면)가 없을 경우 텍스트 기반 속성이나 언어적 단서로 대체 가능."
        },
        {
            "category": "Interpretation",
            "default_sub_q": "What seems to be the general meaning or situation described here?",
            "note": "장면이 없어도 텍스트 내용만으로 사건의 전반적 의미를 추론하도록 유도."
        },
        {
            "category": "Causal antecedent",
            "default_sub_q": "What might have caused this event or statement to occur?",
            "note": "원인 정보가 없을 경우, 일반적인 배경이나 가능성을 묻는 형태."
        },
        {
            "category": "Causal consequence",
            "default_sub_q": "What could be the possible result or effect of this event?",
            "note": "결과가 명시되지 않아도 합리적 추론을 유도할 수 있는 일반형."
        },
        {
            "category": "Goal orientation",
            "default_sub_q": "What might the person's intention or goal be in this context?",
            "note": "인물, 화자 등의 목적을 묻는 일반형. 시각 정보가 없어도 텍스트로 충분히 적용 가능."
        }
    ]
    '''
    # import pdb; pdb.set_trace()
    result = []
    raw_texts = output_texts.split("The decomposed sub-questions for Q is:")[-1]
    # print(f"b: {b:2d} raw_texts: {raw_texts}")
    raw_texts = raw_texts.split('\n')
    # filter empty lines
    raw_texts = [raw_text for raw_text in raw_texts if raw_text.strip()]

    # split by '?'
    for raw_text in raw_texts:
        sub_qs = raw_text.split('?')
        for sub_q in sub_qs:
            if sub_q.strip():
                result.append(sub_q.strip() + '?')
    
    result = result[:N]

    # filter numbers
    result = [re.sub(r'\d+\.\s*', '', sub_q) for sub_q in result]
    # import pdb; pdb.set_trace()
    if len(result) < N:
        result.extend(DEFAULT_SUBQS[:N - len(result)])
        # result.extend([result[-1]] * (self.N - len(result)))

    return result

def postprocess_bases(output_text: str) -> str:
    """
        Postprocess the base answer.
        Args:
            output_text         : str
        Returns:
            base_answer         : str
    """
    # The answer is
    output_text = output_text.split("The answer is")[-1].split("the answer is")[-1]
    return output_text

def format_vllm_outputs(
    mode: str, 
    outputs: List[Any], 
    qids: List[str], 
    N: int,
) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Format the vLLM outputs.
    Args:
        mode                : str.       모드 (subq, suba, base, refined)
        outputs             : List[Any]. vLLM outputs. model.generate()의 결과 리스트
        qids                : List[str]. 각 output에 대응하는 query ID 리스트 (outputs와 순서가 같아야 함)
        N                   : int.      
    Returns:
        merged              : Dict[str, Dict[str, Any]] or something. merged results.
    """
    assert mode in ["subq", "suba", "base", "refined"], f"Invalid mode: {mode}"
    
    KEY_NAME = {
        "output_text": {
            "subq": "subq_list",
            "suba": "suba_list",
            "base": "base_answer",
            "refined": "refined_answer_list"
        },
    }
    key_name = KEY_NAME["output_text"][mode]

    merged = {}

    # outputs와 qids를 순서대로 매핑
    for output, qid in zip(outputs, qids):
        generation = generation_result_from_vllm(output)
        seq_ppl = generation.confidence["seq_ppl"]
        token_min_prob = generation.confidence["token_min_prob"]

        # output_text postprocess
        if mode == "subq":
            output_text = postprocess_subqs(generation.text, N)
        elif mode == "suba":
            output_text = generation.text   # postprocess_subas(generation.text)
        # elif mode == "refined":
        #     output_text = postprocess_refineds(completion.text)
        else: # "base"
            output_text = postprocess_bases(generation.text)


        # 3. 결과 딕셔너리 생성
        result_item = {
            qid: {
                key_name: output_text,
                f"conf_{mode}": {
                    "seq_ppl": seq_ppl,
                    "token_min_prob": token_min_prob
                },
            }
        }
        
        # 4. 병합을 수행하는 재귀 함수 (반복문 안에서 호출)
        def update_recursive(target_dict, source_item):
            """
            target_dict: 누적된 결과를 저장하는 딕셔너리 (merged)
            source_item: 새로 생성된 결과 딕셔너리 (result_item)
            """
            for key, value in source_item.items():
                if key in target_dict:
                    # Case 1: 둘 다 딕셔너리인 경우 -> 더 깊이 재귀 호출
                    if isinstance(target_dict[key], dict) and isinstance(value, dict):
                        update_recursive(target_dict[key], value)
                    
                    # Case 2: 값 충돌 발생 (Leaf Node) -> 리스트로 변환 및 추가
                    else:
                        # 기존 값이 리스트가 아니면 리스트로 변환
                        if not isinstance(target_dict[key], list):
                            target_dict[key] = [target_dict[key]]
                        
                        # 새로운 값 추가
                        target_dict[key].append(value)
                else:
                    # 타겟에 키가 없는 경우 그대로 추가
                    target_dict[key] = copy.deepcopy(value)
        
        update_recursive(merged, result_item)

    if mode in ("suba", "refined"):
        for result in merged.values():
            if not isinstance(result[key_name], list):
                result[key_name] = [result[key_name]]
            confidence = result[f"conf_{mode}"]
            for confidence_name, value in confidence.items():
                if not isinstance(value, list):
                    confidence[confidence_name] = [value]

    return merged
