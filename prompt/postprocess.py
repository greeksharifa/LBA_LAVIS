import re
import math

from typing import List, Any, Dict


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

def format_vllm_outputs(
    mode: str, 
    outputs: List[Any], 
    qids: List[str], 
    N: int,
) -> List[Dict[str, Any]]:
    """
    Format the vLLM outputs.
    Args:
        mode                : str.       모드 (subq, suba, base, refined)
        outputs             : List[Any]. vLLM outputs. model.generate()의 결과 리스트
        qids                : List[str]. 각 output에 대응하는 query ID 리스트 (outputs와 순서가 같아야 함)
        N                   : int.      
    Returns:
        formatted_results   : List[Dict[str, Any]]
    """
    assert mode in ["subq", "suba", "base", "refined"], f"Invalid mode: {mode}"

    formatted_results = {}

    # outputs와 qids를 순서대로 매핑
    for output, qid in zip(outputs, qids):
        # vLLM은 n=1일 때 outputs[0]에 결과를 담음
        completion = output.outputs[0]
        
        # 1. Sequence Perplexity (PPL) 계산
        cumulative_logprob = completion.cumulative_logprob
        num_tokens = len(completion.token_ids)
        
        if num_tokens > 0:
            seq_ppl = math.exp(-cumulative_logprob / num_tokens)
        else:
            seq_ppl = 0.0

        # 2. Token Minimum Probability 계산
        # 초기값 1.0 (확률은 0~1 사이)
        min_prob = 1.0
        has_tokens = False

        if completion.logprobs:
            for idx, token_logprob_dict in enumerate(completion.logprobs):
                token_id = completion.token_ids[idx]
                if token_id in token_logprob_dict:
                    has_tokens = True
                    log_p = token_logprob_dict[token_id].logprob
                    prob = math.exp(log_p)
                    if prob < min_prob:
                        min_prob = prob
        
        # 토큰이 하나도 없었다면 min_prob는 0.0 처리 (안전장치)
        token_min_prob = min_prob if has_tokens else 0.0


        key_name = {
            "output_text": {
                "subq": "sub_qs",
                "suba": "sub_as",
                "base": "base_answer",
                "refined": "refined_answer"
            },
        }

        # 3. 결과 딕셔너리 생성
        result_item = {
            # "qid": qid,
            key_name["output_text"][mode]: postprocess_subqs(completion.text, N),
            f"conf_{mode}": {
                "seq_ppl": seq_ppl,
                "token_min_prob": token_min_prob
            },
        }
        formatted_results[qid] = result_item
    
    return formatted_results
