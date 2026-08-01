import re
from typing import List, Union


NormalizedAnswer = Union[str, float]

_NUMBER_WITH_COMMAS = r"-?\b\d{1,3}(?:,\d{3})+\b"
_SCIENTIFIC_NUMBER = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
_SIMPLE_NUMBER = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
_KEY_INDICATORS = (
    "could be ",
    "so ",
    "is ",
    "thus ",
    "therefore ",
    "final ",
    "answer ",
    "result ",
)


def _normalize_string(value: str) -> List[NormalizedAnswer]:
    value = " ".join(value.strip().split())
    try:
        return [round(float(value.replace(",", "")), 2)]
    except ValueError:
        value = value.lower()
        if len(value) == 1:
            return [f" {value}", f"{value} "]
        return [value]


def _extract_numbers(value: str) -> List[str]:
    return (
        re.findall(_NUMBER_WITH_COMMAS, value)
        + re.findall(_SCIENTIFIC_NUMBER, value)
        + re.findall(_SIMPLE_NUMBER, value)
    )


def _key_subresponses(response: str) -> List[str]:
    response = response.strip().strip(".").lower()
    subresponses = re.split(r"\.\s+(?=[A-Z])|\n", response)
    key_responses = []

    for index, subresponse in enumerate(subresponses):
        lowered = subresponse
        indicators = _KEY_INDICATORS + (("=",) if index == len(subresponses) - 1 else ())
        candidates = [
            lowered.split(indicator)[-1].strip()
            for indicator in indicators
            if indicator in lowered
        ]
        candidates = [
            candidate
            for candidate in candidates
            if candidate not in ("", ":", ",", ".", ";", "!", "?", "'")
        ]
        if candidates:
            key_responses.append(min(candidates, key=len))

    return key_responses or [response]


def parse_open_response(response: str) -> List[NormalizedAnswer]:
    key_responses = _key_subresponses(response)
    candidates = list(key_responses)
    for key_response in key_responses:
        candidates.extend(_extract_numbers(key_response))

    normalized = []
    for candidate in candidates:
        for value in _normalize_string(candidate):
            if value not in normalized:
                normalized.append(value)
    return normalized


def evaluate_open(gold: Union[str, List[str]], prediction: str) -> bool:
    gold_answers = gold if isinstance(gold, list) else [gold]
    normalized_gold = []
    for answer in gold_answers:
        normalized_gold.extend(_normalize_string(answer))

    for predicted in parse_open_response(prediction):
        if isinstance(predicted, str):
            if any(
                isinstance(answer, str) and answer in predicted
                for answer in normalized_gold
            ):
                return True
        elif predicted in normalized_gold:
            return True
    return False


def _normalize_option_letter(value: str) -> str:
    value = value.strip().rstrip(".,!?;:'\"").strip()
    match = re.fullmatch(r"\(?([A-Za-z])\)?", value)
    return match.group(1).lower() if match else value.lower()


def evaluate_multiple_choice(gold: str, prediction: str) -> bool:
    return _normalize_option_letter(prediction) == _normalize_option_letter(gold)


def evaluate_answer(prediction: str, gold: str, question_type: str) -> bool:
    normalized_type = question_type.lower().replace("-", "_")
    if normalized_type in ("open", "open_ended"):
        return evaluate_open(gold, prediction)
    if normalized_type == "multiple_choice":
        return evaluate_multiple_choice(gold, prediction)
    raise ValueError(f"unsupported MMMU question type: {question_type}")
