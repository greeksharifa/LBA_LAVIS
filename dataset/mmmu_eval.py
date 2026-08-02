import re
from typing import List, NamedTuple, Optional, Union


NormalizedAnswer = Union[str, float]


class _ConclusionEvent(NamedTuple):
    position: int
    choice: Optional[str]


class _ChoiceSpan(NamedTuple):
    start: int
    end: int


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

_TERMINAL_PUNCTUATION = r"[.,!?;:'\"]*"
_EXPLICIT_TERMINAL_PUNCTUATION = r"[.!?;:'\"]*"
_COORDINATED_CONTINUATION_SOURCE = rf"""
    \s*(?:\*\*\s*)?{_TERMINAL_PUNCTUATION}\s*
    (?:>\s*)*
    (?:(?:and|or)\b|[,/&])\s*
    (?:>\s*)*
    (?:(?:option|choice)\s+)?
    (?:\*\*\s*)?(?:\(\s*)?[A-Z](?![A-Z])
"""
_COORDINATED_ALTERNATIVE = rf"(?!{_COORDINATED_CONTINUATION_SOURCE})"
_BARE_EXPLICIT_COMPLETION = rf"""
(?=\s*(?:
    [.!?]+\s+\S
    |{_TERMINAL_PUNCTUATION}\s*$
    |(?:because|since|as)\b
    |(?:explanation|reasoning)\s*:
))
"""
_COORDINATED_CONTINUATION = re.compile(
    _COORDINATED_CONTINUATION_SOURCE,
    re.IGNORECASE | re.VERBOSE,
)
_COORDINATED_SEPARATOR = re.compile(
    rf"""
    \s*(?:\*\*\s*)?{_TERMINAL_PUNCTUATION}\s*
    (?:>\s*)*(?:(?:and|or)\b|[,/&])\s*(?:>\s*)*
    """,
    re.IGNORECASE | re.VERBOSE,
)
_EXPLANATORY_GENERIC_PREDICATE = re.compile(
    r"""
    \s*(?:
        based\s+(?:on|upon)
        |supported\s+by
        |derived\s+from
        |calculated\s+(?:from|using|by)
        |obtained\s+(?:from|using|by)
        |consistent\s+with
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_WHOLE_CHOICE = re.compile(
    rf"""
    ^\s*(?:
        \*\*\s*\(\s*([A-Z])(?![A-Z])\s*\)\s*{_TERMINAL_PUNCTUATION}\s*\*\*
        |\*\*\s*([A-Z])(?![A-Z])\s*{_TERMINAL_PUNCTUATION}\s*\*\*
        |\(\s*([A-Z])(?![A-Z])\s*\)\s*{_TERMINAL_PUNCTUATION}
        |([A-Z])(?![A-Z])\s*{_TERMINAL_PUNCTUATION}
    )\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_EXPLICIT_MARKER = re.compile(
    r"""
    \b(?:
        (?P<strong>(?:final|correct)\s+answer\s*(?:is\b|:)|answer\s*:)
        |(?:(?P<referential>this|that)\s+|the\s+)?
         (?P<generic>answer\s+is\b)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_EXPLICIT_CHOICE_PAYLOAD = re.compile(
    rf"""
    \s*(?:>\s*)?(?:
        (?:option|choice)\s+(?:
            \*\*\s*([A-Z])(?![A-Z]){_COORDINATED_ALTERNATIVE}
                \s*[.):]\s+\S(?:(?!\*\*).)*\*\*
            |\*\*\s*\(\s*([A-Z])(?![A-Z])\s*\)\s*{_EXPLICIT_TERMINAL_PUNCTUATION}\s*\*\*
            |\*\*\s*([A-Z])(?![A-Z])\s*{_EXPLICIT_TERMINAL_PUNCTUATION}\s*\*\*
            |\(\s*([A-Z])(?![A-Z])\s*\)\s*{_EXPLICIT_TERMINAL_PUNCTUATION}
            |([A-Z])(?![A-Z)*])\s*{_EXPLICIT_TERMINAL_PUNCTUATION}
        )
        |\*\*\s*([A-Z])(?![A-Z]){_COORDINATED_ALTERNATIVE}
            \s*[.):]\s+\S(?:(?!\*\*).)*\*\*
        |\*\*\s*\(\s*([A-Z])(?![A-Z])\s*\)\s*{_EXPLICIT_TERMINAL_PUNCTUATION}\s*\*\*
        |\*\*\s*([A-Z])(?![A-Z])\s*{_EXPLICIT_TERMINAL_PUNCTUATION}\s*\*\*
        |\(\s*([A-Z])(?![A-Z])\s*\)\s*{_EXPLICIT_TERMINAL_PUNCTUATION}
        |([A-Z])(?![A-Z]){_BARE_EXPLICIT_COMPLETION}
    ){_COORDINATED_ALTERNATIVE}
    """,
    re.IGNORECASE | re.VERBOSE,
)
_MARKDOWN_MARKER_CHOICE_PAYLOAD = re.compile(
    rf"""
    \s*(?:>\s*)?(?:
        ([A-Z])(?![A-Z]){_COORDINATED_ALTERNATIVE}
            \s*[.):]\s+\S(?:(?!\*\*).)*\*\*
        |([A-Z])(?![A-Z])\s*{_EXPLICIT_TERMINAL_PUNCTUATION}\s*\*\*
    ){_COORDINATED_ALTERNATIVE}
    """,
    re.IGNORECASE | re.VERBOSE,
)
_BOXED_MARKER = re.compile(r"(?<!\\)\\boxed\b", re.IGNORECASE)
_BOXED_CHOICE = re.compile(
    rf"""
    (?<!\\)\\boxed\s*\{{\s*(?:
        ([A-Z])(?![A-Z])\s*
        |([A-Z])(?![A-Z]){_COORDINATED_ALTERNATIVE}
            \s*[.):]\s+\S[^}}\n]*
    )\}}
    """,
    re.IGNORECASE | re.VERBOSE,
)
_LEADING_CHOICE = re.compile(
    r"""
    ^\s*(?:
        \*\*\s*\(\s*([A-Z])(?![A-Z])\s*\)\s*[.:]?\s*\*\*
        |\*\*\s*([A-Z])(?![A-Z])\s*[.):]\s*\*\*
        |\(\s*([A-Z])(?![A-Z])\s*\)\s*[.:]?
        |([A-Z])(?![A-Z])\s*[.):]
    )(?=\s+\S)
    """,
    re.IGNORECASE | re.VERBOSE,
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
            key_responses = candidates

    if key_responses:
        # Choose the shortest tail within the final answer-bearing subresponse.
        return [min(reversed(key_responses), key=len)]
    return [response]


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


def _matched_choice(match: re.Match) -> str:
    return next(group for group in match.groups() if group).lower()


def _coordinated_choice_events(
    response: str, choice_spans: List[_ChoiceSpan]
) -> List[_ConclusionEvent]:
    choice_spans = sorted(choice_spans)
    return [
        _ConclusionEvent(second.end, None)
        for first, second in zip(choice_spans, choice_spans[1:])
        if _COORDINATED_SEPARATOR.fullmatch(
            response[first.end : second.start]
        )
    ]


def parse_multiple_choice_response(response: str) -> Optional[str]:
    if not isinstance(response, str) or not response.strip():
        return None

    whole = _WHOLE_CHOICE.fullmatch(response)
    if whole:
        return _matched_choice(whole)

    events = []
    choice_spans = []
    for marker in _EXPLICIT_MARKER.finditer(response):
        choice = _EXPLICIT_CHOICE_PAYLOAD.match(response, marker.end())
        choice_start = marker.start()
        if (
            choice is None
            and marker.start() >= 2
            and response[marker.start() - 2 : marker.start()] == "**"
        ):
            choice = _MARKDOWN_MARKER_CHOICE_PAYLOAD.match(
                response, marker.end()
            )
            choice_start -= 2
        if choice:
            choice_spans.append(_ChoiceSpan(choice_start, choice.end()))
            events.append(
                _ConclusionEvent(marker.start(), _matched_choice(choice))
            )
            continuation = _COORDINATED_CONTINUATION.match(
                response, choice.end()
            )
            if continuation:
                events.append(_ConclusionEvent(continuation.end(), None))
        elif marker.group("strong") or (
            not marker.group("referential")
            and not _EXPLANATORY_GENERIC_PREDICATE.match(
                response, marker.end()
            )
        ):
            events.append(_ConclusionEvent(marker.start(), None))

    for marker in _BOXED_MARKER.finditer(response):
        choice = _BOXED_CHOICE.match(response, marker.start())
        if choice:
            choice_spans.append(_ChoiceSpan(marker.start(), choice.end()))
            events.append(
                _ConclusionEvent(marker.start(), _matched_choice(choice))
            )
            continuation = _COORDINATED_CONTINUATION.match(
                response, choice.end()
            )
            if continuation:
                events.append(_ConclusionEvent(continuation.end(), None))
        else:
            events.append(_ConclusionEvent(marker.start(), None))

    events.extend(_coordinated_choice_events(response, choice_spans))
    if events:
        return max(events, key=lambda event: event.position).choice

    leading = _LEADING_CHOICE.match(response)
    if leading:
        if _COORDINATED_CONTINUATION.match(response, leading.end()):
            return None
        return _matched_choice(leading)
    return None


def evaluate_multiple_choice(gold: str, prediction: str) -> bool:
    parsed_prediction = parse_multiple_choice_response(prediction)
    if parsed_prediction is None:
        return False
    return parsed_prediction == _normalize_option_letter(gold)


def evaluate_answer(prediction: str, gold: str, question_type: str) -> bool:
    normalized_type = question_type.lower().replace("-", "_")
    if normalized_type in ("open", "open_ended"):
        return evaluate_open(gold, prediction)
    if normalized_type == "multiple_choice":
        return evaluate_multiple_choice(gold, prediction)
    raise ValueError(f"unsupported MMMU question type: {question_type}")
