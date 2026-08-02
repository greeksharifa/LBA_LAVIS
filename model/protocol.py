"""Lightweight generation result protocol shared by model adapters."""

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationResult:
    text: str
    confidence: dict[str, float]

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise ValueError("generation result text must be a string")
        if not isinstance(self.confidence, Mapping):
            raise ValueError("generation result confidence must be a mapping")
        normalized = {}
        for name, value in self.confidence.items():
            if (
                not isinstance(name, str)
                or not name
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    "generation result confidence must contain finite numeric metrics"
                )
            normalized[name] = float(value)
        object.__setattr__(self, "confidence", normalized)


def generation_result_from_vllm(output) -> GenerationResult:
    """Adapt one vLLM request output without leaking its shape to callers."""
    completions = getattr(output, "outputs", None)
    if not isinstance(completions, (list, tuple)) or len(completions) != 1:
        raise ValueError("vLLM output must contain exactly one completion")
    completion = completions[0]
    text = getattr(completion, "text", None)
    token_ids = getattr(completion, "token_ids", None)
    cumulative_logprob = getattr(completion, "cumulative_logprob", None)
    logprobs = getattr(completion, "logprobs", None)
    if not isinstance(text, str) or not isinstance(token_ids, (list, tuple)):
        raise ValueError("malformed vLLM completion")

    if token_ids:
        if isinstance(cumulative_logprob, bool) or not isinstance(
            cumulative_logprob, (int, float)
        ):
            raise ValueError("vLLM cumulative_logprob must be numeric")
        seq_ppl = math.exp(-float(cumulative_logprob) / len(token_ids))
    else:
        seq_ppl = 0.0

    minimum_probability = 1.0
    has_token_probability = False
    if logprobs:
        if len(logprobs) != len(token_ids):
            raise ValueError("vLLM token/logprob count mismatch")
        for token_id, token_logprobs in zip(token_ids, logprobs):
            if token_logprobs and token_id in token_logprobs:
                token_logprob = getattr(token_logprobs[token_id], "logprob", None)
                if isinstance(token_logprob, bool) or not isinstance(
                    token_logprob, (int, float)
                ):
                    raise ValueError("vLLM token logprob must be numeric")
                has_token_probability = True
                minimum_probability = min(
                    minimum_probability, math.exp(float(token_logprob))
                )

    return GenerationResult(
        text=text,
        confidence={
            "seq_ppl": seq_ppl,
            "token_min_prob": (
                minimum_probability if has_token_probability else 0.0
            ),
        },
    )

