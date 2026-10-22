from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch


VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".webm"}


def configure_qwen_processor(processor):
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
    return processor


def _media_content_items(media_path: Any) -> list[dict[str, str]]:
    if media_path is None:
        return []
    if isinstance(media_path, list):
        items = media_path
    else:
        items = [media_path]

    content = []
    for item in items:
        suffix = Path(str(item)).suffix.lower()
        media_type = "video" if suffix in VIDEO_SUFFIXES else "image"
        content.append({"type": media_type, media_type: str(item)})
    return content


def build_qwen_user_messages(prompt: str, image_path: Any) -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                *_media_content_items(image_path),
                {"type": "text", "text": prompt},
            ],
        }
    ]


def trim_generated_ids(
    generated_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[torch.Tensor]:
    # Decoder-only generation retains the complete padded input prefix.
    input_width = attention_mask.shape[1]
    return [row[input_width:] for row in generated_ids]


def decode_qwen_generations(
    processor,
    generated_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[str]:
    trimmed_ids = trim_generated_ids(generated_ids=generated_ids, attention_mask=attention_mask)
    return [
        decoded.strip()
        for decoded in processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]


def render_qwen_chat_prompts(
    processor,
    prompts: Sequence[str],
    image_paths: Sequence[Any],
) -> list[str]:
    rendered = []
    for prompt, image_path in zip(prompts, image_paths):
        messages = build_qwen_user_messages(prompt=prompt, image_path=image_path)
        rendered.append(
            processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return rendered
