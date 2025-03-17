"""
Chat template utilities for different models.
Two cases: OpenAI API (returns message list) or HuggingFace models (uses tokenizer).
"""

from typing import List, Dict, Any, Optional, Union


def apply_chat_template(
    prompts: Union[str, List[str]],
    tokenizer: Any,
    model_type: Optional[str] = None,
    system_prompt: Optional[str] = None,
    add_generation_prompt: bool = True,
) -> Union[str, List[str], List[Dict], List[List[Dict]]]:
    """
    Apply chat template to prompts.
    
    Args:
        prompts: Single prompt or list of prompts
        tokenizer: Tokenizer from VLLM (required for HuggingFace models)
        model_type: Model type to apply chat template (huggingface|openai)
        system_prompt: Optional system prompt
        add_generation_prompt: Whether to add generation prompt (HF models only)
        
    Returns:
        - For HuggingFace models: Templated string(s)
        - For OpenAI API: List of message dicts
    """
    single_input = isinstance(prompts, str)
    if single_input:
        prompts = [prompts]
    
    if model_type == "openai":
        # OpenAI API: return message format
        result = [_format_openai(p, system_prompt) for p in prompts]
    else:
        # HuggingFace models: use tokenizer's chat template
        result = [
            _apply_tokenizer_template(p, tokenizer, system_prompt, add_generation_prompt) 
            for p in prompts
        ]
    
    return result[0] if single_input else result


def _apply_tokenizer_template(
    prompt: str,
    tokenizer: Any,
    system_prompt: Optional[str],
    add_generation_prompt: bool,
) -> str:
    """Apply template using HuggingFace tokenizer."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _format_openai(
    prompt: str,
    system_prompt: Optional[str],
) -> List[Dict[str, Any]]:
    """Format for OpenAI API. Returns message list."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages
