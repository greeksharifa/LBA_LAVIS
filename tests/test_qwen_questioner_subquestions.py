import torch

from subqa.qwen_questioner_subquestions import (
    build_qwen_user_messages,
    configure_qwen_processor,
    trim_generated_ids,
)


def test_build_qwen_user_messages_wraps_image_and_prompt():
    messages = build_qwen_user_messages(
        prompt="Generate a useful sub-question.",
        image_path="/tmp/example.jpg",
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/tmp/example.jpg"},
                {"type": "text", "text": "Generate a useful sub-question."},
            ],
        }
    ]


def test_trim_generated_ids_removes_full_left_padded_input_width():
    generated_ids = torch.tensor(
        [
            [10, 11, 12, 21, 22],
            [0, 30, 31, 41, 42],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1],
            [0, 1, 1],
        ]
    )

    trimmed = trim_generated_ids(generated_ids, attention_mask)

    assert [row.tolist() for row in trimmed] == [
        [21, 22],
        [41, 42],
    ]


def test_configure_qwen_processor_sets_left_padding():
    class Tokenizer:
        padding_side = "right"

    class Processor:
        def __init__(self):
            self.tokenizer = Tokenizer()

    processor = Processor()

    configured = configure_qwen_processor(processor)

    assert configured.tokenizer.padding_side == "left"
