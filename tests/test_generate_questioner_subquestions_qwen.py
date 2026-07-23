import json
from argparse import Namespace
from pathlib import Path

import torch
from PIL import Image

import generate_questioner_subquestions_qwen as module


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self):
        self.rendered_prompts = []
        self.tokenizer = type("Tokenizer", (), {"padding_side": "right"})()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prompt = messages[0]["content"][-1]["text"]
        self.rendered_prompts.append(prompt)
        return f"CHAT::{prompt}"

    def __call__(self, text, images=None, videos=None, padding=True, return_tensors="pt"):
        batch_size = len(text)
        return FakeBatch(
            {
                "input_ids": torch.ones((batch_size, 3), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, 3), dtype=torch.long),
            }
        )


class FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        batch_size = kwargs["input_ids"].shape[0]
        return torch.ones((batch_size, 4), dtype=torch.long)


class FakeProgressBar:
    def __init__(self, iterable, total=None, desc=None, disable=None):
        self.items = list(iterable)
        self.total = total
        self.desc = desc
        self.disable = disable
        self.updates = []

    def __iter__(self):
        return iter(self.items)

    def update(self, n=1):
        self.updates.append(n)

    def close(self):
        pass


def test_generate_sub_questions_for_batch_chains_previous_questions(monkeypatch, tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (4, 4), color="white").save(image_path)

    fake_processor = FakeProcessor()
    fake_model = FakeModel()
    prompt_config = {
        "init_prompt": "main-question is '{}'.",
        "after_prompt": " Previous: ",
        "pair_prompt": "sub-question-{}: '{}'",
    }
    decoded_outputs = iter(
        [
            ["first follow-up", "first other"],
            ["second follow-up", "second other"],
        ]
    )

    monkeypatch.setattr(
        module,
        "decode_qwen_generations",
        lambda processor, generated_ids, attention_mask: next(decoded_outputs),
    )
    monkeypatch.setattr(
        module,
        "process_vision_info",
        lambda messages, return_video_kwargs=True: ([["image-a"], ["image-b"]], None, {}),
    )

    generated = module.generate_sub_questions_for_batch(
        model=fake_model,
        processor=fake_processor,
        image_paths=[str(image_path), str(image_path)],
        model_questions=["where is the cat?", "what is on the table?"],
        prompt_config=prompt_config,
        num_sub_questions=2,
        num_beams=3,
        max_len=32,
        min_len=1,
    )

    assert generated == [
        ["first follow-up", "second follow-up"],
        ["first other", "second other"],
    ]
    assert fake_processor.rendered_prompts == [
        "main-question is 'where is the cat?'.",
        "main-question is 'what is on the table?'.",
        "main-question is 'where is the cat?'. Previous: sub-question-1: 'first follow-up'.",
        "main-question is 'what is on the table?'. Previous: sub-question-1: 'first other'.",
    ]


def test_generate_sub_questions_for_batch_supports_old_qwen_vl_utils_signature(monkeypatch, tmp_path):
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (4, 4), color="white").save(image_path)

    monkeypatch.setattr(
        module,
        "decode_qwen_generations",
        lambda processor, generated_ids, attention_mask: ["only output"],
    )
    monkeypatch.setattr(
        module,
        "process_vision_info",
        lambda messages: ([["image-a"]], None),
    )

    generated = module.generate_sub_questions_for_batch(
        model=FakeModel(),
        processor=FakeProcessor(),
        image_paths=[str(image_path)],
        model_questions=["where is the cat?"],
        prompt_config={
            "init_prompt": "main-question is '{}'.",
            "after_prompt": " Previous: ",
            "pair_prompt": "sub-question-{}: '{}'",
        },
        num_sub_questions=1,
        num_beams=1,
        max_len=16,
        min_len=1,
    )

    assert generated == [["only output"]]


def test_load_c2r_records_filters_qids(monkeypatch, tmp_path):
    c2r_root = tmp_path / "C2R"
    qids_root = tmp_path / "qids"
    ann_dir = c2r_root / "StrategyQA"
    ann_dir.mkdir(parents=True)
    ann_path = ann_dir / "C2R_test.json"
    ann_path.write_text(
        json.dumps(
            [
                {"qid": "keep-me", "question": "Is the sky blue?", "answer": True},
                {"qid": "drop-me", "question": "Is water dry?", "answer": False},
            ]
        )
    )
    qids_root.mkdir(parents=True)
    (qids_root / "StrategyQA.json").write_text(
        json.dumps({"dataset": "StrategyQA", "qids": ["keep-me"]})
    )

    records = module.load_c2r_records(
        dataset_name="StrategyQA",
        c2r_root=c2r_root,
        qids_root=qids_root,
        limit=None,
    )

    assert records == [
        {
            "source_index": 0,
            "question_id": "keep-me",
            "main_question": "Is the sky blue?",
            "media_path": None,
        }
    ]


def test_load_c2r_records_falls_back_to_c2r_baseline_for_mmlu(tmp_path):
    c2r_root = tmp_path / "C2R"
    qids_root = tmp_path / "qids"
    baseline_dir = c2r_root / "MMLU" / "Baseline-SingleSubQ"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "sub_qas.json").write_text(
        json.dumps(
            [
                {
                    "MMLU_0": {
                        "main_question": "What is algebra?",
                        "sub_q": ["sq"],
                        "sub_a": [],
                        "media_path": None,
                    }
                },
                {
                    "MMLU_1": {
                        "main_question": "What is geometry?",
                        "sub_q": ["sq"],
                        "sub_a": [],
                        "media_path": None,
                    }
                },
            ]
        )
    )
    qids_root.mkdir(parents=True)
    (qids_root / "MMLU.json").write_text(json.dumps({"dataset": "MMLU", "qids": ["MMLU_1"]}))

    records = module.load_c2r_records(
        dataset_name="MMLU",
        c2r_root=c2r_root,
        qids_root=qids_root,
        limit=None,
    )

    assert records == [
        {
            "source_index": 0,
            "question_id": "MMLU_1",
            "main_question": "What is geometry?",
            "media_path": None,
        }
    ]


def test_run_dataset_writes_c2r_public_payload(monkeypatch, tmp_path):
    c2r_root = tmp_path / "C2R"
    qids_root = tmp_path / "qids"
    ann_dir = c2r_root / "MMMU" / "mmmu_json"
    ann_dir.mkdir(parents=True)
    image_path = c2r_root / "images" / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color="blue").save(image_path)
    (ann_dir / "validation.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "qid-1",
                    "question": "<image 1> What is shown?",
                    "answer": "A",
                    "options": ["cat", "dog"],
                    "question_type": "multiple-choice",
                    "subfield": "biology",
                    "image_path_list": [str(image_path)],
                }
            ]
        )
    )
    qids_root.mkdir(parents=True)
    (qids_root / "MMMU.json").write_text(json.dumps({"dataset": "MMMU", "qids": ["qid-1"]}))

    monkeypatch.setattr(
        module,
        "generate_sub_questions_for_batch",
        lambda **kwargs: [["describe the animal"]],
    )

    args = Namespace(
        c2r_root=str(c2r_root),
        qids_root=str(qids_root),
        output_root=str(tmp_path / "output"),
        limit=None,
        batch_size=1,
        num_workers=0,
        num_sub_questions=1,
        num_beams=3,
        max_len=32,
        min_len=1,
        log_every=10,
    )

    result_path = module.run_dataset(
        args=args,
        dataset_name="MMMU",
        model=FakeModel(),
        processor=FakeProcessor(),
        prompt_config={
            "init_prompt": "main-question is '{}'.",
            "after_prompt": " Previous: ",
            "pair_prompt": "sub-question-{}: '{}'",
        },
        job_id="job123",
    )

    assert result_path == str(Path(args.output_root) / "C2R" / "MMMU" / module.MODEL_OUTPUT_NAME / "sub_qas.json")
    assert json.loads(Path(result_path).read_text()) == [
        {
            "qid-1": {
                "main_question": "<image 1> What is shown?",
                "sub_q": ["describe the animal"],
                "sub_a": [],
                "media_path": [str(image_path)],
            }
        }
    ]


def test_load_qwen_model_and_processor_uses_local_rank_device_map(monkeypatch):
    captured = {}

    class FakeQwenModel:
        def eval(self):
            return self

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_name):
            return FakeProcessor()

    class FakeQwenLoader:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs
            return FakeQwenModel()

    monkeypatch.setattr(module, "configure_qwen_processor", lambda processor: processor)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(module, "__import__", __import__, raising=False)

    import types
    fake_transformers = types.SimpleNamespace(
        AutoProcessor=FakeAutoProcessor,
        Qwen2_5_VLForConditionalGeneration=FakeQwenLoader,
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    model, processor = module.load_qwen_model_and_processor(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype="auto",
        attn_implementation="sdpa",
    )

    assert captured["kwargs"]["device_map"] == {"": 3}


def test_run_dataset_uses_tqdm_progress_bar(monkeypatch, tmp_path):
    c2r_root = tmp_path / "C2R"
    qids_root = tmp_path / "qids"
    ann_dir = c2r_root / "StrategyQA"
    ann_dir.mkdir(parents=True)
    (ann_dir / "C2R_test.json").write_text(json.dumps([{"qid": "q1", "question": "Q?", "answer": True}]))
    qids_root.mkdir(parents=True)
    (qids_root / "StrategyQA.json").write_text(json.dumps({"dataset": "StrategyQA", "qids": ["q1"]}))

    fake_bars = []

    def fake_tqdm(iterable, total=None, desc=None, disable=None):
        bar = FakeProgressBar(iterable, total=total, desc=desc, disable=disable)
        fake_bars.append(bar)
        return bar

    monkeypatch.setattr(module, "tqdm", fake_tqdm)
    monkeypatch.setattr(module, "generate_sub_questions_for_batch", lambda **kwargs: [["sq"]])

    args = Namespace(
        c2r_root=str(c2r_root),
        qids_root=str(qids_root),
        output_root=str(tmp_path / "output"),
        limit=None,
        batch_size=1,
        num_workers=0,
        num_sub_questions=1,
        num_beams=1,
        max_len=8,
        min_len=1,
        log_every=10,
    )

    module.run_dataset(
        args=args,
        dataset_name="StrategyQA",
        model=FakeModel(),
        processor=FakeProcessor(),
        prompt_config={
            "init_prompt": "main-question is '{}'.",
            "after_prompt": " Previous: ",
            "pair_prompt": "sub-question-{}: '{}'",
        },
        job_id="job",
    )

    assert fake_bars
    assert fake_bars[0].desc == "StrategyQA rank0"
