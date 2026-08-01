import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from PIL import Image

from dataset.MMMU import MMMU
from prompt.prompts import get_base_prompt
from util.utils import create_answer_mapping


def make_cfg(*, mode="base"):
    return SimpleNamespace(
        runner_cfg=SimpleNamespace(mode=mode),
        dataset_cfg=SimpleNamespace(
            question_type="both",
            vqa_acc=False,
            prompt=SimpleNamespace(add_base_prompt=True),
        ),
    )


def make_adapter(*, mode="base"):
    dataset = MMMU.__new__(MMMU)
    dataset.cfg = make_cfg(mode=mode)
    dataset.ANSWER_MAPPING = create_answer_mapping()
    dataset.annotation = []
    return dataset


class MMMUOpenTests(unittest.TestCase):
    def test_annotation_load_canonicalizes_source_open_question_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            Image.new("RGB", (1, 1)).save(image_path)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    [
                        {
                            "answer": "1.06",
                            "image_path_list": [str(image_path)],
                            "options": [],
                            "question": "What is the result? <image 1>",
                            "question_id": "open-1",
                            "question_type": "open",
                            "subfield": "Engineering",
                        }
                    ]
                )
            )
            dataset = make_adapter()

            dataset.load_annotation([annotation_path])

            self.assertEqual("open_ended", dataset.annotation[0]["question_type"])
            dataset.annotation[0]["image_list"][0].close()

    def test_open_sample_gets_short_answer_prompt(self):
        sample = {
            "main_q": "What is the result?",
            "candidate_list": [],
            "question_type": "open",
        }

        prompt = get_base_prompt(sample, make_cfg())

        self.assertIn("single word or phrase", prompt)
        self.assertNotIn("option's letter", prompt)

    def test_numeric_open_answers_use_mmmu_normalization_and_conclusion_parsing(self):
        dataset = make_adapter()

        for prediction in (
            "1,234.50",
            "1234.5",
            "Therefore, the final answer is 1,234.50.",
        ):
            with self.subTest(prediction=prediction):
                self.assertEqual(
                    1,
                    dataset.get_score(prediction, "1234.5", "open_ended"),
                )

    def test_string_open_answers_normalize_case_and_spaces_with_substring_semantics(self):
        dataset = make_adapter()

        self.assertEqual(
            1,
            dataset.get_score(
                "Therefore, the answer is   NEW    YORK City.",
                "new york",
                "open_ended",
            ),
        )

    def test_final_conclusion_overrides_conflicting_intermediate_answer(self):
        dataset = make_adapter()

        self.assertEqual(
            0,
            dataset.get_score(
                "The answer is 4. Therefore, the final answer is 5.",
                "4",
                "open_ended",
            ),
        )

    def test_final_conclusion_ignores_unrelated_earlier_number(self):
        dataset = make_adapter()

        self.assertEqual(
            0,
            dataset.get_score(
                "The intermediate result is 4. Therefore, the final answer is 5.",
                "4",
                "open_ended",
            ),
        )

    def test_empty_final_conclusion_does_not_fall_back_to_intermediate_answer(self):
        dataset = make_adapter()

        self.assertEqual(
            0,
            dataset.get_score(
                "The answer is 4. Therefore, the final answer is empty.",
                "4",
                "open_ended",
            ),
        )

    def test_multiline_final_conclusion_overrides_intermediate_answer(self):
        dataset = make_adapter()
        for prediction in (
            "The answer is 4.\nTherefore, the final answer is 5.",
            "The answer is 4\nTherefore, the final answer is 5.",
        ):
            with self.subTest(prediction=prediction):
                self.assertEqual(0, dataset.get_score(prediction, "4", "open_ended"))
                self.assertEqual(1, dataset.get_score(prediction, "5", "open_ended"))

    def test_multiple_choice_remains_exact_normalized_option_letter_match(self):
        dataset = make_adapter()

        self.assertEqual(1, dataset.get_score("(A).", "A", "multiple-choice"))
        self.assertEqual(
            0,
            dataset.get_score("A because it is correct", "A", "multiple-choice"),
        )

    def test_common_cleanser_does_not_truncate_source_open_decimal(self):
        dataset = make_adapter()

        self.assertEqual("1.06", dataset.cleanse_answer("1.06", "open"))

    def test_configured_annotations_have_expected_canonical_open_counts(self):
        dataset_cfg = OmegaConf.load("config/datasets/MMMU.yaml").dataset
        root_dir = Path(OmegaConf.load("config/default.yaml").dataset.root_dir)
        expected = {"val": 9, "test": 53}

        annotation_paths = {
            split: root_dir / dataset_cfg.ann_paths[split][0]
            for split in expected
        }
        missing_paths = [
            str(path) for path in annotation_paths.values() if not path.is_file()
        ]
        if missing_paths:
            self.skipTest(
                "configured MMMU annotation files are unavailable: "
                + ", ".join(missing_paths)
            )

        self.assertTrue(hasattr(MMMU, "canonicalize_question_type"))
        if not hasattr(MMMU, "canonicalize_question_type"):
            return

        for split, expected_count in expected.items():
            with self.subTest(split=split):
                annotation_path = annotation_paths[split]
                samples = json.loads(annotation_path.read_text())
                source_types = [sample["question_type"] for sample in samples]
                canonical_types = [
                    MMMU.canonicalize_question_type(sample["question_type"])
                    for sample in samples
                ]
                self.assertEqual(expected_count, source_types.count("open"))
                self.assertEqual(expected_count, canonical_types.count("open_ended"))


if __name__ == "__main__":
    unittest.main()
