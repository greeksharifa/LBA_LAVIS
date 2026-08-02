import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
from PIL import Image

from dataset.MMMU import MMMU
from dataset.mmmu_eval import parse_multiple_choice_response
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

    def test_last_answer_line_wins_even_when_final_answer_is_longer(self):
        dataset = make_adapter()
        prediction = (
            "The answer is 4\n"
            "Therefore, the final answer is New York City."
        )

        self.assertEqual(0, dataset.get_score(prediction, "4", "open_ended"))
        self.assertEqual(
            1,
            dataset.get_score(prediction, "New York City", "open_ended"),
        )

    def test_adapter_canonicalizes_stringified_open_answer_alternatives(self):
        source_answers = (
            "['$MgS$', 'MgS']",
            "['Tampa', 'Florida']",
            "['24/7', '3.429']",
            "['Tampa', broken]",
        )
        expected_answers = (
            ["$mgs$", "mgs"],
            ["tampa", "florida"],
            ["24/7", "3.429"],
            "['tampa', broken]",
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            Image.new("RGB", (1, 1)).save(image_path)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    [
                        {
                            "answer": answer,
                            "image_path_list": [str(image_path)],
                            "options": [],
                            "question": f"Question {index}? <image 1>",
                            "question_id": f"open-{index}",
                            "question_type": "open",
                            "subfield": "Fixture",
                        }
                        for index, answer in enumerate(source_answers)
                    ]
                )
            )
            dataset = make_adapter()

            dataset.load_annotation([annotation_path])

            try:
                self.assertEqual(
                    list(expected_answers),
                    [ann["gt_ans"] for ann in dataset.annotation],
                )
            finally:
                for ann in dataset.annotation:
                    for image in ann["image_list"]:
                        image.close()

    def test_open_scorer_accepts_each_canonical_answer_alternative(self):
        dataset = make_adapter()

        for prediction, gold in (
            ("MgS", ["$mgs$", "mgs"]),
            ("Tampa", ["tampa", "florida"]),
            ("3.429", ["24/7", "3.429"]),
        ):
            with self.subTest(prediction=prediction):
                self.assertEqual(
                    1,
                    dataset.get_score(prediction, gold, "open_ended"),
                )

        self.assertEqual(
            0,
            dataset.get_score("Tampa", "['tampa', broken]", "open_ended"),
        )

    def test_multiple_choice_accepts_supported_answer_formats(self):
        dataset = make_adapter()

        for prediction in (
            "A",
            "(A).",
            "A. $6",
            "A) explanation",
            "(A) explanation",
            "**A.** $6",
            "The answer is A.",
            "answer: option A",
            "Final answer: option A",
            "final answer is (A)",
            "final answer is a",
            r"\boxed{A}",
        ):
            with self.subTest(prediction=prediction):
                self.assertEqual(
                    1,
                    dataset.get_score(prediction, "A", "multiple-choice"),
                )

    def test_multiple_choice_accepts_artifact_markdown_conclusions(self):
        dataset = make_adapter()

        for prediction, expected in (
            ("**Answer: D**", "d"),
            ("### ✅ Final Answer: **B. t=2.46s**", "b"),
            ("### ✅ Correct Answer: **D. $75,283**", "d"),
            (
                "### Final Answer:\n\n> **A. $y=...$**\n\n"
                "✅ **Answer: A**",
                "a",
            ),
            ("The answer is **C**", "c"),
        ):
            with self.subTest(prediction=prediction):
                self.assertEqual(
                    expected,
                    parse_multiple_choice_response(prediction),
                )
                self.assertEqual(
                    1,
                    dataset.get_score(
                        prediction, expected.upper(), "multiple-choice"
                    ),
                )

    def test_multiple_choice_rejects_ambiguous_markdown_conclusions(self):
        dataset = make_adapter()

        for prediction, possible_gold in (
            ("Answer: **A or B**", "AB"),
            ("Final Answer: **A and B**", "AB"),
            ("**Final Answer: (A)** or **B**", "AB"),
            ("**Final Answer: (A)** and **B**", "AB"),
            ("**Final Answer: (A)** / **B**", "AB"),
            ("**Final Answer: (A)** & **B**", "AB"),
            (
                "Final Answer:\n> **A. first**\n> or **B. second**",
                "AB",
            ),
        ):
            with self.subTest(prediction=prediction):
                self.assertIsNone(parse_multiple_choice_response(prediction))
                for gold in possible_gold:
                    self.assertEqual(
                        0,
                        dataset.get_score(prediction, gold, "multiple-choice"),
                    )

        malformed_final = "Answer: **A**. Final Answer: **unknown**"
        self.assertIsNone(parse_multiple_choice_response(malformed_final))

    def test_multiple_choice_accepts_delimited_boxed_label_only(self):
        dataset = make_adapter()
        prediction = r"Final Answer: \boxed{C. option text}"

        self.assertEqual("c", parse_multiple_choice_response(prediction))
        self.assertEqual(
            1,
            dataset.get_score(prediction, "C", "multiple-choice"),
        )

        for ambiguous in (r"\boxed{A or B}", r"\boxed{157}"):
            with self.subTest(prediction=ambiguous):
                self.assertIsNone(parse_multiple_choice_response(ambiguous))

    def test_multiple_choice_final_answer_overrides_intermediate_answer(self):
        dataset = make_adapter()
        prediction = "The answer is B. After checking, final answer: C."

        for gold, expected in (("C", 1), ("B", 0)):
            with self.subTest(gold=gold):
                self.assertEqual(
                    expected,
                    dataset.get_score(prediction, gold, "multiple-choice"),
                )

    def test_multiple_choice_later_malformed_conclusion_invalidates_earlier(self):
        dataset = make_adapter()
        prediction = "The answer is B. After checking, final answer: unknown."

        self.assertIsNone(parse_multiple_choice_response(prediction))
        for gold in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            with self.subTest(gold=gold):
                self.assertEqual(
                    0,
                    dataset.get_score(prediction, gold, "multiple-choice"),
                )

    def test_multiple_choice_rejects_ambiguous_explicit_bare_letters(self):
        dataset = make_adapter()

        for prediction, possible_gold in (
            ("The answer is a complex expression.", "A"),
            ("The answer is A or B.", "AB"),
            ("Final answer: C and D.", "CD"),
        ):
            with self.subTest(prediction=prediction):
                self.assertIsNone(parse_multiple_choice_response(prediction))
                for gold in possible_gold:
                    self.assertEqual(
                        0,
                        dataset.get_score(prediction, gold, "multiple-choice"),
                    )

    def test_multiple_choice_accepts_delimited_bare_conclusion_with_explanation(self):
        dataset = make_adapter()

        for prediction in (
            "The answer is A. This follows from the calculation.",
            "The answer is a. This follows from the calculation.",
        ):
            with self.subTest(prediction=prediction):
                self.assertEqual(
                    "a",
                    parse_multiple_choice_response(prediction),
                )
                self.assertEqual(
                    1,
                    dataset.get_score(prediction, "A", "multiple-choice"),
                )

    def test_multiple_choice_accepts_explicit_choice_before_bounded_explanation(self):
        dataset = make_adapter()

        for prediction, expected in (
            ("Answer: B because the result follows.", "b"),
            ("The correct answer is C since the calculation.", "c"),
            ("Final answer: A\nExplanation: details", "a"),
            ("Answer: B, because the result follows.", "b"),
            ("The correct answer is C, since calculation.", "c"),
            ("Final answer: A, Explanation: details", "a"),
        ):
            with self.subTest(prediction=prediction):
                self.assertEqual(
                    expected,
                    parse_multiple_choice_response(prediction),
                )
                self.assertEqual(
                    1,
                    dataset.get_score(
                        prediction, expected.upper(), "multiple-choice"
                    ),
                )

        self.assertIsNone(
            parse_multiple_choice_response("A because it is correct")
        )
        self.assertIsNone(
            parse_multiple_choice_response(
                "The answer is a complex expression"
            )
        )

    def test_multiple_choice_accepts_colon_after_explicit_is_marker(self):
        dataset = make_adapter()

        for prediction, expected in (
            ("The correct answer is: A", "a"),
            ("The answer is: B", "b"),
            ("Final answer is: C", "c"),
            ("Answer is: D", "d"),
        ):
            with self.subTest(prediction=prediction):
                self.assertEqual(
                    expected,
                    parse_multiple_choice_response(prediction),
                )
                self.assertEqual(
                    1,
                    dataset.get_score(
                        prediction, expected.upper(), "multiple-choice"
                    ),
                )

    def test_multiple_choice_rejects_coordinated_explicit_alternatives(self):
        dataset = make_adapter()

        for prediction, possible_gold in (
            ("Final answer: option C and D.", "CD"),
            ("The answer is (A) or (B).", "AB"),
            ("The answer is **A** or **B**.", "AB"),
        ):
            with self.subTest(prediction=prediction):
                self.assertIsNone(parse_multiple_choice_response(prediction))
                for gold in possible_gold:
                    self.assertEqual(
                        0,
                        dataset.get_score(prediction, gold, "multiple-choice"),
                    )

    def test_multiple_choice_rejects_comma_separated_explicit_alternatives(self):
        dataset = make_adapter()

        for prediction in (
            "Answer: (A), (B)",
            r"Final answer: \boxed{A}, \boxed{B}",
        ):
            with self.subTest(prediction=prediction):
                self.assertIsNone(parse_multiple_choice_response(prediction))
                for gold in "AB":
                    self.assertEqual(
                        0,
                        dataset.get_score(
                            prediction, gold, "multiple-choice"
                        ),
                    )

        self.assertEqual(
            "a",
            parse_multiple_choice_response("Answer: (A), explanation"),
        )

    def test_multiple_choice_applies_comma_pair_event_in_source_order(self):
        pair_then_final = "Answer: (A), (B). Final answer: C."
        final_then_pair = "Final answer: C. Answer: (A), (B)"

        self.assertEqual("c", parse_multiple_choice_response(pair_then_final))
        self.assertIsNone(parse_multiple_choice_response(final_then_pair))

    def test_multiple_choice_ignores_explanatory_generic_answer_is_marker(self):
        dataset = make_adapter()
        for explanation in (
            "This answer is based on the calculation.",
            "The answer is based on the calculation.",
            "The answer is supported by the calculation.",
            "The answer is derived from the calculation.",
            "The answer is calculated from the given values.",
            "The answer is obtained from the equation.",
            "The answer is consistent with the diagram.",
        ):
            prediction = f"Final answer: B. {explanation}"
            with self.subTest(explanation=explanation):
                self.assertEqual(
                    "b",
                    parse_multiple_choice_response(prediction),
                )
                self.assertEqual(
                    1,
                    dataset.get_score(prediction, "B", "multiple-choice"),
                )

        valid_generic = "Final answer: option B. Actually, the answer is C."
        self.assertEqual("c", parse_multiple_choice_response(valid_generic))

        malformed_generic = "The answer is B. Actually, the answer is unknown."
        self.assertIsNone(parse_multiple_choice_response(malformed_generic))

        malformed_strong = "Final answer: option B. Later, answer: unknown."
        self.assertIsNone(parse_multiple_choice_response(malformed_strong))

    def test_multiple_choice_rejects_coordinated_boxed_alternatives(self):
        dataset = make_adapter()

        for coordinator in ("and", "or", "/", "&"):
            prediction = rf"\boxed{{A}} {coordinator} \boxed{{B}}"
            with self.subTest(coordinator=coordinator):
                self.assertIsNone(parse_multiple_choice_response(prediction))
                for gold in "AB":
                    self.assertEqual(
                        0,
                        dataset.get_score(prediction, gold, "multiple-choice"),
                    )

    def test_multiple_choice_applies_coordinated_boxed_event_in_source_order(self):
        dataset = make_adapter()
        pair_then_final = r"\boxed{A} or \boxed{B}. Final answer: C."
        final_then_pair = r"Final answer: C. \boxed{A} or \boxed{B}."

        self.assertEqual("c", parse_multiple_choice_response(pair_then_final))
        self.assertEqual(
            1,
            dataset.get_score(pair_then_final, "C", "multiple-choice"),
        )
        self.assertIsNone(parse_multiple_choice_response(final_then_pair))
        for gold in "ABC":
            with self.subTest(gold=gold):
                self.assertEqual(
                    0,
                    dataset.get_score(final_then_pair, gold, "multiple-choice"),
                )

    def test_multiple_choice_rejects_coordinated_leading_labels(self):
        dataset = make_adapter()

        for prediction in (
            "(A) or (B)",
            "A) and B)",
            "**A.** & **B.**",
            "(A) / (B)",
        ):
            with self.subTest(prediction=prediction):
                self.assertIsNone(parse_multiple_choice_response(prediction))
                for gold in "AB":
                    self.assertEqual(
                        0,
                        dataset.get_score(prediction, gold, "multiple-choice"),
                    )

    def test_multiple_choice_rejects_ambiguous_and_non_answer_inputs(self):
        dataset = make_adapter()

        for prediction in (
            "A because it is correct",
            "A result was observed",
            "",
            "No conclusion",
            None,
        ):
            with self.subTest(prediction=prediction):
                try:
                    score = dataset.get_score(prediction, "A", "multiple-choice")
                except Exception as exc:
                    self.fail(f"get_score raised {type(exc).__name__}: {exc}")
                self.assertEqual(0, score)

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
