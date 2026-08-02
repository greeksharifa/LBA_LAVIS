import unittest


class HierarchicalPromptTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "qid": "q0",
            "main_q": "Why is the bridge closed?",
            "question_type": "multiple-choice",
            "data_type": "image",
            "candidate_list": ["CHOICE_SENTINEL"],
            "gt_ans": "GT_SENTINEL",
            "vision": [object()],
        }

    def test_root_decomposition_uses_main_question_modality_and_exact_count(self):
        from subqa.prompts import build_decomposition_prompt

        prompt = build_decomposition_prompt(self.sample, child_count=4)

        self.assertIn("Why is the bridge closed?", prompt)
        self.assertIn("Modality: image", prompt)
        self.assertIn("exactly 4", prompt)
        self.assertNotIn("Parent question:", prompt)
        self._assert_private_sample_fields_absent(prompt)

    def test_descendant_decomposition_includes_ordered_ancestor_path_and_parent(self):
        from subqa.prompts import build_decomposition_prompt

        prompt = build_decomposition_prompt(
            self.sample,
            child_count=3,
            parent_question="What physical damage is visible?",
            ancestor_questions=(
                "What can be observed?",
                "Which bridge is shown?",
            ),
        )

        self.assertIn("exactly 3", prompt)
        self.assertIn("1. What can be observed?", prompt)
        self.assertIn("2. Which bridge is shown?", prompt)
        self.assertIn(
            "Parent question: What physical damage is visible?", prompt
        )
        self.assertLess(
            prompt.index("What can be observed?"),
            prompt.index("Which bridge is shown?"),
        )
        self.assertLess(
            prompt.index("Which bridge is shown?"),
            prompt.index("What physical damage is visible?"),
        )
        self._assert_private_sample_fields_absent(prompt)

    def test_question_repair_lists_existing_questions_and_only_missing_count(self):
        from subqa.prompts import build_question_repair_prompt

        prompt = build_question_repair_prompt(
            self.sample,
            child_count=4,
            existing_questions=("First valid question?", "Second valid question?"),
            parent_question="What caused the closure?",
            ancestor_questions=("What can be observed?",),
        )

        self.assertIn("exactly 2 additional", prompt)
        self.assertIn("1. First valid question?", prompt)
        self.assertIn("2. Second valid question?", prompt)
        self.assertIn("Parent question: What caused the closure?", prompt)
        self.assertIn("Do not repeat", prompt)
        self._assert_private_sample_fields_absent(prompt)

    def test_question_repair_requires_a_new_reasoning_focus(self):
        from subqa.prompts import build_question_repair_prompt

        prompt = build_question_repair_prompt(
            self.sample,
            child_count=4,
            existing_questions=(
                "What is the bridge closure called?",
                "What term refers to the bridge closure?",
                "Which name describes the bridge closure?",
            ),
        )

        self.assertIn("different evidence dimension", prompt)
        self.assertIn("asks for the same answer is invalid", prompt)
        self.assertIn("prerequisite observation or concept", prompt)

    def test_direct_answer_uses_ancestor_path_target_and_one_sentence_limit(self):
        from subqa.prompts import build_direct_answer_prompt

        prompt = build_direct_answer_prompt(
            self.sample,
            target_question="What damage is visible?",
            ancestor_questions=("What can be observed?",),
        )

        self.assertIn("Why is the bridge closed?", prompt)
        self.assertIn("1. What can be observed?", prompt)
        self.assertIn("Target question: What damage is visible?", prompt)
        self.assertIn("Modality: image", prompt)
        self.assertIn("maximum of one sentence", prompt)
        self._assert_private_sample_fields_absent(prompt)

    def test_refined_answer_includes_selected_child_qa_and_direct_draft(self):
        from subqa.prompts import build_refined_answer_prompt

        prompt = build_refined_answer_prompt(
            self.sample,
            target_question="What caused the closure?",
            child_qa=(
                ("What damage is visible?", "A support beam is cracked."),
                ("Is traffic present?", "No traffic is visible."),
            ),
            direct_draft="The bridge is closed because it is damaged.",
            condition_on_direct_suba=True,
        )

        self.assertIn("Direct draft:", prompt)
        self.assertIn("The bridge is closed because it is damaged.", prompt)
        self.assertIn("Child question 1: What damage is visible?", prompt)
        self.assertIn("Child answer 1: A support beam is cracked.", prompt)
        self.assertIn("Child question 2: Is traffic present?", prompt)
        self.assertIn("Child answer 2: No traffic is visible.", prompt)
        self._assert_private_sample_fields_absent(prompt)

    def test_refined_answer_omits_direct_draft_when_conditioning_is_disabled(self):
        from subqa.prompts import build_refined_answer_prompt

        prompt = build_refined_answer_prompt(
            self.sample,
            target_question="What caused the closure?",
            child_qa=(("What damage is visible?", "A support beam is cracked."),),
            direct_draft="DIRECT_DRAFT_SENTINEL",
            condition_on_direct_suba=False,
        )

        self.assertNotIn("Direct draft:", prompt)
        self.assertNotIn("DIRECT_DRAFT_SENTINEL", prompt)
        self.assertIn("Child question 1: What damage is visible?", prompt)
        self.assertIn("Child answer 1: A support beam is cracked.", prompt)
        self._assert_private_sample_fields_absent(prompt)

    def test_unknown_modality_is_rejected_by_every_prompt_builder(self):
        from subqa.prompts import (
            build_decomposition_prompt,
            build_direct_answer_prompt,
            build_question_repair_prompt,
            build_refined_answer_prompt,
        )

        sample = dict(self.sample, data_type="audio")
        calls = (
            lambda: build_decomposition_prompt(sample, child_count=2),
            lambda: build_question_repair_prompt(
                sample,
                child_count=2,
                existing_questions=("Existing?",),
            ),
            lambda: build_direct_answer_prompt(sample, target_question="Target?"),
            lambda: build_refined_answer_prompt(
                sample,
                target_question="Target?",
                child_qa=(("Child?", "Answer."),),
                direct_draft="Draft.",
                condition_on_direct_suba=True,
            ),
        )

        for call in calls:
            with self.subTest(call=call):
                with self.assertRaisesRegex(ValueError, "unsupported modality.*audio"):
                    call()

    def _assert_private_sample_fields_absent(self, prompt):
        self.assertNotIn("CHOICE_SENTINEL", prompt)
        self.assertNotIn("GT_SENTINEL", prompt)


class HierarchicalParsingTests(unittest.TestCase):
    def test_questions_remove_numbering_bullets_and_normalize_whitespace(self):
        from subqa.parsing import parse_questions

        questions = parse_questions(
            "  1.   First   question?  \n"
            "- Second\tquestion ?\n"
            "*   Third question.  \n"
            "\u2022 Fourth question?\n",
            expected_count=4,
        )

        self.assertEqual(
            [
                "First question?",
                "Second question ?",
                "Third question.",
                "Fourth question?",
            ],
            questions,
        )

    def test_questions_dedupe_casefold_and_terminal_punctuation_in_first_order(self):
        from subqa.parsing import parse_questions

        questions = parse_questions(
            "1. What Color Is It?\n"
            "2. what color is it!!!\n"
            "3. Where is it located?\n"
            "4. WHERE IS IT LOCATED .\n"
            "5. What material is it made from?\n",
            expected_count=4,
        )

        self.assertEqual(
            [
                "What Color Is It?",
                "Where is it located?",
                "What material is it made from?",
            ],
            questions,
        )

    def test_questions_stop_at_expected_count_without_generic_padding(self):
        from subqa.parsing import parse_questions

        enough = parse_questions("A?\nB?\nC?", expected_count=2)
        partial = parse_questions("Only one?", expected_count=3)

        self.assertEqual(["A?", "B?"], enough)
        self.assertEqual(["Only one?"], partial)

    def test_blank_answer_is_invalid_and_nonblank_answer_is_normalized(self):
        from subqa.parsing import parse_answer

        self.assertIsNone(parse_answer(" \n\t "))
        self.assertEqual("A concise answer.", parse_answer(" A   concise\nanswer. "))


if __name__ == "__main__":
    unittest.main()
