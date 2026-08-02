import unittest


class CanonicalTreeSchemaTests(unittest.TestCase):
    @staticmethod
    def _complete_tree(branching=(4, 3)):
        from subqa.schema import build_complete_subq_tree

        return build_complete_subq_tree(
            branching,
            question_factory=lambda node_id: f"question {node_id}",
        )

    def test_depth_two_tree_has_stable_ids_and_preorder_serialization(self):
        from subqa.schema import preorder_nodes, validate_subq_tree

        tree = self._complete_tree()
        expected_ids = [
            "0",
            "0.0",
            "0.1",
            "0.2",
            "1",
            "1.0",
            "1.1",
            "1.2",
            "2",
            "2.0",
            "2.1",
            "2.2",
            "3",
            "3.0",
            "3.1",
            "3.2",
        ]

        self.assertEqual(2, tree["schema_version"])
        self.assertEqual("root", tree["root_id"])
        self.assertEqual(2, tree["max_depth"])
        self.assertEqual(expected_ids, [node["id"] for node in tree["nodes"]])
        self.assertEqual(expected_ids, [node["id"] for node in preorder_nodes(tree)])
        self.assertEqual(["0.0", "0.1", "0.2"], tree["nodes"][0]["child_ids"])
        validate_subq_tree(tree, branching=(4, 3))

    def test_depth_three_tree_uses_the_same_recursive_parent_contract(self):
        from subqa.schema import validate_subq_tree

        tree = self._complete_tree((2, 2, 2))
        nodes = {node["id"]: node for node in tree["nodes"]}

        self.assertEqual(14, len(nodes))
        self.assertEqual(
            [
                "0",
                "0.0",
                "0.0.0",
                "0.0.1",
                "0.1",
                "0.1.0",
                "0.1.1",
                "1",
                "1.0",
                "1.0.0",
                "1.0.1",
                "1.1",
                "1.1.0",
                "1.1.1",
            ],
            [node["id"] for node in tree["nodes"]],
        )
        self.assertEqual("0.1", nodes["0.1.1"]["parent_id"])
        self.assertEqual(3, nodes["0.1.1"]["depth"])
        self.assertEqual(1, nodes["0.1.1"]["sibling_index"])
        self.assertEqual(["0.1.0", "0.1.1"], nodes["0.1"]["child_ids"])
        validate_subq_tree(tree, branching=(2, 2, 2))

    def test_malformed_structural_invariants_are_rejected(self):
        from subqa.schema import validate_subq_tree

        cases = {}

        duplicate = self._complete_tree()
        duplicate["nodes"][-1]["id"] = duplicate["nodes"][0]["id"]
        cases["duplicate id"] = duplicate

        missing_parent = self._complete_tree()
        missing_parent["nodes"][1]["parent_id"] = "missing"
        cases["parent"] = missing_parent

        wrong_depth = self._complete_tree()
        wrong_depth["nodes"][1]["depth"] = 1
        cases["depth"] = wrong_depth

        wrong_sibling = self._complete_tree()
        wrong_sibling["nodes"][1]["sibling_index"] = 1
        cases["sibling"] = wrong_sibling

        wrong_child_order = self._complete_tree()
        wrong_child_order["nodes"][0]["child_ids"] = ["0.1", "0.0", "0.2"]
        cases["child"] = wrong_child_order

        extra_depth = self._complete_tree()
        extra_depth["nodes"][1]["depth"] = 3
        cases["max_depth"] = extra_depth

        for expected_message, tree in cases.items():
            with self.subTest(invariant=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    validate_subq_tree(tree, branching=(4, 3))

    def test_malformed_status_and_failure_reason_pairing_is_rejected(self):
        from subqa.schema import validate_subq_tree

        cases = []

        expanded_with_failure = self._complete_tree()
        expanded_with_failure["nodes"][0]["failure_reason"] = "unexpected"
        cases.append(("failure_reason", expanded_with_failure))

        partial_without_failure = self._complete_tree()
        partial_without_failure["nodes"][0]["child_ids"].pop()
        partial_without_failure["nodes"].pop(3)
        partial_without_failure["nodes"][0]["expansion_status"] = "partial"
        cases.append(("failure_reason", partial_without_failure))

        failed_with_children = self._complete_tree()
        failed_with_children["nodes"][0]["expansion_status"] = "failed"
        failed_with_children["nodes"][0]["failure_reason"] = "generation failed"
        cases.append(("failed", failed_with_children))

        internal_leaf = self._complete_tree()
        internal_leaf["nodes"][0]["expansion_status"] = "leaf"
        cases.append(("leaf", internal_leaf))

        unsupported = self._complete_tree()
        unsupported["nodes"][0]["expansion_status"] = "done"
        cases.append(("status", unsupported))

        for expected_message, tree in cases:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    validate_subq_tree(tree, branching=(4, 3))

    def test_expansion_confidence_rejects_missing_or_invalid_metrics(self):
        from subqa.schema import validate_confidence_mapping, validate_subq_tree

        missing_metric = self._complete_tree()
        missing_metric["nodes"][0]["expansion_confidence"] = {
            "token_min_prob": 0.5,
        }
        with self.assertRaisesRegex(ValueError, "seq_ppl"):
            validate_confidence_mapping(
                missing_metric["nodes"][0]["expansion_confidence"],
                "expansion_confidence",
                ("seq_ppl", "token_min_prob"),
            )

        cases = []
        non_finite = self._complete_tree()
        non_finite["nodes"][0]["expansion_confidence"] = {
            "seq_ppl": 2.0,
            "token_min_prob": float("inf"),
        }
        cases.append(("token_min_prob", non_finite))

        out_of_range = self._complete_tree()
        out_of_range["nodes"][0]["expansion_confidence"] = {
            "seq_ppl": 2.0,
            "token_min_prob": 1.1,
        }
        cases.append(("token_min_prob", out_of_range))

        for expected_message, tree in cases:
            with self.subTest(expected_message=expected_message):
                with self.assertRaisesRegex(ValueError, expected_message):
                    validate_subq_tree(tree, branching=(4, 3))

    def test_preorder_is_validated_instead_of_silently_resorted(self):
        from subqa.schema import validate_subq_tree

        tree = self._complete_tree()
        tree["nodes"][1], tree["nodes"][2] = tree["nodes"][2], tree["nodes"][1]

        with self.assertRaisesRegex(ValueError, "preorder"):
            validate_subq_tree(tree, branching=(4, 3))

    def test_depth_one_question_projection_uses_stable_id_order(self):
        from subqa.schema import project_depth_one_questions

        tree = self._complete_tree()

        self.assertEqual(
            ["question 0", "question 1", "question 2", "question 3"],
            project_depth_one_questions(tree, expected_count=4),
        )

    def test_depth_one_question_projection_rejects_incomplete_root(self):
        from subqa.schema import project_depth_one_questions

        tree = self._complete_tree()
        tree["nodes"] = [
            node
            for node in tree["nodes"]
            if node["id"] != "3" and not node["id"].startswith("3.")
        ]

        with self.assertRaisesRegex(ValueError, "depth-1"):
            project_depth_one_questions(tree, expected_count=4)


class DeterministicSelectionTests(unittest.TestCase):
    def test_evidence_subsets_are_lexicographic_and_limited_by_prefix(self):
        from subqa.selection import evidence_subsets

        self.assertEqual(
            ((0, 1), (0, 2), (1, 2)),
            evidence_subsets([0, 1, 2], m=2, k=3),
        )
        self.assertEqual(
            ((0, 1), (0, 2)),
            evidence_subsets([0, 1, 2], m=2, k=2),
        )

    def test_evidence_subsets_preserve_available_sibling_order(self):
        from subqa.selection import evidence_subsets

        available_children = ["0.0", "0.2", "0.3"]

        self.assertEqual(
            (("0.0", "0.2"), ("0.0", "0.3"), ("0.2", "0.3")),
            evidence_subsets(available_children, m=2, k=3),
        )
        self.assertEqual((), evidence_subsets(["0.0"], m=2, k=3))

    def test_best_candidate_excludes_invalid_and_uses_first_normalized_maximum(self):
        from subqa.selection import select_best_candidate

        candidates = [
            {
                "answer": "invalid high",
                "status": "invalid",
                "confidence": {"token_min_prob": 1.0},
                "failure_reason": "blank answer",
            },
            {
                "answer": "first valid",
                "status": "valid",
                "confidence": {"token_min_prob": 0.8},
                "failure_reason": None,
            },
            {
                "answer": "second valid",
                "status": "valid",
                "confidence": {"token_min_prob": 0.8},
                "failure_reason": None,
            },
        ]

        selected, normalized, original_index = select_best_candidate(
            candidates, "token_min_prob"
        )

        self.assertIs(candidates[1], selected)
        self.assertEqual(0.8, normalized)
        self.assertEqual(1, original_index)

    def test_best_candidate_normalizes_seq_ppl_and_can_have_no_valid_input(self):
        from subqa.selection import select_best_candidate

        candidates = [
            {
                "answer": "ppl two",
                "status": "valid",
                "confidence": {"seq_ppl": 2.0},
                "failure_reason": None,
            },
            {
                "answer": "ppl four",
                "status": "valid",
                "confidence": {"seq_ppl": 4.0},
                "failure_reason": None,
            },
        ]

        selected, normalized, original_index = select_best_candidate(
            candidates, "seq_ppl"
        )

        self.assertIs(candidates[0], selected)
        self.assertEqual(0.5, normalized)
        self.assertEqual(0, original_index)
        self.assertIsNone(
            select_best_candidate(
                [
                    {
                        "answer": "bad",
                        "status": "invalid",
                        "confidence": {},
                        "failure_reason": "failed",
                    }
                ],
                "seq_ppl",
            )
        )


if __name__ == "__main__":
    unittest.main()
