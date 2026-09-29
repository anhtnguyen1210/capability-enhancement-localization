from __future__ import annotations

import unittest

from hellaswag_tasks import preprocess, process_source, select_source_rows


class HellaSwagTasksTest(unittest.TestCase):
    def test_preprocess_matches_task_contract(self):
        self.assertEqual(preprocess(" x [title] [remove]  y "), "x.  y")

    def test_process_source_matches_lm_eval_prompt(self):
        source = {
            "activity_label": "Cooking",
            "ctx_a": "A person stirs.",
            "ctx_b": "then",
            "endings": ["one", "two", "three", "four"],
            "label": "2",
        }
        processed = process_source(source)
        self.assertEqual(processed["query"], "Cooking: A person stirs. Then")
        self.assertEqual(processed["gold"], 2)

    def test_source_selection_is_stable_and_unique(self):
        rows = [
            {
                "ind": index,
                "activity_label": "task",
                "ctx_a": f"context {index}",
                "ctx_b": "then",
                "ctx": f"context {index} then",
                "endings": ["a", "b", "c", "d"],
                "source_id": f"source-{index}",
                "split": "train",
                "split_type": "indomain",
                "label": str(index % 4),
            }
            for index in range(12)
        ]
        first = select_source_rows(rows, dataset_split="train", seed=7, count=5)
        second = select_source_rows(rows, dataset_split="train", seed=7, count=5)
        self.assertEqual(first, second)
        self.assertEqual(len({row["dataset_row_index"] for row in first}), 5)


if __name__ == "__main__":
    unittest.main()
