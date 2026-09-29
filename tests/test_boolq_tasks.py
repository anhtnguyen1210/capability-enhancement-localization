from __future__ import annotations

import unittest

from boolq_tasks import canonical_hash, process_source, select_source_rows


class BoolQTasksTest(unittest.TestCase):
    def test_process_source_matches_local_lm_eval_alias(self):
        source = {
            "question": "is this a question",
            "passage": "A short passage.",
            "idx": 4,
            "label": 1,
        }
        processed = process_source(source)
        self.assertEqual(
            processed["query"],
            "A short passage.\nQuestion: is this a question?\nAnswer:",
        )
        self.assertEqual(processed["choices"], ["no", "yes"])
        self.assertEqual(processed["gold"], 1)

    def test_source_selection_is_stable_and_passage_grouped(self):
        rows = [
            {
                "question": f"question {index}",
                "passage": f"passage {index // 2}",
                "idx": index,
                "label": index % 2,
            }
            for index in range(12)
        ]
        first = select_source_rows(rows, dataset_split="train", seed=7, count=5)
        second = select_source_rows(rows, dataset_split="train", seed=7, count=5)
        self.assertEqual(first, second)
        self.assertEqual(len({row["passage_sha256"] for row in first}), 5)

    def test_source_selection_excludes_passages(self):
        rows = [
            {
                "question": f"question {index}",
                "passage": f"passage {index}",
                "idx": index,
                "label": index % 2,
            }
            for index in range(6)
        ]
        excluded = {canonical_hash("passage 3")}
        selected = select_source_rows(
            rows,
            dataset_split="validation",
            seed=11,
            count=5,
            excluded_passage_sha256=excluded,
        )
        self.assertFalse(excluded & {row["passage_sha256"] for row in selected})


if __name__ == "__main__":
    unittest.main()
