from __future__ import annotations

import unittest

from hellaswag_evaluation import summarize_condition


class HellaSwagEvaluationTest(unittest.TestCase):
    def test_summary_uses_per_example_correctness(self):
        records = [
            {"correct": True, "tie": False, "margin": 0.4},
            {"correct": False, "tie": True, "margin": -0.2},
        ]
        summary = summarize_condition(records)
        self.assertEqual(summary["example_count"], 2)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["correct_count"], 1)
        self.assertEqual(summary["tie_count"], 1)


if __name__ == "__main__":
    unittest.main()
