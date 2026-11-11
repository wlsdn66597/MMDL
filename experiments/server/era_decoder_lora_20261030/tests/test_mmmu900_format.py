"""Boundary cases: format adoption is not semantic correctness."""
import importlib.util
from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from mmmu900_epoch_campaign import format_metrics


class FormatTests(unittest.TestCase):
    def assess(self, text, finish='stop'):
        return format_metrics({'model_response': {'raw': text, 'finish_reason': finish}})

    def test_valid(self):
        self.assertTrue(self.assess('Evidence: x\nRule: y\nApplication: z\nFinal answer: A')['complete_era_format'])

    def test_markdown_headers(self):
        self.assertTrue(self.assess('**Evidence:** x\n**Rule:** y\n**Application:** z\n**Final answer:** A')['complete_era_format'])

    def test_empty_section(self):
        self.assertFalse(self.assess('Evidence:\nRule: y\nApplication: z\nFinal answer: A')['complete_era_format'])

    def test_reordered(self):
        self.assertFalse(self.assess('Rule: y\nEvidence: x\nApplication: z\nFinal answer: A')['complete_era_format'])

    def test_duplicate(self):
        self.assertFalse(self.assess('Evidence: x\nRule: y\nApplication: z\nRule: again\nFinal answer: A')['complete_era_format'])

    def test_truncation(self):
        self.assertFalse(self.assess('Evidence: x\nRule: y\nApplication: z\nFinal answer: A', 'length')['complete_era_format'])

    def test_nonterminal_answer(self):
        self.assertFalse(self.assess('Evidence: x\nRule: y\nApplication: z\nFinal answer: A\nActually B')['complete_era_format'])

    def test_wrong_content_can_pass_format(self):
        self.assertTrue(self.assess('Evidence: 2\nRule: 2+2=9\nApplication: therefore 9\nFinal answer: 9')['complete_era_format'])


if __name__ == '__main__':
    unittest.main()
