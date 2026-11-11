"""Run with: python -m unittest -v test_selection.py"""
import copy
import unittest
import select_questions as s


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = s.chart_rows([
            {"imgname": "2.png", "query": "two first", "label": "yes"},
            {"imgname": "10.png", "query": "ten first", "label": "3"},
            {"imgname": "2.png", "query": "two second", "label": "4"},
            {"imgname": "10.png", "query": "ten second", "label": "5"},
            {"imgname": "3.png", "query": "three", "label": "6"},
            {"imgname": "4.png", "query": "four", "label": "7"},
            {"imgname": "5.png", "query": "five", "label": "8"},
        ])

    def test_lexical_groups_one_question_and_disjoint_owners(self):
        a, sa, _ = s.select("A", self.rows, "fingerprint", 2, [])
        b, sb, _ = s.select("B", self.rows, "fingerprint", 2, [])
        self.assertEqual([r["image_group_no"] for r in sa], [1, 3])
        self.assertEqual(sa[0]["question"], "ten first")
        self.assertEqual(sb[0]["question"], "two second")
        self.assertFalse({r["image_path"] for r in sa} & {r["image_path"] for r in sb})
        self.assertEqual(s.select("A", self.rows, "fingerprint", 2, [])[0], a)

    def test_question_replacement_and_preserved_notes(self):
        log, _, _ = s.select("A", self.rows, "fingerprint", 2, [])
        log[0].update({"상태": "제외", "제외범위": "문항", "메모": "모호한 질문"})
        log[1].update({"상태": "작성완료", "사용모델": "model-x", "제작방법": "모델+사람검수", "메모": "keep me"})
        updated, selected, _ = s.select("A", self.rows, "fingerprint", 2, log)
        self.assertEqual([r["question"] for r in selected], ["ten second", "three"])
        self.assertEqual(updated[-1]["메모"], "keep me")
        self.assertEqual(len(updated), 3)

    def test_image_exclusion_keeps_numbering(self):
        log, _, _ = s.select("A", self.rows, "fingerprint", 2, [])
        log[0].update({"상태": "제외", "제외범위": "이미지", "메모": "판독 불가"})
        _, selected, _ = s.select("A", self.rows, "fingerprint", 2, log)
        self.assertEqual([r["image_group_no"] for r in selected], [3, 5])

    def test_changed_source_and_duplicate_history_rejected(self):
        log, _, _ = s.select("A", self.rows, "fingerprint", 2, [])
        with self.assertRaises(ValueError):
            s.select("A", self.rows, "other-version", 2, log)
        with self.assertRaises(ValueError):
            s.select("A", self.rows, "fingerprint", 2, log + [copy.deepcopy(log[0])])
        log[0]["이미지그룹번호"] = "2"
        with self.assertRaises(ValueError):
            s.select("A", self.rows, "fingerprint", 2, log)

    def test_only_train_with_image_and_full_id_path(self):
        base = {"image": "image.png", "question": "q", "choices": ["a", "b"], "answer": 0}
        problems = {"2": base, "10": base, "20": dict(base, image=None), "30": base}
        rows = s.science_rows(problems, {"train": ["2", "10", "20"], "test": ["30"]})
        _, c, _ = s.select("C", rows, "f", 1, [])
        _, d, _ = s.select("D", rows, "f", 1, [])
        self.assertEqual(c[0]["image_path"], "train/10/image.png")
        self.assertEqual(d[0]["image_path"], "train/2/image.png")

    def test_insufficient_candidates_fail(self):
        with self.assertRaises(ValueError):
            s.select("B", self.rows, "fingerprint", 3, [])


if __name__ == "__main__":
    unittest.main()
