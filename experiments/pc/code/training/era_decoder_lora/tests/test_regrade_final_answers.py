"""Regression checks for isolated open-answer repair; no model/GPU required."""
import importlib.util
from pathlib import Path
import unittest

MODULE=Path(__file__).resolve().parents[1]/'tools/regrade_final_answers.py'
SPEC=importlib.util.spec_from_file_location('repair',MODULE)
repair=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(repair)

def row(answer,gold='2.83',kind='open',finish='stop'):
    return {'question_type':kind,'options':{'A':'red','B':'green','C':'blue'} if kind=='multiple-choice' else [],
            'reference_answer':gold,'model_response':{'raw':answer,'finish_reason':finish}}

def proposal(r):
    return repair.proposed_grade(r,repair.frozen.grade_response(r))

class RepairTests(unittest.TestCase):
    def test_ampere_unit_cannot_select_synthetic_A(self):
        for answer in ('4 A','2 A','5/3 A'):
            r=row('Final answer: '+answer)
            self.assertEqual(repair.frozen.grade_response(r)['hit'],1)
            g,payload=proposal(r)
            self.assertEqual(g['status'],'pending_judge')
            self.assertEqual(payload,answer)

    def test_reference_substring_not_exact_match(self):
        r=row('Final answer: 30 V',gold='3')
        self.assertEqual(repair.frozen.grade_response(r)['hit'],1)
        self.assertEqual(proposal(r)[0]['status'],'pending_judge')

    def test_literal_open_letter_reference_is_not_internal_choice(self):
        r=row('Final answer: C',gold='C')
        self.assertEqual(repair.frozen.grade_response(r)['hit'],0)
        self.assertEqual(proposal(r)[0]['hit'],1)
        self.assertEqual(repair.grade_new_judge(r,'A')['hit'],1)
        self.assertEqual(repair.grade_new_judge(r,'B')['hit'],0)

    def test_exact_alias_and_wrappers(self):
        r=row('Final answer: **$MgS$**',gold="['$MgS$', 'MgS']")
        self.assertEqual(proposal(r)[0]['hit'],1)

    def test_fraction_rounding_left_to_same_judge(self):
        r=row('Final answer: $\\frac{19}{3}\\mu F$',gold='6.333')
        g,payload=proposal(r)
        self.assertIsNone(g['hit'])
        self.assertEqual(payload,'$\\frac{19}{3}\\mu F$')

    def test_no_new_numeric_tolerance(self):
        for answer in ('12.6 g','13.0 g','12.9701'):
            self.assertEqual(proposal(row('Final answer: '+answer,gold='12.97'))[0]['status'],'pending_judge')

    def test_mc_correct_letter_not_penalized_for_bad_explanation(self):
        r=row('Evidence: 2+2=9.\nRule: Wrong reasoning.\nFinal answer: B',gold='B',kind='multiple-choice')
        g,payload=proposal(r)
        self.assertEqual(g['hit'],1);self.assertIsNone(payload)
        self.assertEqual(g['repair_route'],'unchanged_multiple_choice')

    def test_length_stays_wrong_even_with_correct_terminal_answer(self):
        g,payload=proposal(row('Final answer: 2.83',finish='length'))
        self.assertEqual(g['hit'],0);self.assertIsNone(payload)

    def test_conflicting_final_fields_keep_ambiguity(self):
        text='Final answer: 4\nFinal answer: 2.83'
        g,payload=proposal(row(text))
        self.assertEqual(g['status'],'pending_judge');self.assertEqual(payload,text)

    def test_terminal_inline_label_does_not_send_body_to_judge(self):
        r=row('Earlier body says MgS.\nOn the last line: Final answer: MgO',gold='MgS')
        g,payload=proposal(r)
        self.assertEqual(g['status'],'pending_judge');self.assertEqual(payload,'MgO')

    def test_no_label_fallback_keeps_full_answer(self):
        text='The answer could be 3 or 4.'
        self.assertEqual(proposal(row(text))[1],text)

    def test_invalid_or_Z_does_not_gain_credit(self):
        r=row('Final answer: 4 A')
        self.assertEqual(repair.grade_new_judge(r,'Z')['hit'],0)
        self.assertEqual(repair.grade_new_judge(r,'C')['hit'],0)

if __name__=='__main__':unittest.main()
