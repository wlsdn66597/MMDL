"""CPU tests for exact-token comparison and pause behavior (no model imports)."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import pilot


class PilotLogicTests(unittest.TestCase):
    def setUp(self):
        self.test_root = Path(__file__).resolve().parent / 'test_tmp'
        self.test_root.mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=self.test_root)
        self.previous = pilot.ROOT
        pilot.ROOT = Path(self.tmp.name)
        self.bundle = {'items':[{'id':'validation_Math_13'}] + [{'id':f'item_{i}'} for i in range(1,12)],
                       'judges':[{'id':f'judge_{i}'} for i in range(4)]}
        pilot.write(pilot.ROOT/'inputs/frozen.json', self.bundle)
        for name,_,cap,order in pilot.STAGES:
            for item in pilot.stage_items(self.bundle,name,order):
                p = pilot.ROOT/'outputs'/f'{name}.jsonl'
                p.parent.mkdir(exist_ok=True)
                pilot.append(p, dict(id=item['id'], token_ids=[17,18], text='same text',
                                     finish_reason='length', stop_reason=None, input_tokens=123))

    def tearDown(self):
        pilot.ROOT = self.previous
        assert Path(self.tmp.name).resolve().is_relative_to(self.test_root.resolve())
        self.tmp.cleanup()

    def summary(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return pilot.summarize()

    def test_all_stages_and_comparisons(self):
        result = self.summary()
        self.assertTrue(result['passed'])
        self.assertEqual(sum(s['expected'] for s in result['stage_counts'].values()),84)
        self.assertEqual(len(result['comparisons']),65)

    def test_equal_text_different_token_ids_is_failure(self):
        p = pilot.ROOT/'outputs/vl_pair.jsonl'
        rows = pilot.records(p)
        rows[0]['token_ids'] = [99,18]
        p.unlink()
        for row in rows:
            pilot.append(p,row)
        result = self.summary()
        self.assertTrue(result['complete'])
        self.assertFalse(result['passed'])
        self.assertEqual([r['differences'] for r in result['comparisons'] if not r['passed']], [['token_ids']])

    def test_missing_stage_never_passes(self):
        (pilot.ROOT/'outputs/judge_repeat.jsonl').unlink()
        result = self.summary()
        self.assertFalse(result['complete'])
        self.assertFalse(result['passed'])

    def test_pause_exits_before_model_import(self):
        (pilot.ROOT/'PAUSE').touch()
        with self.assertRaises(SystemExit) as cm:
            pilot.run_stage('vl_reference')
        self.assertEqual(cm.exception.code,75)


if __name__ == '__main__':
    unittest.main()
