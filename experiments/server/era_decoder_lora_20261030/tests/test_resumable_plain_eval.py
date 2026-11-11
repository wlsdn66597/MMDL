"""Crash/restart and preservation tests; no model packages or GPU required."""
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('resume', TOOLS / 'resume_plain_mmmu900.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.p = self.root / 'predictions.jsonl'
        self.archive = self.root / 'archive'; self.archive.mkdir()

    def test_truncated_tail_preserves_all_complete_records_and_original(self):
        original = b'{"id":"a"}\n{"id":"b"}\n{"id":'
        self.p.write_bytes(original)
        rows, issue = r.journal(self.p)
        self.assertEqual(len(rows), 2); self.assertEqual(issue['kind'], 'incomplete_tail')
        r.repair_journal(self.p, self.archive)
        self.assertEqual(self.p.read_bytes(), b'{"id":"a"}\n{"id":"b"}\n')
        self.assertEqual((self.archive / 'predictions.jsonl.before-repair').read_bytes(), original)

    def test_valid_last_record_without_newline_is_not_dropped(self):
        self.p.write_bytes(b'{"id":"a"}')
        r.repair_journal(self.p, self.archive)
        self.assertEqual(self.p.read_bytes(), b'{"id":"a"}\n')

    def test_malformed_middle_or_terminated_line_never_repaired(self):
        for data in (b'{bad}\n{"id":"b"}\n', b'{bad}\n'):
            self.p.write_bytes(data)
            with self.assertRaises(ValueError): r.repair_journal(self.p, self.archive)
            self.assertEqual(self.p.read_bytes(), data)

    def test_duplicate_foreign_and_reordered_ids_rejected(self):
        for ids in (['a', 'a'], ['a', 'c'], ['b', 'a']):
            with self.assertRaises(ValueError):
                r.require_prefix([{'id': x} for x in ids], ['a', 'b'])

    def test_saved_response_mutation_rejected_even_when_count_unchanged(self):
        self.p.write_bytes(b'original\n')
        r.check_append_only(self.root, {'predictions.jsonl': b'original\n'})
        self.p.write_bytes(b'modified\n')
        with self.assertRaises(ValueError):
            r.check_append_only(self.root, {'predictions.jsonl': b'original\n'})


@unittest.skipUnless(os.name == 'posix', 'Controller uses server Linux flock/process groups')
class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.out = self.root / 'out'; self.out.mkdir()
        self.cfg = {'evaluation': {'python': sys.executable, 'model_cache': 'model-cache', 'dataset_cache': 'data-cache'}}
        self.helpers = {'tools/plain_mmmu900_campaign_20261102.py': 'frozen'}
        self.plan = {'helper_hashes': self.helpers, 'python_probe': {}, 'source_hashes': {}}
        r.atomic(self.out / 'plan.json', self.plan)
        self.completed = set(r.RUN_NAMES[:3]); self.eval_calls = []; self.fail_eval_once = False
        self.fail_repeat = False; self.fail_cuda = False
        for name in r.RUN_NAMES[:4]:
            d = self.out / name; d.mkdir()
            (d / 'predictions.jsonl').write_text('{"id":"a"}\n')
            (d / 'launcher.json').write_text('{}')
        self.initial = {str(p): p.read_bytes() for name in self.completed for p in (self.out / name).iterdir()}
        record = {'id': 'a', 'model_response': {'input_tokens': 1}}
        run = {'records': [record], 'artifacts_sha256': {}}
        self.c = SimpleNamespace(CONFIG=self.root / 'config.json', helper_hashes=lambda: self.helpers,
            load_run=lambda *a: run, validate_pair=lambda *a: None, references=lambda *a: None,
            summarize=lambda *a: None,
            compare_repeats=lambda first, second: {'passed': not(self.fail_repeat and 'epoch_3' in str(first))})
        def proof(cfg, e): return {'model': str(self.root / f'model{e}'), 'hashes': {}}
        self.b = (self.c, self.cfg, self.root / 'source', self.root / 'protocol', {3407:{}},
                  ['a'], {'baseline': run}, self.plan, proof, lambda cfg,e: self.root, None)
        patches = [patch.object(r, 'ROOT', self.root), patch.object(r, 'OUT', self.out),
                   patch.object(r, 'backend', return_value=self.b), patch.object(r, 'existing_workers', return_value=[]),
                   patch.object(r, 'inspect_run', side_effect=self.inspect), patch.object(r, 'child', side_effect=self.child),
                   patch.object(sys, 'argv', ['resume', '--resume']), patch.object(r.signal, 'signal'),
                   patch.object(sys, 'stdout', new_callable=io.StringIO)]
        for p in patches: p.start(); self.addCleanup(p.stop)

    def inspect(self, dest, b, proof):
        return {'state': 'complete' if dest.name in self.completed else 'partial',
                'saved': len(r.journal(dest / 'predictions.jsonl')[0])}

    def child(self, command, log, lock_fd, **kw):
        log.write_text('mock child log\n')
        if '-c' in command:
            if self.fail_cuda: raise RuntimeError('CUDA unavailable')
            return
        name = command[command.index('--run-name') + 1]
        self.eval_calls.append(name)
        p = self.out / name / 'predictions.jsonl'
        with p.open('a') as f: f.write('{"id":"b"}\n')
        if self.fail_eval_once:
            self.fail_eval_once = False
            raise RuntimeError('simulated mid-run shutdown')
        self.completed.add(name)

    def test_resume_skips_completed_runs_and_preserves_bytes(self):
        r.main()
        self.assertEqual(self.eval_calls, r.RUN_NAMES[3:])
        for path, data in self.initial.items(): self.assertEqual(Path(path).read_bytes(), data)
        self.assertTrue(r.read(self.out / 'completed.json')['all_repeats_passed'])
        self.assertEqual(r.read(self.out / 'resume_status.json')['stage'], 'complete')

    def test_second_resume_after_interruption_retains_prior_logs_and_saved_prefix(self):
        self.fail_eval_once = True
        with self.assertRaisesRegex(RuntimeError, 'shutdown'): r.main()
        first = next((self.out / 'resume_attempts').iterdir())
        failure = (first / 'failed.json').read_bytes()
        p = self.out / r.RUN_NAMES[3] / 'predictions.jsonl'; prefix = p.read_bytes()
        r.main()
        self.assertTrue(p.read_bytes().startswith(prefix))
        self.assertEqual((first / 'failed.json').read_bytes(), failure)
        self.assertEqual(len(list((self.out / 'resume_attempts').iterdir())), 2)
        self.assertEqual(self.eval_calls, [r.RUN_NAMES[3]] + r.RUN_NAMES[3:])

    def test_cuda_failure_never_starts_evaluation_or_changes_saved_data(self):
        self.fail_cuda = True
        with self.assertRaisesRegex(RuntimeError, 'CUDA'): r.main()
        self.assertFalse(self.eval_calls)
        for path, data in self.initial.items(): self.assertEqual(Path(path).read_bytes(), data)
        self.assertFalse((self.out / r.RUN_NAMES[4]).exists())

    def test_repeat_failure_blocks_later_epoch(self):
        self.fail_repeat = True
        with self.assertRaisesRegex(AssertionError, 'repeatability'): r.main()
        self.assertEqual(self.eval_calls, [r.RUN_NAMES[3]])
        self.assertFalse((self.out / 'completed.json').exists())

    def test_second_invocation_on_completed_campaign_calls_no_model(self):
        r.main(); self.eval_calls.clear(); self.fail_cuda = True
        r.main()
        self.assertFalse(self.eval_calls)

    def test_campaign_lock_rejects_duplicate_controller(self):
        import fcntl
        with (self.out / '.campaign.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError): r.main()
        self.assertFalse(self.eval_calls)

    def test_orphan_evaluator_blocks_resume_even_without_campaign_lock(self):
        with patch.object(r, 'existing_workers', return_value=[{'pid':123,'argv':['evaluate.py']} ]):
            with self.assertRaisesRegex(RuntimeError, 'refusing duplicate'): r.main()
        self.assertFalse(self.eval_calls)


if __name__ == '__main__':
    unittest.main()
