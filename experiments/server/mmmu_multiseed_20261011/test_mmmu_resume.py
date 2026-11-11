"""CPU-only tests, including an isolated Linux pause with real child processes."""
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from mmmu_resume_utils import check_manifest, repair_jsonl
from mmmu_multiseed_report import read, write


class ResumeTests(unittest.TestCase):
    def test_truncated_tail_and_missing_newline(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rows.jsonl'
            valid = b'{"id":"done"}\n'
            path.write_bytes(valid + b'{"id":"interrupted')
            result = repair_jsonl(path)
            self.assertEqual(path.read_bytes(), valid)
            self.assertEqual(result['records'], 1)
            self.assertEqual(Path(result['backup']).read_bytes(), valid + b'{"id":"interrupted')
            path.write_bytes(b'{"id":"done"}')
            self.assertEqual(repair_jsonl(path)['records'], 1)
            self.assertEqual(path.read_bytes(), valid)

    def test_corruption_in_complete_row_is_refused_without_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'rows.jsonl'
            for content in (b'{bad}\n{"id":"done"}\n', b'{bad}\n'):
                path.write_bytes(content)
                with self.assertRaises(ValueError):
                    repair_jsonl(path)
                self.assertEqual(path.read_bytes(), content)

    def test_changed_checkpoint_seed_or_code_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            config, identity, code = {'seed': 123}, {'weights': 'abc'}, {'evaluate.py': 'def'}
            fingerprint = hashlib.sha256(json.dumps({'configuration': config, 'model_identity': identity, 'code': code}, sort_keys=True).encode()).hexdigest()
            write(folder / 'manifest.json', {'fingerprint': fingerprint, 'mode': 'inference'})
            check_manifest(folder, config, identity, code)
            for changed in [({'seed': 42}, identity, code), (config, {'weights': 'other'}, code), (config, identity, {'evaluate.py': 'other'})]:
                with self.assertRaises(ValueError):
                    check_manifest(folder, *changed)

    @unittest.skipUnless(sys.platform == 'linux', 'Linux process-group integration test')
    def test_real_pause_stops_only_test_group_and_protects_3407(self):
        import control_mmmu_multiseed as control
        from mmmu_resume_utils import active_group
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            driver = root / 'run_mmmu_multiseed.py'
            driver.write_text('import subprocess,sys,time\np=subprocess.Popen([sys.executable,"-c","import time; time.sleep(90)"])\nprint(p.pid,flush=True)\ntime.sleep(90)\n')
            process = subprocess.Popen([sys.executable, str(driver), '--root', str(root)], start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(90)'], start_new_session=True)
            try:
                child = int(process.stdout.readline())
                state = {'state': 'running', 'current_run': 'baseline_seed3407_repeat', 'driver_pid': process.pid}
                write(root / 'status.json', state)
                with self.assertRaisesRegex(RuntimeError, 'protected'):
                    control.pause(root)
                self.assertIsNone(process.poll())
                self.assertIsNone(unrelated.poll())
                state['current_run'] = 'baseline_seed123'
                write(root / 'status.json', state)
                control.pause(root)
                process.wait(timeout=5)
                self.assertFalse(active_group(process.pid))
                self.assertEqual(read(root / 'status.json')['state'], 'paused')
                self.assertIsNone(unrelated.poll())
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                process.stdout.close()
                unrelated.terminate()
                unrelated.wait()

    @unittest.skipUnless(sys.platform == 'linux', 'Requires server frozen code and historical fixture')
    def test_frozen_infer_skips_saved_item(self):
        sys.path.insert(0, '/home/rlatjswo/mmdl_test/src')
        import evaluate
        config = read('/home/rlatjswo/mmmu_baseline_20261009/protocol/config.seed3407.json')
        config['sampling']['seed'] = 123
        records = read('/home/rlatjswo/mmmu_baseline_20261009/baseline_seed3407/results.json')['records'][:2]
        saved = {k: v for k, v in records[0].items() if k != 'grading'}
        rows = [{**r, 'answer': r['reference_answer'], 'options': list(r['options'].values())} for r in records]
        class RGB:
            size = (32, 32)
            def convert(self, mode):
                return self
        class Processor:
            image_processor = types.SimpleNamespace(patch_size=16, merge_size=2)
            def apply_chat_template(self, *a, **kw):
                return 'test rendered'
            def __call__(self, **kw):
                return {'input_ids': types.SimpleNamespace(shape=(1, 3))}
        generated = []
        class LLM:
            def __init__(self, **kw):
                pass
            def generate(self, requests, sample, **kw):
                generated.append(requests)
                return [types.SimpleNamespace(prompt_token_ids=[1, 2, 3], outputs=[types.SimpleNamespace(text='Final answer: A', token_ids=[4, 5], finish_reason='stop', stop_reason=None)])]
        fake_modules = {'transformers': types.SimpleNamespace(AutoProcessor=types.SimpleNamespace(from_pretrained=lambda *a, **kw: Processor())),
                        'vllm': types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw),
                        'qwen_vl_utils': types.SimpleNamespace(process_vision_info=lambda *a, **kw: ([RGB()], None, {}))}
        data = {r['subject']: {r['dataset_index']: {'image_1': RGB()} for r in rows} for r in rows}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            evaluate.prepare.append(folder / 'predictions.jsonl', saved)
            with patch.dict(sys.modules, fake_modules), patch.object(evaluate.prepare, 'runtime_check'), patch.object(evaluate.prepare, 'load_data', return_value=(data, rows)), patch.object(evaluate.prepare, 'snapshot', return_value=folder), patch.object(evaluate, 'prompt_for', side_effect=lambda row, c: row['input_prompt']):
                evaluate.infer(types.SimpleNamespace(checkpoint=None), config, folder, {'cache_paths': {'dataset': temp, 'model': temp}})
            output = evaluate.prepare.read_jsonl(folder / 'predictions.jsonl')
            self.assertEqual(len(generated), 1)
            self.assertEqual(output[0], saved)
            self.assertEqual([r['id'] for r in output], [r['id'] for r in records])
            # A second invocation reuses both rows and makes no additional generation call.
            with patch.dict(sys.modules, fake_modules), patch.object(evaluate.prepare, 'runtime_check'), patch.object(evaluate.prepare, 'load_data', return_value=(data, rows)), patch.object(evaluate.prepare, 'snapshot', return_value=folder), patch.object(evaluate, 'prompt_for', side_effect=lambda row, c: row['input_prompt']):
                evaluate.infer(types.SimpleNamespace(checkpoint=None), config, folder, {'cache_paths': {'dataset': temp, 'model': temp}})
            self.assertEqual(len(generated), 1)

    @unittest.skipUnless(sys.platform == 'linux', 'Requires server frozen judge code')
    def test_frozen_judge_continues_attempt_and_skips_completed_target(self):
        sys.path.insert(0, '/home/rlatjswo/mmdl_test/src')
        import evaluate
        import judge
        config = read('/home/rlatjswo/mmmu_baseline_20261009/protocol/config.seed3407.json')
        original = read('/home/rlatjswo/mmmu_baseline_20261009/baseline_seed3407/results.json')['records']
        rows = [r for r in original if r['grading']['final']['method'] == 'local_qwen3_8b_judge_parser'][:2]
        old_attempts = [{'id': r['id'], 'attempt': 1, 'raw': ('A' if i == 0 else 'unparsed answer'),
                         'input_sha256': evaluate.judge_input_digest(r, config)} for i, r in enumerate(rows)]
        calls = []
        class Tokenizer:
            def apply_chat_template(self, *a, tokenize, **kw):
                return [1] if tokenize else 'test judge prompt'
        class LLM:
            def __init__(self, **kw):
                pass
            def generate(self, request, *a, **kw):
                calls.append(request)
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='A', token_ids=[1], finish_reason='stop')])]
        modules = {'transformers': types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())),
                   'vllm': types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **kw: kw)}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for row in rows:
                evaluate.prepare.append(folder / 'predictions.jsonl', row)
            for attempt in old_attempts:
                evaluate.prepare.append(folder / 'judge_attempts.jsonl', attempt)
            write(folder / 'pending_judge.json', [rows[1]['id']])
            with patch.dict(sys.modules, modules), patch.object(evaluate.prepare, 'runtime_check'), patch.object(evaluate.prepare, 'snapshot', return_value=folder):
                judge.run(config, folder, {'cache_paths': {'model': temp}})
            attempts = evaluate.prepare.read_jsonl(folder / 'judge_attempts.jsonl')
            self.assertEqual(attempts[:2], old_attempts)
            self.assertEqual(len(calls), 1)
            self.assertEqual(attempts[-1]['id'], rows[1]['id'])
            self.assertEqual(attempts[-1]['attempt'], 2)

    @unittest.skipUnless(sys.platform == 'linux', 'Requires server frozen campaign module')
    def test_resume_controller_skips_finished_seed_and_retains_partial_seed(self):
        import resume_mmmu_multiseed as runner
        original = runner.original
        config = read(original.OLD / 'protocol/config.seed3407.json')
        baseline = read(original.BASELINE / 'results.json')
        identity = baseline['manifest']['model_identity']
        code = {str(p.relative_to(original.REPO)): original.prepare.digest(p) for p in sorted((original.REPO / 'src').glob('*')) if p.is_file()}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'campaign'
            root.mkdir()
            write(root / 'status.json', {'state': 'paused', 'current_run': 'baseline_seed2026', 'completed_new_runs': []})
            write(root / 'protocol/config.seed3407.json', config)
            write(root / 'protocol/evaluation_items.json', [{'id': r['id']} for r in baseline['records']])
            for seed in (123, 2026, 9876):
                current = copy.deepcopy(config)
                current['sampling']['seed'] = seed
                write(root / f'protocol/config.seed{seed}.json', current)
                if seed in (123, 2026):
                    fingerprint = hashlib.sha256(json.dumps({'configuration': current, 'model_identity': identity, 'code': code}, sort_keys=True).encode()).hexdigest()
                    write(root / f'baseline_seed{seed}/manifest.json', {'fingerprint': fingerprint, 'mode': 'inference'})
            write(root / 'baseline_seed123/summary.json', {})
            saved = {k: v for k, v in baseline['records'][0].items() if k != 'grading'}
            original.prepare.append(root / 'baseline_seed2026/predictions.jsonl', saved)
            calls = []
            def execute(command, **kwargs):
                if command[0] == 'tar':
                    Path(command[2]).write_bytes(b'test archive')
                else:
                    calls.append(command[command.index('--run-name')+1])
                    self.assertEqual(original.prepare.read_jsonl(root / 'baseline_seed2026/predictions.jsonl'), [saved])
                return types.SimpleNamespace(returncode=0)
            with patch.object(sys, 'argv', ['resume', '--root', str(root)]), patch.object(original, 'preflight'), patch.object(runner, 'validate', return_value=baseline), patch.object(runner, 'compare_repeat', return_value={'passed': True}), patch.object(runner, 'aggregate'), patch.object(original.evaluate, 'model_identity', return_value=identity), patch.object(original.evaluate, 'summaries', return_value=baseline['summary']), patch.object(subprocess, 'check_output', return_value='24, 0\n'), patch.object(subprocess, 'run', side_effect=execute):
                runner.main()
            self.assertEqual(calls, ['baseline_seed2026', 'baseline_seed9876'])
            self.assertEqual(read(root / 'status.json')['state'], 'complete')
            self.assertTrue(read(root / 'baseline_seed2026/verification.json')['resumed'])
            self.assertFalse(read(root / 'baseline_seed9876/verification.json')['resumed'])
            self.assertEqual(original.prepare.read_jsonl(root / 'baseline_seed2026/predictions.jsonl'), [saved])


if __name__ == '__main__':
    unittest.main(verbosity=2)
