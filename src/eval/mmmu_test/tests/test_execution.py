"""Exercise the real entrypoint and generator with fake model/GPU interfaces.

These tests never download models or run GPU inference. They check which model,
revision, Test split, rendered prompt, seed and decoding settings reach inference.
"""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/diag'))
spec = importlib.util.spec_from_file_location('run_generate_under_test', ROOT / 'scripts/diag/run_generate.py')
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)
spec = importlib.util.spec_from_file_location('judge_under_test', ROOT / 'scripts/diag/judge.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def module(name, **values):
    result = types.ModuleType(name)
    result.__dict__.update(values)
    return result


class GeneratorRoutingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = json.loads((ROOT / 'eval_settings.json').read_text(encoding='utf-8'))['settings']
        self.settings['dataset']['split'] = 'test'
        self.settings['runtime']['max_images_per_prompt'] = 7
        self.record = {'id': 'test_Art_1', 'dataset_config': 'Art', 'image_fields': ['image_1'],
                       'input_prompt': 'Synthetic test question.\nFinal answer: LETTER',
                       'rendered_chat': '<synthetic-user><image>EXACT RENDERED PROMPT<assistant>',
                       'question_type': 'multiple-choice', 'reference_answer': 'A',
                       'options': {'A': 'synthetic yes', 'B': 'synthetic no'},
                       'question': 'Synthetic question?', 'model_response': {'input_tokens': None}}
        self.records = self.root / 'records.json'
        self.records.write_text(json.dumps({'evaluation': {'settings': self.settings},
                                            'records': [self.record]}), encoding='utf-8')
        self.calls = {}
        calls = self.calls

        class Image:
            def convert(self, mode):
                calls['image_mode'] = mode
                return self

        class Dataset:
            def __getitem__(self, key):
                return ['test_Art_1'] if key == 'id' else {'image_1': Image()}

        def load_dataset(repo, config, **kwargs):
            calls['dataset'] = (repo, config, kwargs)
            return Dataset()

        def process_vision_info(messages, **kwargs):
            calls['vision'] = (messages, kwargs)
            return ['synthetic-image'], None, {}

        class Arguments:
            def __init__(self, **kwargs):
                calls['engine'] = kwargs

        class Sampling:
            def __init__(self, **kwargs):
                calls['sampling'] = kwargs

        class Engine:
            @classmethod
            def from_engine_args(cls, args):
                return cls()

            async def generate(self, prompt, sampling, request_id):
                calls['prompt'] = prompt
                output = types.SimpleNamespace(text='Final answer: A', token_ids=[1, 2], finish_reason='stop')
                yield types.SimpleNamespace(prompt_token_ids=[3], outputs=[output], finished=True)

            def shutdown(self):
                calls['shutdown'] = True

        self.modules = {
            'vllm': module('vllm', AsyncLLMEngine=Engine, SamplingParams=Sampling),
            'vllm.engine': module('vllm.engine'),
            'vllm.engine.arg_utils': module('vllm.engine.arg_utils', AsyncEngineArgs=Arguments),
            'vllm.sampling_params': module('vllm.sampling_params', RequestOutputKind=types.SimpleNamespace(DELTA='delta')),
            'datasets': module('datasets', load_dataset=load_dataset),
            'qwen_vl_utils': module('qwen_vl_utils', process_vision_info=process_vision_info),
        }

    def run_generator(self, model, seed):
        args = types.SimpleNamespace(baseline=str(self.records), out=str(self.root / 'predictions.jsonl'),
                                     model_path=model, seed=seed, tag='test_run', ids_file=None, types=None, limit=None,
                                     max_num_seqs=16, max_tokens=32768, max_num_batched_tokens=4096,
                                     gpu_mem=0.9, no_eager=False, loop_stop=True, loop_th=0.5, loop_consec=2)
        with patch.dict(sys.modules, self.modules), patch.dict(os.environ):
            asyncio.run(generate.run(args))
        row = json.loads(Path(args.out).read_text(encoding='utf-8'))
        self.assertEqual(row['seed'], seed)
        self.assertEqual(row['model_path'], model)
        self.assertEqual(row['id'], self.record['id'])
        self.assertEqual(self.calls['dataset'], (self.settings['dataset']['repo'], 'Art',
                         {'split': 'test', 'revision': self.settings['dataset']['revision']}))
        self.assertEqual(self.calls['prompt']['prompt'], self.record['rendered_chat'])
        self.assertFalse(self.calls['prompt']['mm_processor_kwargs']['do_resize'])
        self.assertEqual(self.calls['image_mode'], 'RGB')
        self.assertEqual(self.calls['engine']['seed'], seed)
        self.assertEqual(self.calls['engine']['limit_mm_per_prompt'], {'image': 7})
        self.assertEqual(self.calls['engine']['max_num_seqs'], 16)
        self.assertEqual(self.calls['engine']['max_model_len'], 64000)
        self.assertEqual(self.calls['engine']['dtype'], 'bfloat16')
        for key in ['temperature', 'top_p', 'top_k', 'repetition_penalty', 'presence_penalty']:
            self.assertEqual(self.calls['sampling'][key], self.settings['sampling'][key])
        self.assertEqual(self.calls['sampling']['seed'], seed)
        self.assertEqual(self.calls['sampling']['max_tokens'], 32768)
        self.assertTrue(self.calls['shutdown'])

    def test_baseline_routes_to_official_pinned_model_and_test_images(self):
        self.run_generator(None, 3407)
        self.assertEqual(self.calls['engine']['model'], 'Qwen/Qwen3-VL-4B-Instruct')
        self.assertEqual(self.calls['engine']['revision'], self.settings['model']['revision'])

    def test_dpo_routes_to_supplied_weights_with_same_prompt_and_settings(self):
        self.run_generator('/models/dpo-e merged', 3407)
        self.assertEqual(self.calls['engine']['model'], '/models/dpo-e merged')
        self.assertIsNone(self.calls['engine']['revision'])

    def test_seed42_changes_both_engine_and_request_seed(self):
        self.run_generator('/models/dpo-e merged', 42)

    def test_judge_seed_stays3407_when_prediction_seed_is42(self):
        calls = {}

        class LLM:
            def __init__(self, **kwargs):
                calls['engine'] = kwargs

            def chat(self, conversations, sampling, **kwargs):
                calls['chat'] = kwargs
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='Your output: A')])]

        class Sampling:
            def __init__(self, **kwargs):
                calls['sampling'] = kwargs

        pred = self.root / 'judge-input.jsonl'
        pred.write_text(json.dumps({'id': self.record['id'], 'seed': 42, 'model_response': {
            'raw': 'The response requires semantic matching.', 'generation_status': 'normal_stop'}})+'\n', encoding='utf-8')
        out = self.root / 'judge-output.jsonl'
        argv = ['judge.py', '--baseline', str(self.records), '--pred', str(pred), '--out', str(out)]
        with patch.dict(sys.modules, {'vllm': module('vllm', LLM=LLM, SamplingParams=Sampling)}), patch.object(sys, 'argv', argv):
            judge.main()
        self.assertEqual(calls['engine']['model'], 'Qwen/Qwen3-8B')
        self.assertEqual(calls['engine']['seed'], 3407)
        self.assertEqual(calls['sampling']['seed'], 3407)
        self.assertFalse(calls['chat']['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(json.loads(out.read_text())['answer'], 'A')


class ShellEntrypointTest(unittest.TestCase):
    def setUp(self):
        self.bash = os.environ.get('BASH_FOR_TESTS') or shutil.which('bash')
        if not self.bash:
            self.skipTest('Bash is required for entrypoint integration checks')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'scripts').mkdir()
        (self.root / 'checkpoint with spaces').mkdir()
        (self.root / 'official snapshot').mkdir()
        (self.root / 'scripts/run_test.sh').write_bytes((ROOT / 'scripts/run_test.sh').read_bytes())
        self.write('fakepy', '#!/usr/bin/env bash\nset -eu\nprintf "<%s>\\n" "$@" >> calls.txt\n'
                   'if [[ "${2:-}" == base-path ]]; then printf "%s/official snapshot\\n" "$PWD"; fi\n')
        self.write('scripts/run_eval_dpo2e.sh', '#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$@" > spec.txt\n')

    def write(self, rel, text):
        p = self.root / rel
        with p.open('w', encoding='utf-8', newline='\n') as f:
            f.write(text)
        p.chmod(0o755)

    def invoke(self, *args, lock_status=0):
        # Model/Python and locking are injected interfaces; real flock is Linux-only.
        setup = ('flock() { return "$TEST_LOCK_STATUS"; }; export -f flock; '
                 'export PY="$PWD/fakepy"; exec bash scripts/run_test.sh "$@"')
        return subprocess.run([self.bash, '-c', setup, 'entry-test', *args], cwd=self.root,
                              env={**os.environ, 'TEST_LOCK_STATUS': str(lock_status)},
                              capture_output=True, text=True)

    def check_spec(self, seed, tag, args, *, baseline=False):
        result = self.invoke(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        spec = (self.root / 'spec.txt').read_text().strip().split(':')
        self.assertEqual(spec[:3], [tag, str(seed), '32768'])
        self.assertEqual(spec[4:], ['records/mmmu_test.json', '1'])
        self.assertEqual(spec[3] == '', baseline)
        if not baseline:
            self.assertTrue(spec[3].endswith('/checkpoint with spaces'))
        calls = (self.root / 'calls.txt').read_text()
        self.assertIn('<--generation-seed>\n<'+str(seed)+'>', calls)
        self.assertEqual('<--baseline>' in calls, baseline)
        self.assertEqual('<base-path>' in calls, baseline)
        if baseline:
            self.assertIn('/official snapshot>', calls)
        self.assertIn('<verify>\n<'+tag+'>', calls)

    def test_baseline_seed3407_routes_to_official_model(self):
        self.check_spec(3407, 'base_test', ['baseline', 'base_test', '3407'], baseline=True)

    def test_existing_merged_command_preserves_argument_order(self):
        self.check_spec(3407, 'historical', ['checkpoint with spaces', 'historical', '3407'])

    def test_existing_merged_command_defaults_to_seed42(self):
        self.check_spec(42, 'trained_test_seed42', ['checkpoint with spaces', 'trained_test_seed42'])

    def test_invalid_argument_count_stops_before_inference(self):
        result = self.invoke('baseline')
        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.root / 'spec.txt').exists())

    def test_failed_lock_stops_before_inference(self):
        result = self.invoke('baseline', 'test_lock', lock_status=1)
        self.assertEqual(result.returncode, 1)
        self.assertIn('already running', result.stderr)
        self.assertFalse((self.root / 'spec.txt').exists())


if __name__ == '__main__':
    unittest.main()
