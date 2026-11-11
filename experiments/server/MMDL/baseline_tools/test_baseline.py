"""Offline tests for the helper's wiring. No real Qwen inference or MMMU download.
The evaluator fixture is a MOCK, not an alternate benchmark scorer.
"""
import argparse
import base64
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
import types
import unittest
from unittest import mock

import baseline_eval as b

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWioAAAAASUVORK5CYII=')
MOCK_SCORER = '''import random

def parse_multi_choice_response(text, choices, index2ans):
    for choice in choices:
        if '('+choice+')' in text:
            return choice
    return random.choice(choices)

def eval_multi_choice(gold, parsed):
    return gold == parsed

def parse_open_response(text):
    return [text.strip().lower()]

def eval_open(gold, parsed):
    return gold.lower() in parsed
'''


def sample(i=1, **updates):
    s = dict(id=f'validation_Art_{i}', subject='Art', category='Art and Design',
             question_type='multiple-choice', question='<image 1> Choose.',
             options=['red', 'blue', 'green'], answer='A', images={
                 'image_1': dict(path='one.png', sha256=b.sha_bytes(PNG), width=1, height=1)})
    s.update(updates)
    return s


class UnitTests(unittest.TestCase):
    def test_parse_options_safe(self):
        self.assertEqual(b.parse_options("['x','y']", 'multiple-choice'), ['x', 'y'])
        with self.assertRaises((ValueError, SyntaxError)):
            b.parse_options("__import__('os').system('false')", 'multiple-choice')

    def test_open_has_no_options(self):
        self.assertEqual(b.parse_options('not a literal', 'open'), [])
        text = b.prompt_text(sample(question_type='open', options=[]))
        self.assertNotIn('Options:', text)
        self.assertIn(b.OPEN_INSTRUCTION, text)

    def test_images_in_options_preserve_order(self):
        s = sample(options=['<image 2>', 'something'], images={'image_1': {}, 'image_2': {}})
        parts = b.content_plan(s)
        self.assertEqual([p['image_key'] for p in parts if p['type'] == 'local_image'], ['image_1', 'image_2'])

    def test_repeated_reference_is_not_dropped(self):
        s = sample(question='<image 1> and again <image 1>')
        self.assertEqual(b.audit_samples([s])['max_request_images'], 2)

    def test_missing_reference_raises(self):
        with self.assertRaises(ValueError):
            b.content_plan(sample(question='<image 2>'))

    def test_unreferenced_image_is_retained_and_reported(self):
        s = sample(question='Choose without a placeholder')
        self.assertEqual(b.audit_samples([s])['samples_with_unreferenced_images'], [s['id']])
        self.assertEqual(b.audit_samples([s])['max_request_images'], 1)

    def test_no_gold_or_explanation_in_request(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'one.png').write_bytes(PNG)
            s = sample(answer='SECRET_GOLD', explanation='SECRET_EXPLANATION')
            payload = b.jd(b.messages(s, root))
            self.assertNotIn('SECRET_GOLD', payload)
            self.assertNotIn('SECRET_EXPLANATION', payload)
            self.assertIn('data:image/png;base64,', payload)

    def test_modified_image_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'one.png').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):
                b.messages(sample(), root)

    def test_scorer_random_fallback_tracking(self):
        mod = types.ModuleType('mock_scorer')
        exec(MOCK_SCORER, mod.__dict__)
        s = sample()
        state = random.getstate()
        row = b.score_rows([s], {s['id']: {'status': 'ok', 'raw_response': 'ambiguous'}}, mod)[0]
        self.assertTrue(row['parser_random_fallback'])
        self.assertEqual(state, random.getstate())
        row = b.score_rows([s], {s['id']: {'status': 'ok', 'raw_response': '(A)'}}, mod)[0]
        self.assertTrue(row['correct'])
        self.assertFalse(row['parser_random_fallback'])

    def test_partial_not_reported_as_complete_accuracy(self):
        mod = types.ModuleType('mock_scorer')
        exec(MOCK_SCORER, mod.__dict__)
        rows = b.score_rows([sample(1), sample(2)], {sample(1)['id']: {'status': 'ok', 'raw_response': '(A)'}}, mod)
        result = b.stats(rows)
        self.assertIsNone(result['accuracy'])
        self.assertEqual(result['partial_accuracy_on_completed'], 1.0)
        self.assertEqual(result['pending'], 1)

    def test_local_only_api(self):
        b.LocalAPI('http://127.0.0.1:8000', 5)
        with self.assertRaises(ValueError):
            b.LocalAPI('https://some-paid-service.example/v1', 5)

    def test_jsonl_bad_line_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'broken.jsonl'
            p.write_text('{"id":"x"}\n{"id":')
            with self.assertRaises(RuntimeError):
                b.read_jsonl(p)

    def test_smoke_includes_multiimage_and_open(self):
        a, c = sample(1), sample(2, question_type='open', options=[])
        d = sample(3, question='<image 1> <image 2>', images={'image_1': {}, 'image_2': {}})
        chosen = b.choose_samples([a, c, d], argparse.Namespace(subject=None, smoke=3, limit=0))
        self.assertEqual({x['id'] for x in chosen}, {a['id'], c['id'], d['id']})


class IntegrationTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory) / 'data'
        root.mkdir()
        (root / 'official_mmmu').mkdir()
        (root / 'one.png').write_bytes(PNG)
        official = root / 'official_mmmu' / 'eval_utils.py'
        official.write_text(MOCK_SCORER)
        rows = []
        for group in range(30):
            for i in range(1, 31):
                rows.append(sample(i, subject=f'Subject{group}', id=f'validation_Subject{group}_{i}'))
        data = root / 'samples.jsonl'
        data.write_text(''.join(b.jd(row)+'\n' for row in rows))
        b.write_json(root / 'manifest.json', {'samples_sha256': b.sha_file(data),
                     'official_scorer': {'commit':'mock-only', 'sha256': {'eval_utils.py': b.sha_file(official)}}})
        config = Path(directory) / 'server.json'
        model = Path(directory) / 'model'
        model.mkdir()
        b.write_json(config, {'model_path': str(model.resolve()), 'vllm_version': 'mock',
                             'torch_version': 'mock', 'transformers_version': 'mock',
                             'model_hashes': {}, 'settings': {'seed':3407, 'load_format':'auto',
                                                             'max_model_len':9048}})
        return root, config, model

    def make_api(self, model, required=20, blank=False):
        class FakeAPI:
            calls = []
            def __init__(self, base_url, timeout):
                pass
            def request(self, route, payload=None):
                self.calls.append((route, payload))
                if route == '/health':
                    return {}
                if route == '/v1/models':
                    return {'data':[{'id':b.MODEL_ID, 'root':str(model.resolve()), 'max_model_len':9048}]}
                if route == '/tokenize':
                    return {'count':required, 'max_model_len':9048, 'tokens':[]}
                return {'choices':[{'message':{'content': '' if blank else '(A)'}, 'finish_reason':'stop'}],
                        'usage':{'prompt_tokens':required, 'completion_tokens':3}}
        return FakeAPI

    def args(self, root, config, out, **kwargs):
        data = dict(data_path=str(root), out=str(out), server_config=str(config),
                    base_url='http://127.0.0.1:8000', timeout=30, subject=None,
                    smoke=0, limit=3, resume=False, preflight_only=False)
        data.update(kwargs)
        return argparse.Namespace(**data)

    def test_run_resume_and_changed_settings(self):
        with tempfile.TemporaryDirectory() as d:
            root, config, model = self.fixture(d)
            fake = self.make_api(model)
            out = Path(d) / 'run'
            args = self.args(root, config, out)
            with mock.patch.object(b, 'LocalAPI', fake), mock.patch.object(b, 'environment', return_value={'mock':True}), contextlib.redirect_stdout(io.StringIO()):
                b.run(args)
                self.assertEqual(len([x for x in fake.calls if x[0] == '/v1/chat/completions']), 3)
                result = json.loads((out/'summary.json').read_text())
                self.assertTrue(result['run_complete'])
                self.assertFalse(result['full_validation_complete'])
                self.assertEqual(result['overall']['accuracy'], 1.0)
                args.resume = True
                b.run(args)
                self.assertEqual(len([x for x in fake.calls if x[0] == '/v1/chat/completions']), 3)
                args.limit = 4
                with self.assertRaises(RuntimeError):
                    b.run(args)

    def test_preflight_overflow_prevents_all_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root, config, model = self.fixture(d)
            fake = self.make_api(model, required=8000)
            out = Path(d)/'overflow'
            with mock.patch.object(b,'LocalAPI',fake), mock.patch.object(b,'environment',return_value={}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    b.run(self.args(root, config, out))
            self.assertFalse(any(x[0] == '/v1/chat/completions' for x in fake.calls))
            self.assertEqual(json.loads((out/'preflight.json').read_text())['max_required_context'], 10048)

    def test_empty_response_is_an_error_not_a_random_answer(self):
        with tempfile.TemporaryDirectory() as d:
            root, config, model = self.fixture(d)
            out = Path(d)/'empty'
            fake = self.make_api(model, blank=True)
            with mock.patch.object(b,'LocalAPI',fake), mock.patch.object(b,'environment',return_value={}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    b.run(self.args(root, config, out))
            result = json.loads((out/'summary.json').read_text())['overall']
            self.assertEqual(result['completed'], 0)
            self.assertEqual(result['errors'], 1)
            self.assertEqual(result['parser_random_fallbacks'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
