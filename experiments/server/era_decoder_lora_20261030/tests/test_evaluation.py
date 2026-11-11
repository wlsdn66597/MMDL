"""CPU-only regression checks for frozen evaluation and paired analyses."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from era_lora.common import load_config
from era_lora.evaluate import (build_plan, execute_plan, SOURCE_HASHES, VERSIONS,
    validate_frozen_config, validate_source, validate_replay_gate, validate_training_pair,
    checkpoint_preflight, BASE_REVISION)
from era_lora.compare import (BASE_WEIGHTS, compare_records, compare_repeats, load_run,
                              repetition_metrics, require_seed_sets, validate_records)

ROOT = Path(__file__).resolve().parents[1]


def example_record(identifier='validation_Accounting_1', hit=0):
    return {'id': identifier, 'subject': 'Accounting', 'dataset_config': 'Accounting',
        'dataset_index': 0, 'question_type': 'multiple-choice', 'question': 'What is shown?',
        'options': {'A': 'one', 'B': 'two'}, 'reference_answer': 'B',
        'image_fields': ['image_1'], 'original_image_sizes_wh': [[32, 32]],
        'input_prompt': 'Question: What is shown?', 'rendered_chat': 'frozen chat',
        'model_response': {'raw': 'Final answer: ' + ('B' if hit else 'A'),
            'finish_reason': 'stop', 'stop_reason': None, 'output_tokens': 5,
            'input_tokens': 12, 'elapsed_seconds': 1.0},
        'grading': {'final': {'status': 'graded', 'hit': hit,
            'method': 'explicit_final_parser', 'extracted_answer': 'B' if hit else 'A'}}}


def write_run(path, records, seed=3407):
    path.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / f'baseline/protocol/config.seed{seed}.json').read_text())
    manifest = {'configuration': config, 'mode': 'inference', 'code_sha256': SOURCE_HASHES,
        'model_identity': {'path': '/fixed/checkpoint', 'weights_sha256': BASE_WEIGHTS},
        'environment': {'versions': VERSIONS, 'torch_cuda': '12.8',
            'gpu': 'NVIDIA GeForce RTX 4090', 'python': '3.12.14', 'platform': 'Linux-test'}}
    summary = {'total': len(records), 'correct': sum(r['grading']['final']['hit'] for r in records)}
    summary['accuracy_percent'] = 100 * summary['correct'] / summary['total']
    result = {'schema_version': 'mmmu-frozen-evaluation-v1', 'manifest': manifest,
              'summary': summary, 'records': records}
    for name, value in [('manifest.json', manifest), ('summary.json', summary), ('results.json', result),
                         ('launcher.json', {'fresh_output_verified': True,
                          'vllm_environment': {'VLLM_ENABLE_V1_MULTIPROCESSING': '0'},
                          'python_probe': {'python': [3, 12], 'versions': VERSIONS},
                          'checkpoint_sha256': BASE_WEIGHTS})]:
        (path / name).write_text(json.dumps(value), encoding='utf-8')
    predictions = [{k: v for k, v in record.items() if k != 'grading'} for record in records]
    (path / 'predictions.jsonl').write_text('\n'.join(json.dumps(r) for r in predictions) + '\n', encoding='utf-8')
    return config


class FrozenEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / 'configs/pilot.json')

    def test_archived_source_and_all_seed_configs(self):
        _, _, configs, _ = validate_source(self.cfg)
        self.assertEqual(list(configs), [3407, 42, 123, 2026, 9876])

    def test_generation_config_drift_is_rejected(self):
        c = json.loads((ROOT / 'baseline/protocol/config.seed3407.json').read_text())
        c['sampling']['presence_penalty'] = 0
        with self.assertRaisesRegex(ValueError, 'drift'):
            validate_frozen_config(c, 3407)

    def test_frozen_prompt_drift_is_rejected_even_if_manifest_rewritten(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'baseline'
            shutil.copytree(ROOT / 'baseline', root)
            config = root / 'protocol/config.seed42.json'
            c = json.loads(config.read_text())
            c['prompts']['open'] += '\nUse ERA.'
            config.write_text(json.dumps(c))
            from era_lora.common import sha256_file
            manifest_path = root / 'source_manifest.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['sha256']['protocol/config.seed42.json'] = sha256_file(config)
            manifest_path.write_text(json.dumps(manifest))
            cfg = copy.deepcopy(self.cfg)
            cfg['evaluation'].update(source_dir=str(root), protocol_dir=str(root / 'protocol'),
                                     source_manifest=str(manifest_path))
            with self.assertRaisesRegex(ValueError, 'beyond seed'):
                validate_source(cfg)

    def test_plan_never_launches_python_or_gpu(self):
        with patch('era_lora.evaluate.subprocess.run') as run, patch('era_lora.evaluate.subprocess.check_output') as probe:
            plan = build_plan(self.cfg)
        run.assert_not_called()
        probe.assert_not_called()
        self.assertFalse(plan['gpu_execution_performed'])
        self.assertEqual(len(plan['commands']), 12)
        self.assertTrue(plan['commands'][1]['repeat'])
        self.assertEqual(plan['environment_unset'], ['VLLM_BATCH_INVARIANT'])

    def test_missing_and_unexpected_seeds_are_rejected(self):
        runs = {arm: {str(s): '/placeholder' for s in [3407, 42, 123, 2026, 9876]}
                for arm in ('baseline', 'plain', 'era')}
        del runs['era']['9876']
        with self.assertRaisesRegex(ValueError, 'Missing/unexpected'):
            require_seed_sets(runs)

    def test_baseline_plan_contains_only_one_fresh_replay(self):
        plan = build_plan(self.cfg, ('baseline',))
        self.assertEqual(len(plan['commands']), 1)
        self.assertEqual(plan['commands'][0]['seed'], 3407)
        self.assertFalse(plan['commands'][0]['repeat'])
        self.assertIn('baseline_seed3407_replay', plan['commands'][0]['run_dir'])

    def test_checkpoint_rejects_additional_processor_overrides_before_hashing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('special_tokens_map.json', 'added_tokens.json', 'processor_config.json',
                         'chat_template.jinja', 'chat_templates'):
                with self.subTest(name=name):
                    checkpoint = root / name.replace('.', '_')
                    checkpoint.mkdir()
                    override = checkpoint / name
                    if name == 'chat_templates':
                        override.mkdir()
                    else:
                        override.write_text('{}')
                    with patch('era_lora.evaluate.sha256_file') as digest:
                        with self.assertRaises(ValueError):
                            checkpoint_preflight(checkpoint)
                        digest.assert_not_called()

    def test_only_pinned_baseline_can_follow_nested_hf_global_blob_symlinks(self):
        from era_lora.common import sha256_file
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / 'models--Qwen--Qwen3-VL-4B-Instruct'
            snapshot = repo / 'snapshots' / BASE_REVISION
            snapshot.mkdir(parents=True)
            blobs = repo / 'blobs'
            blobs.mkdir()
            blob = blobs / 'weight-blob'
            shared = Path(temp) / 'blobs' / 'e7' / 'shared-weight'
            shared.parent.mkdir(parents=True)
            shared.write_bytes(b'fake original weight for containment test')
            shard = snapshot / 'model.safetensors'
            try:
                blob.symlink_to(Path('..') / '..' / 'blobs' / 'e7' / shared.name)
                shard.symlink_to(Path('..') / '..' / 'blobs' / blob.name)
            except OSError:
                self.skipTest('Windows symlink permission unavailable')
            self.assertEqual(shard.resolve(), shared.resolve())
            self.assertFalse(shard.resolve().is_relative_to(blobs.resolve()))
            config = {'model_type': 'qwen3_vl', 'text_config': {'dtype': 'bfloat16',
                'hidden_size': 2560, 'num_hidden_layers': 36, 'num_attention_heads': 32, 'vocab_size': 151936}}
            (snapshot / 'config.json').write_text(json.dumps(config))
            expected = {'model.safetensors': sha256_file(blob)}
            with patch('era_lora.evaluate.AUX_HASHES', {}), \
                    patch('era_lora.evaluate.BASE_WEIGHTS', expected), \
                    patch('era_lora.evaluate.BASE_CONFIG_SHA256', sha256_file(snapshot / 'config.json')):
                with self.assertRaisesRegex(ValueError, 'invalid checkpoint shard'):
                    checkpoint_preflight(snapshot)
                hashes = checkpoint_preflight(snapshot, allow_baseline_snapshot=True)
                self.assertEqual(hashes['model.safetensors'], expected['model.safetensors'])
                blob.write_bytes(b'corrupted cache blob')
                with self.assertRaisesRegex(ValueError, 'original baseline weight'):
                    checkpoint_preflight(snapshot, allow_baseline_snapshot=True)

    def test_missing_replay_gate_blocks_execution_before_subprocess(self):
        plan = build_plan(self.cfg)
        for item in plan['readiness'].values():
            item['ready'] = True
        with tempfile.TemporaryDirectory() as temp:
            cfg = copy.deepcopy(self.cfg)
            cfg['evaluation']['replay_receipt'] = str(Path(temp) / 'missing.json')
            with patch('era_lora.evaluate.platform.system', return_value='Linux'), \
                    patch('era_lora.evaluate.subprocess.run') as run, \
                    patch('era_lora.evaluate.subprocess.check_output') as probe:
                with self.assertRaises(FileNotFoundError):
                    execute_plan(plan, cfg)
            run.assert_not_called()
            probe.assert_not_called()

    def test_replay_receipt_is_bound_to_experiment_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            receipt = Path(temp) / 'replay.json'
            receipt.write_text(json.dumps({'passed': True, 'config_sha256': 'another-config',
                                          'source_hashes': SOURCE_HASHES}))
            cfg = copy.deepcopy(self.cfg)
            cfg['evaluation']['replay_receipt'] = str(receipt)
            with self.assertRaisesRegex(ValueError, 'configuration/source'):
                validate_replay_gate(cfg, SOURCE_HASHES)

    def test_two_training_arms_require_matched_data_init_and_distinct_adapters(self):
        proofs = {arm: {'identity': {'arm': arm, 'config_sha256': 'config',
                  'data_sha256': 'paired-data', 'source_sha256': {'train.py': 'code'}},
                  'initial_adapter_sha256': 'same-init', 'final_adapter_sha256': arm,
                  'actual_optimizer_steps': 26} for arm in ('plain', 'era')}
        validate_training_pair(proofs)
        bad = copy.deepcopy(proofs)
        bad['era']['identity']['data_sha256'] = 'other-data'
        with self.assertRaisesRegex(ValueError, 'configuration, data'):
            validate_training_pair(bad)
        bad = copy.deepcopy(proofs)
        bad['era']['initial_adapter_sha256'] = 'other-init'
        with self.assertRaisesRegex(ValueError, 'initialization'):
            validate_training_pair(bad)
        bad = copy.deepcopy(proofs)
        bad['era']['final_adapter_sha256'] = 'plain'
        with self.assertRaisesRegex(ValueError, 'distinct'):
            validate_training_pair(bad)

    def test_pair_requires_same_ids_and_same_prompt(self):
        a = example_record()
        b = example_record('validation_Accounting_2')
        with self.assertRaisesRegex(ValueError, 'IDs'):
            compare_records([a], [b])
        b = copy.deepcopy(a)
        b['input_prompt'] += ' Use ERA.'
        with self.assertRaisesRegex(ValueError, 'prompt drift'):
            compare_records([a], [b])

    def test_duplicate_and_truncated_correct_records_are_rejected(self):
        a = example_record()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_records([a, a])
        b = example_record(hit=1)
        b['model_response']['finish_reason'] = 'length'
        with self.assertRaisesRegex(ValueError, 'incorrect'):
            validate_records([b])

    def test_paired_transitions_and_mechanical_repetition(self):
        comparison = compare_records([example_record()], [example_record(hit=1)])
        self.assertEqual(comparison['accuracy_delta_pp'], 100)
        self.assertEqual(comparison['transition_counts']['wrong_to_right'], 1)
        repeated = repetition_metrics(('these same six words repeat over here\n' * 3))
        self.assertTrue(repeated['has_line_repeated_3_times'])
        self.assertGreater(repeated['repeated_4gram_fraction'], 0)

    def test_saved_config_drift_and_missing_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            config = write_run(path, [example_record()])
            drifted = copy.deepcopy(config)
            drifted['max_new_tokens'] = 1024
            with self.assertRaisesRegex(ValueError, 'configuration drift'):
                load_run(path, 3407, drifted)
            load_run(path, 3407, config)
            (path / 'launcher.json').unlink()
            with self.assertRaises(FileNotFoundError):
                load_run(path, 3407, config)

    def test_repeatability_checks_raw_text_and_grades_not_only_accuracy(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / 'first', Path(temp) / 'second'
            a = example_record()
            b = copy.deepcopy(a)
            b['model_response']['elapsed_seconds'] = 50
            write_run(first, [a])
            write_run(second, [b])
            self.assertTrue(compare_repeats(first, second)['passed'])
            b['model_response']['raw'] = 'Another rationale. Final answer: A'
            write_run(second, [b])
            report = compare_repeats(first, second)
            self.assertFalse(report['passed'])
            self.assertIn(a['id'] + '.model_response', report['differences'])

    def test_prediction_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            write_run(path, [example_record()])
            record = json.loads((path / 'predictions.jsonl').read_text())
            record['model_response']['raw'] = 'changed'
            (path / 'predictions.jsonl').write_text(json.dumps(record) + '\n')
            with self.assertRaisesRegex(ValueError, 'responses differ'):
                load_run(path)


if __name__ == '__main__':
    unittest.main()
