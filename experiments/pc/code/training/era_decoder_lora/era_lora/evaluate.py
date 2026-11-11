"""Plan and preflight the archived MMMU evaluator; inference requires --execute."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import subprocess

from .common import (config_digest, load_config, reject_unpinned_processor_overrides,
                     resolve_path, sha256_file, write_json)

SEEDS = [3407, 42, 123, 2026, 9876]
SOURCE_HASHES = {
    'src/config.json': 'b55e958623cd59e3da64aef4e228a98e83e586d924b80a8b3fd22dafd7c938e5',
    'src/evaluate.py': '2dbc17e8953cbdf1ea1091e2890489c7f0592efe21e5d56a13bf1b61b642e75d',
    'src/judge.py': '443a28c9cbfdd90fa6f6c5021a557dc97d58df3bed7967318ec972b773e3c0fb',
    'src/parsers.py': '5023ed38d2e8af91304b258d5b3798ffc03df24324f49740953d943378a30b04',
    'src/prepare.py': '581b4c2905fbd71f77c3e5f6f23a13c4444a125609a3c9b3ac4bfbe267f476ef',
}
AUX_HASHES = {
    'merges.txt': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3',
    'chat_template.json': '6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4',
    'generation_config.json': '8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44',
    'tokenizer.json': 'a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7',
    'tokenizer_config.json': 'c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'video_preprocessor_config.json': '7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13',
    'preprocessor_config.json': '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516',
}
VERSIONS = {'torch': '2.8.0', 'vllm': '0.11.0', 'transformers': '4.57.0',
            'qwen-vl-utils': '0.0.14', 'datasets': '4.1.1', 'Pillow': '11.3.0',
            'huggingface-hub': '0.36.2'}
BASE_REVISION = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
BASE_CONFIG_SHA256 = 'edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d'
BASE_WEIGHTS = {
    'model-00001-of-00002.safetensors': '30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9',
    'model-00002-of-00002.safetensors': '046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_frozen_config(c, seed):
    checks = {
        'model': {'repo': 'Qwen/Qwen3-VL-4B-Instruct', 'revision': BASE_REVISION,
                  'dtype': 'bfloat16', 'quantization': None},
        'sampling': {'temperature': .7, 'top_p': .8, 'top_k': 20,
                     'repetition_penalty': 1., 'presence_penalty': 1.5, 'seed': seed},
        'vision': {'min_pixels': 1003520, 'max_pixels': 4014080},
        'max_new_tokens': 32768,
    }
    for key, expected in checks.items():
        if c.get(key) != expected:
            raise ValueError(f'Frozen evaluation config drift: {key}')
    expected_runtime = {'dtype': 'bfloat16', 'quantization': None,
        'tensor_parallel_size': 1, 'max_model_len': 64000,
        'gpu_memory_utilization': .9, 'max_num_seqs': 1,
        'max_num_batched_tokens': 4096, 'enforce_eager': True, 'max_images_per_prompt': 5}
    for key, value in expected_runtime.items():
        if c['runtime'].get(key) != value:
            raise ValueError(f'Frozen evaluation runtime drift: {key}')
    if c['dataset']['repo'] != 'MMMU/MMMU' or c['dataset']['revision'] != '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68' or c['dataset']['split'] != 'validation':
        raise ValueError('Frozen evaluation dataset drift')
    if c['environment']['required_versions'] != VERSIONS:
        raise ValueError('Frozen evaluation dependency drift')
    if c['judge']['repo'] != 'Qwen/Qwen3-8B' or c['judge']['revision'] != 'b968826d9c46dd6066d109eabc6255188de91218':
        raise ValueError('Frozen judge identity drift')
    if c['judge']['sampling'] != {'temperature': 0, 'top_p': 1, 'top_k': -1,
            'repetition_penalty': 1, 'presence_penalty': 0, 'seed': 3407}:
        raise ValueError('Frozen judge sampling drift')
    if c['scoring']['length_policy'] != 'all_incorrect' or c['scoring']['random_fallback'] is not False:
        raise ValueError('Frozen scoring drift')


def validate_source(cfg):
    e = cfg['evaluation']
    root = Path(resolve_path(cfg, e['source_dir']))
    manifest = read_json(resolve_path(cfg, e['source_manifest']))
    hashes = manifest['sha256']
    if not all(hashes.get(k) == v for k, v in SOURCE_HASHES.items()):
        raise ValueError('Source manifest does not identify the executed baseline archive')
    for rel, digest in hashes.items():
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Source manifest path escapes source directory')
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f'Frozen source/artifact hash mismatch: {rel}')
    protocol = Path(resolve_path(cfg, e['protocol_dir']))
    reference = read_json(root / 'src/config.json')
    if sha256_file(protocol / 'evaluation_items.json') != '13b00a2003f30e017df49c57f92781b97781f23c5fb578c808f59fc29e3bd612':
        raise ValueError('Frozen evaluation item manifest drift')
    configs = {}
    if e['seeds'] != SEEDS:
        raise ValueError('The baseline protocol requires all five seeds in the frozen order')
    for seed in SEEDS:
        path = protocol / f'config.seed{seed}.json'
        c = read_json(path)
        validate_frozen_config(c, seed)
        normalized = copy.deepcopy(c)
        normalized['sampling']['seed'] = 3407
        if normalized != reference:
            raise ValueError(f'Config differs from archived baseline beyond seed: {seed}')
        configs[seed] = c
    return root, protocol, configs, hashes


def checkpoint_preflight(path, allow_baseline_snapshot=False):
    path = Path(path)
    reject_unpinned_processor_overrides(path)
    if allow_baseline_snapshot and (path.name != BASE_REVISION
            or sha256_file(path / 'config.json') != BASE_CONFIG_SHA256):
        raise ValueError('Only the original revision/config may use cached snapshot symlinks')
    if (path / 'adapter_config.json').exists():
        raise ValueError('Evaluation needs a merged full checkpoint, not an adapter')
    config = read_json(path / 'config.json')
    if config.get('model_type') != 'qwen3_vl' or config.get('quantization_config'):
        raise ValueError('Checkpoint architecture/quantization differs from baseline')
    text = config['text_config']
    if text.get('dtype', text.get('torch_dtype', config.get('torch_dtype'))) not in (None, 'bfloat16'):
        raise ValueError('Merged checkpoint must be BF16')
    if any(text.get(k) != v for k, v in {'hidden_size': 2560,
            'num_hidden_layers': 36, 'num_attention_heads': 32, 'vocab_size': 151936}.items()):
        raise ValueError('Checkpoint text architecture differs from frozen base')
    hashes = {}
    for rel, expected in AUX_HASHES.items():
        file = path / rel
        if not file.is_file() or sha256_file(file) != expected:
            raise ValueError(f'Checkpoint processor/tokenizer drift: {rel}')
        hashes[rel] = expected
    index = path / 'model.safetensors.index.json'
    shards = set(read_json(index)['weight_map'].values()) if index.exists() else {'model.safetensors'}
    for rel in sorted(shards):
        file = (path / rel).resolve()
        contained = file.is_relative_to(path.resolve())
        cache_blob_roots = ((path.parent.parent / 'blobs').resolve(),
                            (path.parent.parent.parent / 'blobs').resolve())
        cached_original = (allow_baseline_snapshot and rel in BASE_WEIGHTS
            and Path(rel).name == rel
            and any(file.is_relative_to(root) for root in cache_blob_roots))
        if not file.is_file() or not (contained or cached_original):
            raise ValueError(f'Missing or invalid checkpoint shard: {rel}')
        hashes[rel] = sha256_file(file)
        if allow_baseline_snapshot and hashes[rel] != BASE_WEIGHTS.get(rel):
            raise ValueError('Snapshot shard is not an original baseline weight')
    hashes['config.json'] = sha256_file(path / 'config.json')
    return hashes


def training_provenance(cfg, arm, checkpoint):
    checkpoint = Path(checkpoint)
    artifact = read_json(checkpoint / 'artifact_manifest.json')
    merged = read_json(checkpoint.parent / 'merge_manifest.json')
    trained = read_json(checkpoint.parent / 'run_manifest.json')
    if artifact != merged or not artifact.get('baseline_evaluator_compatible'):
        raise ValueError('Merged artifact and merge manifest disagree')
    identity = trained.get('identity', {})
    if (trained.get('status') != 'complete' or identity != artifact.get('training_identity')
            or identity.get('arm') != arm or identity.get('config_sha256') != config_digest(cfg)
            or identity.get('base_repo') != 'Qwen/Qwen3-VL-4B-Instruct'
            or identity.get('base_revision') != BASE_REVISION):
        raise ValueError('Candidate is not the completed requested training arm')
    if trained.get('adapter_files_sha256') != artifact.get('adapter_files_sha256'):
        raise ValueError('Merged adapter differs from completed training')
    if trained.get('frozen_weights_unchanged') is not True:
        raise ValueError('Training did not verify that original model weights stayed frozen')
    structure = artifact.get('structure_validation', {})
    if (structure.get('state_keys_identical') is not True
            or structure.get('all_non_target_tensors_identical') is not True
            or structure.get('changed_target_count', 0) <= 0
            or artifact.get('serialized_validation', {}).get('all_weights_bfloat16') is not True):
        raise ValueError('Merge did not verify decoder-only updates and BF16 serialization')
    adapter_root = checkpoint.parent / 'adapter'
    for rel, digest in trained.get('adapter_files_sha256', {}).items():
        file = (adapter_root / rel).resolve()
        if not file.is_relative_to(adapter_root.resolve()) or not file.is_file() or sha256_file(file) != digest:
            raise ValueError('Final adapter files changed after training/merge')
    for name, digest in BASE_WEIGHTS.items():
        if trained.get('base_files_sha256', {}).get(name) != digest:
            raise ValueError('Training did not start from the original baseline weights')
    if not trained.get('initial_adapter_sha256') or not trained.get('final_adapter_sha256') or trained.get('actual_optimizer_steps', 0) <= 0:
        raise ValueError('Training initialization/final adapter/step provenance is missing')
    for rel, digest in artifact.get('artifact_files_sha256', {}).items():
        file = (checkpoint / rel).resolve()
        if not file.is_relative_to(checkpoint.resolve()) or not file.is_file() or sha256_file(file) != digest:
            raise ValueError(f'Merged checkpoint artifact drift: {rel}')
    if not artifact.get('artifact_files_sha256'):
        raise ValueError('Missing merged artifact hash inventory')
    return {'identity': identity, 'initial_adapter_sha256': trained['initial_adapter_sha256'],
        'final_adapter_sha256': trained['final_adapter_sha256'],
        'actual_optimizer_steps': trained['actual_optimizer_steps'],
        'artifact_manifest_sha256': sha256_file(checkpoint / 'artifact_manifest.json'),
        'merge_manifest_sha256': sha256_file(checkpoint.parent / 'merge_manifest.json'),
        'run_manifest_sha256': sha256_file(checkpoint.parent / 'run_manifest.json')}


def validate_training_pair(proofs):
    if set(proofs) != {'plain', 'era'}:
        raise ValueError('Both completed training arms are required for controlled evaluation')
    normalized = []
    for arm in ('plain', 'era'):
        proof = proofs[arm]
        identity = proof.get('identity', {})
        if identity.get('arm') != arm or not all(identity.get(k) for k in ('config_sha256', 'data_sha256', 'source_sha256')):
            raise ValueError('Missing or incorrect training lineage')
        normalized.append({k: v for k, v in identity.items() if k != 'arm'})
    if normalized[0] != normalized[1]:
        raise ValueError('Training arms differ in configuration, data, base or source code')
    if not proofs['plain'].get('initial_adapter_sha256') or proofs['plain']['initial_adapter_sha256'] != proofs['era'].get('initial_adapter_sha256'):
        raise ValueError('Training arms did not use identical LoRA initialization')
    if proofs['plain'].get('actual_optimizer_steps', 0) <= 0 or proofs['plain']['actual_optimizer_steps'] != proofs['era'].get('actual_optimizer_steps'):
        raise ValueError('Training arms have unequal/empty optimizer budgets')
    if not proofs['plain'].get('final_adapter_sha256') or proofs['plain']['final_adapter_sha256'] == proofs['era'].get('final_adapter_sha256'):
        raise ValueError('Two distinct trained adapters are required')


def replay_receipt_path(cfg):
    e = cfg['evaluation']
    return (Path(resolve_path(cfg, e['replay_receipt'])) if e.get('replay_receipt')
            else Path(resolve_path(cfg, e['output_root'])) / 'baseline_replay_receipt.json')


def validate_replay_gate(cfg, source_hashes):
    from .compare import HISTORICAL_RESULTS, compare_repeats
    receipt = read_json(replay_receipt_path(cfg))
    if not receipt.get('passed') or receipt.get('config_sha256') != config_digest(cfg) or receipt.get('source_hashes') != source_hashes:
        raise ValueError('Baseline replay receipt does not match this configuration/source')
    original = Path(resolve_path(cfg, cfg['evaluation']['baseline_runs']['3407']))
    fresh = Path(receipt['fresh_run'])
    if original.resolve() == fresh.resolve() or sha256_file(original / 'results.json') != HISTORICAL_RESULTS[3407]:
        raise ValueError('Baseline replay must compare the audited original against a fresh run')
    fresh_manifest = read_json(fresh / 'manifest.json')
    if fresh_manifest['model_identity'].get('weights_sha256') != BASE_WEIGHTS:
        raise ValueError('Baseline replay checkpoint is not the original baseline')
    check = compare_repeats(original, fresh)
    if not check['passed'] or check['first_artifacts_sha256'] != receipt.get('original_artifacts_sha256') or check['second_artifacts_sha256'] != receipt.get('fresh_artifacts_sha256'):
        raise ValueError('Baseline replay artifacts changed or no longer reproduce')
    return {'passed': True, 'receipt': str(replay_receipt_path(cfg)),
            'receipt_sha256': sha256_file(replay_receipt_path(cfg))}


def build_plan(cfg, arms=('plain', 'era'), include_repeat=True):
    root, protocol, configs, hashes = validate_source(cfg)
    e = cfg['evaluation']
    output = Path(resolve_path(cfg, e['output_root']))
    commands, readiness = [], {}
    for arm in arms:
        if arm not in ('baseline', 'plain', 'era'):
            raise ValueError(f'Unknown evaluation arm: {arm}')
        checkpoint = Path(resolve_path(cfg, cfg['model']['local_path'] if arm == 'baseline' else e['checkpoints'][arm]))
        if checkpoint is not None:
            if checkpoint.is_dir():
                readiness[arm] = {'ready': True, 'checkpoint_hashes': checkpoint_preflight(checkpoint, allow_baseline_snapshot=arm == 'baseline')}
                if arm == 'baseline':
                    if any(readiness[arm]['checkpoint_hashes'].get(k) != v for k, v in BASE_WEIGHTS.items()):
                        raise ValueError('Baseline replay requires the original frozen model weights')
                else:
                    readiness[arm]['training_provenance'] = training_provenance(cfg, arm, checkpoint)
            else:
                readiness[arm] = {'ready': False, 'reason': f'Merged checkpoint missing: {checkpoint}'}
        else:
            readiness[arm] = {'ready': True, 'note': 'Pinned baseline assets must already exist in the frozen cache'}
        runs = [(3407, '_replay' if arm == 'baseline' else '')]
        if arm != 'baseline' and include_repeat:
            runs.append((3407, '_repeat'))
        if arm != 'baseline':
            runs.extend((s, '') for s in SEEDS[1:])
        for seed, suffix in runs:
            name = f'{arm}_seed{seed}{suffix}'
            argv = [e['python'], str(root / 'src/evaluate.py'), '--config',
                str(protocol / f'config.seed{seed}.json'), '--run-name', name,
                '--output-root', str(output)]
            if checkpoint is not None:
                argv.extend(['--checkpoint', str(checkpoint)])
            for option in ('model_cache', 'dataset_cache'):
                if e.get(option):
                    argv.extend(['--' + option.replace('_', '-'), str(resolve_path(cfg, e[option]))])
            commands.append({'arm': arm, 'seed': seed, 'repeat': suffix == '_repeat',
                'argv': argv, 'run_dir': str(output / name)})
    try:
        gate = validate_replay_gate(cfg, hashes)
    except (OSError, ValueError, KeyError) as exc:
        gate = {'passed': False, 'receipt': str(replay_receipt_path(cfg)), 'reason': str(exc),
                'required_command': 'python -m era_lora.evaluate --config CONFIG --arm baseline --execute'}
    return {'schema_version': 1, 'mode': 'plan_only', 'gpu_execution_performed': False,
        'source_hashes': hashes, 'readiness': readiness,
        'baseline_replay_gate': gate,
        'environment_set': {'VLLM_ENABLE_V1_MULTIPROCESSING': '0'},
        'environment_unset': ['VLLM_BATCH_INVARIANT'], 'commands': commands,
        'note': 'Inference is never started without --execute; fresh runs are required for repeatability.'}


def execute_plan(plan, cfg):
    if platform.system() != 'Linux':
        raise ValueError('Baseline GPU evaluation requires the original Linux environment')
    if any(not item['ready'] for item in plan['readiness'].values()):
        raise ValueError('Checkpoint preflight is incomplete')
    candidate = any(c['arm'] != 'baseline' for c in plan['commands'])
    if candidate:
        validate_replay_gate(cfg, plan['source_hashes'])
        proofs = {arm: training_provenance(cfg, arm, resolve_path(cfg, cfg['evaluation']['checkpoints'][arm]))
                  for arm in ('plain', 'era')}
        validate_training_pair(proofs)
    else:
        from .compare import HISTORICAL_RESULTS
        original = Path(resolve_path(cfg, cfg['evaluation']['baseline_runs']['3407']))
        if sha256_file(original / 'results.json') != HISTORICAL_RESULTS[3407]:
            raise ValueError('Original baseline result no longer matches audited artifact')
    for command in plan['commands']:
        if Path(command['run_dir']).exists():
            raise ValueError(f'Fresh evaluation output required: {command["run_dir"]}')
    # Check versions without importing torch/vLLM or initializing CUDA.
    probe = "import sys,json,importlib.metadata as m; print(json.dumps({'python':list(sys.version_info[:2]),'versions':{n:m.version(n).split('+')[0] for n in " + repr(list(VERSIONS)) + "}}))"
    found = json.loads(subprocess.check_output([cfg['evaluation']['python'], '-c', probe], text=True))
    if found != {'python': [3, 12], 'versions': VERSIONS}:
        raise ValueError('Selected Python does not match frozen baseline dependencies')
    env = os.environ.copy()
    env.update(plan['environment_set'])
    for key in plan['environment_unset']:
        env.pop(key, None)
    from .compare import compare_repeats
    for command in plan['commands']:
        subprocess.run(command['argv'], env=env, check=True)
        write_json(Path(command['run_dir']) / 'launcher.json', {
            'argv': command['argv'], 'fresh_output_verified': True,
            'python_probe': found, 'checkpoint_sha256': plan['readiness'][command['arm']]['checkpoint_hashes'],
            'vllm_environment': {k: v for k, v in env.items() if k.startswith('VLLM_')},
            'source_hashes': plan['source_hashes'],
            'training_provenance': plan['readiness'][command['arm']].get('training_provenance')})
        if command['arm'] == 'baseline':
            from .compare import HISTORICAL_RESULTS
            original = Path(resolve_path(cfg, cfg['evaluation']['baseline_runs']['3407']))
            if sha256_file(original / 'results.json') != HISTORICAL_RESULTS[3407]:
                raise ValueError('Original baseline result no longer matches audited artifact')
            check = compare_repeats(original, command['run_dir'])
            receipt = {'passed': check['passed'], 'config_sha256': config_digest(cfg),
                'source_hashes': plan['source_hashes'], 'fresh_run': command['run_dir'],
                'original_artifacts_sha256': check['first_artifacts_sha256'],
                'fresh_artifacts_sha256': check['second_artifacts_sha256'],
                'differences': check['differences']}
            write_json(replay_receipt_path(cfg), receipt)
            if not check['passed']:
                raise ValueError('Original baseline replay failed; candidates remain blocked')
        if command['repeat']:
            first = Path(command['run_dir']).with_name(f'{command["arm"]}_seed3407')
            check = compare_repeats(first, command['run_dir'])
            write_json(Path(command['run_dir']) / 'repeatability.json', check)
            if not check['passed']:
                raise ValueError('Repeatability failed; remaining inference seeds were not run')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', choices=['plain', 'era', 'baseline', 'all'], default='all')
    parser.add_argument('--output', help='Optional plan JSON destination')
    parser.add_argument('--execute', action='store_true', help='Explicitly run GPU inference and grading')
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    arms = ('plain', 'era') if args.arm == 'all' else (args.arm,)
    plan = build_plan(cfg, arms)
    if args.output:
        write_json(Path(args.output), plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.execute:
        execute_plan(plan, cfg)


if __name__ == '__main__':
    main()
