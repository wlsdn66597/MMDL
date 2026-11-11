"""Linux/Python 3.12 bootstrap, pinned downloads, and reusable input checks."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    # A partial last line is an error: never discard completed results silently.
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def config(path=None):
    c = read(path or ROOT / 'src/config.json')
    assert c['model']['repo'] == 'Qwen/Qwen3-VL-4B-Instruct'
    assert c['model']['revision'] == 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
    assert c['dataset']['repo'] == 'MMMU/MMMU'
    assert c['dataset']['revision'] == '98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68'
    assert c['dataset']['split'] == 'validation' and len(set(c['dataset']['configs'])) == 30
    assert c['judge']['repo'] == 'Qwen/Qwen3-8B'
    assert c['judge']['revision'] == 'b968826d9c46dd6066d109eabc6255188de91218'
    assert c['runtime']['dtype'] == c['judge']['runtime']['dtype'] == 'bfloat16'
    assert c['runtime']['quantization'] is None and c['judge']['runtime']['quantization'] is None
    assert c['scoring']['length_policy'] == 'all_incorrect'
    assert c['scoring']['random_fallback'] is False
    return c


def cache_paths(c, model_cache=None, dataset_cache=None):
    return {
        'model': str(Path(model_cache or os.environ.get('MMMU_MODEL_CACHE') or c['cache']['model']).expanduser().resolve()),
        'dataset': str(Path(dataset_cache or os.environ.get('MMMU_DATASET_CACHE') or c['cache']['dataset']).expanduser().resolve())}


def runtime_check(c, gpu=True):
    if platform.system() != 'Linux' or sys.version_info[:2] != (3, 12):
        raise RuntimeError('GPU execution requires Linux x86-64 and Python 3.12; see README.')
    versions = {}
    for package, expected in c['environment']['required_versions'].items():
        found = metadata.version(package)
        if found.split('+')[0] != expected:
            raise RuntimeError(f'{package}: expected {expected}, found {found}. Run prepare.py.')
        versions[package] = found
    import torch
    if gpu and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('A CUDA GPU with BF16 support is required.')
    return {'python':sys.version, 'platform':platform.platform(), 'versions':versions,
            'torch_cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def snapshot(c, cache, judge=False, offline=True):
    from huggingface_hub import snapshot_download
    spec = c['judge'] if judge else c['model']
    path = Path(snapshot_download(spec['repo'], revision=spec['revision'], cache_dir=cache,
        local_files_only=offline, allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja','merges.txt','vocab.json']))
    if path.name != spec['revision']:
        raise RuntimeError('Unexpected snapshot revision.')
    validate_weights(path, 'qwen3' if judge else 'qwen3_vl')
    return path


def validate_weights(path, model_type):
    path = Path(path)
    if (path / 'adapter_config.json').exists():
        raise RuntimeError('Pass a merged BF16 checkpoint, not a standalone LoRA adapter.')
    mc = read(path / 'config.json')
    if mc.get('model_type') != model_type or mc.get('quantization_config'):
        raise RuntimeError('Unexpected architecture or quantized checkpoint.')
    index = path / 'model.safetensors.index.json'
    shards = set(read(index)['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if any(not (path / name).is_file() for name in shards):
        raise RuntimeError('Incomplete weight download.')
    return mc


def load_data(c, cache, offline=True):
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    from datasets import load_dataset
    datasets, rows, ids = {}, [], set()
    for subject in c['dataset']['configs']:
        ds = load_dataset(c['dataset']['repo'], subject, split='validation',
            revision=c['dataset']['revision'], cache_dir=cache)
        if len(ds) != 30 or not ds.cache_files or any(c['dataset']['revision'] not in Path(f['filename']).parts for f in ds.cache_files):
            raise RuntimeError(f'Unverified revision or row count: {subject}')
        datasets[subject] = ds
        text = ds.remove_columns([name for name in ds.column_names if name.startswith('image_')])
        for i, row in enumerate(text):
            if row['id'] in ids or not row['id'].startswith('validation_' + subject + '_'):
                raise RuntimeError('Duplicate or unexpected question ID.')
            ids.add(row['id'])
            rows.append(dict(row, subject=subject, dataset_index=i))
    if len(rows) != 900:
        raise RuntimeError('Expected 900 questions.')
    return datasets, rows


def prepare_assets(c, paths, offline=False):
    models = {name:str(snapshot(c, paths['model'], judge=name == 'judge', offline=offline)) for name in ('model','judge')}
    _, rows = load_data(c, paths['dataset'], offline=offline)
    # Content hashes are recorded without writing or bundling image files.
    hashes = {name:{p.name:digest(p) for p in Path(model).glob('*.safetensors')} for name,model in models.items()}
    result = {'created_at':now(), 'model_snapshots':models, 'model_sha256':hashes,
              'dataset_revision':c['dataset']['revision'], 'questions':len(rows), 'cache_paths':paths}
    write(ROOT / 'outputs/preparation.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'src/config.json'))
    p.add_argument('--model-cache')
    p.add_argument('--dataset-cache')
    p.add_argument('--existing-env', action='store_true', help='Validate this interpreter; do not create/install another environment.')
    p.add_argument('--offline', action='store_true', help='Validate/reuse cached assets only.')
    p.add_argument('--check-only', action='store_true', help='Check versions/GPU; do not download assets.')
    args = p.parse_args()
    c = config(args.config)
    if not args.existing_env:
        if platform.system() != 'Linux' or sys.version_info[:2] != (3,12):
            p.error('Preparation requires Linux and Python 3.12.')
        env = ROOT / '.venv'
        if not (env / 'bin/python').exists():
            venv.EnvBuilder(with_pip=True).create(env)
        python = str(env / 'bin/python')
        subprocess.run([python,'-m','pip','install','-r',str(ROOT / 'requirements.txt')], check=True)
        command = [python,str(Path(__file__).resolve()),'--existing-env','--config',str(Path(args.config).resolve())]
        for name in ('model_cache','dataset_cache'):
            if getattr(args,name):
                command += ['--' + name.replace('_','-'), getattr(args,name)]
        command += [flag for flag in ('--offline','--check-only') if getattr(args,flag[2:].replace('-','_'))]
        subprocess.run(command, check=True)
        return
    environment = runtime_check(c)
    print(json.dumps(environment, ensure_ascii=False, indent=2))
    write(ROOT / 'outputs/environment.json', environment)
    if not args.check_only:
        prepare_assets(c, cache_paths(c,args.model_cache,args.dataset_cache), args.offline)


if __name__ == '__main__':
    main()
