import json
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path('/home/rlatjswo/mmmu_baseline_20261009')
config = json.loads((root / 'protocol/config.seed3407.json').read_text())
spec = config['judge']
print('Downloading pinned judge:', spec['repo'], spec['revision'], flush=True)
path = snapshot_download(
    spec['repo'], revision=spec['revision'],
    cache_dir='/home/rlatjswo/mmdl-cache/huggingface/hub',
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'merges.txt', 'vocab.json'])
print('Completed:', path, flush=True)
