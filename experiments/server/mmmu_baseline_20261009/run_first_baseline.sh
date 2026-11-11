#!/usr/bin/env bash
set -Eeuo pipefail
export EVAL_REPO=/home/rlatjswo/mmdl_test
export CAMPAIGN_ROOT=/home/rlatjswo/mmmu_baseline_20261009
export PROTOCOL_DIR="$CAMPAIGN_ROOT/protocol"
export HF_HOME=/home/rlatjswo/mmdl-cache/huggingface
export MMMU_MODEL_CACHE="$HF_HOME/hub"
export MMMU_DATASET_CACHE="$HF_HOME/datasets"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
unset VLLM_BATCH_INVARIANT HF_HUB_OFFLINE HF_DATASETS_OFFLINE
BOOT_PY=/home/rlatjswo/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
STARTED=$(date +%s)
EVAL_STARTED=0
phase=preflight
exec >> "$CAMPAIGN_ROOT/driver.log" 2>&1
mark() {
  "$BOOT_PY" - "$CAMPAIGN_ROOT" "$1" "$phase" "$2" "$STARTED" "$EVAL_STARTED" "$$" <<'PY'
import datetime, json, pathlib, sys, time
root, state, phase, code, started, eval_started, pid = sys.argv[1:]
now = time.time()
record = {'state': state, 'phase': phase, 'exit_code': int(code), 'driver_pid': int(pid),
          'started_at_utc': datetime.datetime.fromtimestamp(int(started), datetime.timezone.utc).isoformat(),
          'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'total_elapsed_seconds': now-int(started),
          'evaluation_started_at_unix': int(eval_started) or None,
          'evaluation_elapsed_seconds': now-int(eval_started) if int(eval_started) else None,
          'run_name': 'baseline_seed3407', 'full_evaluation_runs_requested': 1}
p = pathlib.Path(root) / 'status.json'
tmp = p.with_suffix('.tmp')
tmp.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
tmp.replace(p)
print(json.dumps(record), flush=True)
PY
}
trap 'rc=$?; if [ "$rc" -eq 0 ]; then mark complete 0; else mark failed "$rc"; fi' EXIT
mark running 0
cd "$EVAL_REPO"
test ! -e "$CAMPAIGN_ROOT/baseline_seed3407"
"$BOOT_PY" - <<'PY'
import hashlib, json, os
from pathlib import Path
repo=Path(os.environ['EVAL_REPO']); bundle=Path(os.environ['PROTOCOL_DIR'])
c=json.loads((bundle/'evaluation_contract.json').read_text())
linux_hashes=json.loads((bundle/'linux_source_hashes.json').read_text())['sha256']
for name, expected in c['source_sha256'].items():
    if name.startswith('src/') or name=='requirements.txt':
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==linux_hashes[name], name
assert hashlib.sha256((bundle/'config.seed3407.json').read_bytes()).hexdigest()==c['config_sha256']['config.seed3407.json']
assert hashlib.sha256((bundle/'evaluation_items.json').read_bytes()).hexdigest()==c['evaluation_items_sha256']
print('PASS: frozen source/configuration/item hashes')
PY
git rev-parse HEAD > "$CAMPAIGN_ROOT/provenance/git-commit.txt"
git status --porcelain > "$CAMPAIGN_ROOT/provenance/git-status.txt"
nvidia-smi -q > "$CAMPAIGN_ROOT/provenance/nvidia-smi.txt"
tar -czf "$CAMPAIGN_ROOT/provenance/source.tar.gz" src requirements.txt README.md THIRD_PARTY_LICENSES.txt
phase=prepare
mark running 0
"$BOOT_PY" -m venv .venv
/home/rlatjswo/.local/bin/uv pip install --python .venv/bin/python \
  --cache-dir /home/rlatjswo/mmdl-cache/uv -r requirements.txt \
  > "$CAMPAIGN_ROOT/install-uv.log" 2>&1
.venv/bin/python src/prepare.py --existing-env --config "$PROTOCOL_DIR/config.seed3407.json" \
  --model-cache "$MMMU_MODEL_CACHE" --dataset-cache "$MMMU_DATASET_CACHE" \
  > "$CAMPAIGN_ROOT/prepare.log" 2>&1
cp outputs/preparation.json "$CAMPAIGN_ROOT/provenance/preparation.json"
cp outputs/environment.json "$CAMPAIGN_ROOT/provenance/environment-check.json"
.venv/bin/python -m pip freeze > "$CAMPAIGN_ROOT/provenance/pip-freeze.txt"
.venv/bin/python src/evaluate.py --config "$PROTOCOL_DIR/config.seed3407.json" --selfcheck
phase=verify_inputs
mark running 0
.venv/bin/python - <<'PY'
import hashlib, json, os, platform, sys
from pathlib import Path
import torch
sys.path.insert(0, 'src')
import prepare, evaluate
root=Path(os.environ['CAMPAIGN_ROOT']); bundle=Path(os.environ['PROTOCOL_DIR'])
c=prepare.config(bundle/'config.seed3407.json')
datasets, rows=prepare.load_data(c, os.environ['MMMU_DATASET_CACHE'], offline=True)
expected={r['id']: r for r in json.loads((bundle/'evaluation_items.json').read_text())}
image_records=[]
for row in rows:
    r=evaluate.data_row(row)
    source=datasets[row['subject']][row['dataset_index']]
    fields=[f'image_{i}' for i in range(1,8) if source.get(f'image_{i}') is not None]
    assert 1 <= len(fields) <= c['runtime']['max_images_per_prompt']
    r.update(input_prompt=evaluate.prompt_for(row,c), image_fields=fields)
    canonical={k:r[k] for k in ('question','options','reference_answer','input_prompt','image_fields')}
    actual=hashlib.sha256(json.dumps(canonical,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    assert actual==expected[r['id']]['input_sha256'], r['id']
    for field in fields:
        rgb=source[field].convert('RGB')
        image_records.append({'id':r['id'],'field':field,'size_wh':list(rgb.size),
                              'rgb_bytes_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
assert len(rows)==len(expected)==900
(root/'provenance/input-validation.json').write_text(json.dumps({'records':900,'all_frozen_inputs_match':True,'images':image_records},indent=2))
paths=prepare.read(root/'provenance/preparation.json')['model_snapshots']
aux={}
for name, folder in paths.items():
    aux[name]={p.name:prepare.digest(p) for p in Path(folder).iterdir()
               if p.is_file() and p.suffix in ('.json','.txt','.model','.jinja')}
(root/'provenance/model-auxiliary-hashes.json').write_text(json.dumps(aux,indent=2))
keys=sorted(k for k in os.environ if k.startswith('VLLM_') or k in ('CUDA_VISIBLE_DEVICES','HF_HOME','TOKENIZERS_PARALLELISM'))
env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,
     'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
     'environment':{k:os.environ[k] for k in keys}}
(root/'provenance/environment.json').write_text(json.dumps(env,indent=2))
print('PASS: 900 input records match; image/processor/tokenizer hashes recorded')
PY
phase=evaluate
EVAL_STARTED=$(date +%s)
mark running 0
BASELINE_CHECKPOINT="$MMMU_MODEL_CACHE/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
.venv/bin/python src/evaluate.py --config "$PROTOCOL_DIR/config.seed3407.json" \
  --checkpoint "$BASELINE_CHECKPOINT" --run-name baseline_seed3407 \
  --output-root "$CAMPAIGN_ROOT" --model-cache "$MMMU_MODEL_CACHE" \
  --dataset-cache "$MMMU_DATASET_CACHE" > "$CAMPAIGN_ROOT/evaluation.log" 2>&1
phase=completed
