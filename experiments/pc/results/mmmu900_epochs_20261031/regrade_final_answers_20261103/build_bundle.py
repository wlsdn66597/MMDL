"""Package immutable saved responses and the isolated repair runner for the same server."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
CODE=ROOT/'code/training/era_decoder_lora'
EVIDENCE=ROOT/'results/mmmu900_epochs_20261031/evidence'
files=[CODE/'tools/regrade_final_answers.py',CODE/'tests/test_regrade_final_answers.py']
files += [p for p in (CODE/'baseline/src').iterdir() if p.suffix in ('.py','.json')]
files += list((CODE/'baseline').rglob('*LICENSE*'))
for folder in ('historical_baseline','epoch_2_seed3407','epoch_3_seed3407','epoch_4_seed3407'):
    files += [EVIDENCE/folder/name for name in ('results.json','manifest.json','judge_attempts.jsonl')]
files=sorted({p for p in files if p.is_file()})
manifest={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
instructions='''MMMU saved-response regrade: isolated bundle

Extract into a NEW empty directory on the original evaluation server. Keep the
code/ and results/ directory layout. Do not extract over the original workspace.
The model weights and environment are not bundled; use the original pinned ones.

From the extracted root, first run the offline replay and plan:
/home/rlatjswo/mmdl_test/.venv/bin/python code/training/era_decoder_lora/tools/regrade_final_answers.py

Then run the pending Qwen3-8B calls:
/home/rlatjswo/mmdl_test/.venv/bin/python code/training/era_decoder_lora/tools/regrade_final_answers.py --execute

Results are written under results/mmmu900_epochs_20261031/regrade_final_answers_20261103/.
summary.json must have status=complete before reporting a new accuracy.
No student inference or training is performed. Original gold labels and saved
responses remain unchanged. The original protocol scores remain the primary
comparison; the repaired protocol is a separate uniformly applied comparison.

The new inputs must be judged: do not reuse an old judge verdict whose input hash
differs. Execution uses the pinned Qwen3-8B with unchanged matching prompt and
sampling configuration. It may still make semantic/numeric errors; inspect the
known errors and resolved_changes.json after completion rather than forcing
individual answers to agree with a prior audit.
'''
archive=OUT/'qwen_regrade_bundle.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:z.write(p,p.relative_to(ROOT))
    z.writestr('RUN_INSTRUCTIONS.txt',instructions)
    z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for name,expected in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
record={'files':len(files),'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'all_archived_sha256_verified':True,'new_judge_executed':False,'server_status':'off, as reported by user',
        'manifest':manifest}
(OUT/'bundle_validation.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
(OUT/'RUN_INSTRUCTIONS.txt').write_text(instructions,encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='manifest'},indent=2))
