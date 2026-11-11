"""CPU-only measurement with the student's pinned Qwen tokenizer."""
import json, hashlib, re
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
REV = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
MODEL = Path('/home/rlatjswo/mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots')/REV
tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
rows = [json.loads(x) for x in (ROOT/'paired_train.draft.jsonl').read_text(encoding='utf-8').splitlines()]
counts=[]
for row in rows:
    entry={'sample_id':row['sample_id']}
    for condition in ['general','era']:
        response=row['response_'+condition]
        body=response.rsplit('\nFinal answer:',1)[0]
        if condition=='era': body=re.sub(r'(?m)^(Evidence|Rule|Application):\s*','',body)
        entry[condition+'_total']=len(tok.encode(response,add_special_tokens=False))
        entry[condition+'_body']=len(tok.encode(body,add_special_tokens=False))
    lengths=[entry['general_body'],entry['era_body']]
    entry['body_relative_difference']=(max(lengths)-min(lengths))/min(lengths)
    entry['review_length_difference']=entry['body_relative_difference']>0.15
    entry['review_over_512']=max(entry['general_total'],entry['era_total'])>512
    counts.append(entry)
report={'model_id':'Qwen/Qwen3-VL-4B-Instruct','revision':REV,'tokenizer_path':str(MODEL),
        'tokenizer_sha256':hashlib.sha256((MODEL/'tokenizer.json').read_bytes()).hexdigest(),
        'add_special_tokens':False,'body_definition':'Remove ERA labels and final-answer line; preserve remaining wording and paragraph breaks.',
        'counts':counts}
(ROOT/'token_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
