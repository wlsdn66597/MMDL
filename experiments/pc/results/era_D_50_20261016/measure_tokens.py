"""CPU-only measurement with the student's pinned Qwen tokenizer."""
import json, hashlib, re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'.runtime/tokenizers'))
from tokenizers import Tokenizer
import tokenizers

ROOT = Path(__file__).resolve().parent
REV = 'ebb281ec70b05090aa6165b016eac8ec08e71b17'
MODEL = ROOT.parents[1]/'.runtime/qwen_tokenizer'
assert hashlib.sha256((MODEL/'tokenizer.json').read_bytes()).hexdigest()=='a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7'
tok = Tokenizer.from_file(str(MODEL/'tokenizer.json'))
rows = [json.loads(x) for x in (ROOT/'paired_train.draft.jsonl').read_text(encoding='utf-8').splitlines()]
counts=[]
for row in rows:
    entry={'sample_id':row['sample_id']}
    for condition in ['general','era']:
        response=row['response_'+condition]
        body=response.rsplit('\nFinal answer:',1)[0]
        if condition=='era': body=re.sub(r'(?m)^(Evidence|Rule|Application):\s*','',body)
        entry[condition+'_total']=len(tok.encode(response,add_special_tokens=False).ids)
        entry[condition+'_body']=len(tok.encode(body,add_special_tokens=False).ids)
    lengths=[entry['general_body'],entry['era_body']]
    entry['body_relative_difference']=(max(lengths)-min(lengths))/min(lengths)
    entry['review_length_difference']=entry['body_relative_difference']>0.15
    entry['review_over_512']=max(entry['general_total'],entry['era_total'])>512
    counts.append(entry)
report={'model_id':'Qwen/Qwen3-VL-4B-Instruct','revision':REV,'tokenizer_path':str(MODEL),
        'tokenizer_sha256':hashlib.sha256((MODEL/'tokenizer.json').read_bytes()).hexdigest(),
        'runtime':'tokenizers '+tokenizers.__version__, 'execution_location':'local', 'add_special_tokens':False,'body_definition':'Remove ERA labels and final-answer line; preserve remaining wording and paragraph breaks.',
        'paired_file_sha256':hashlib.sha256((ROOT/'paired_train.draft.jsonl').read_bytes()).hexdigest(), 'counts':counts}
old=json.loads((ROOT.parent/'era_D_first5_20261015/token_report.json').read_text(encoding='utf-8'))
assert counts[:5]==old['counts'], 'Local tokenizer counts differ from prior Transformers result'
report['first5_counts_equal_prior_transformers']=True
(ROOT/'token_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'count':len(counts),'first5_counts_match':True,'length_flags':[x for x in counts if x['review_length_difference']],'over512':[x for x in counts if x['review_over_512']]}),flush=True)
