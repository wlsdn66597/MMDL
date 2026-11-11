import os,sys,json,time,hashlib,importlib.util,fcntl,subprocess,traceback
from pathlib import Path
from collections import defaultdict,Counter
ROOT=Path('/home/rlatjswo/MMDL')
DATA=Path('/home/rlatjswo/mmdl-data/mmmu_val')
EXP=ROOT/'logs/prompt_compare/B2_final_line_90_20260923'
OUT=ROOT/'results/baseline/B2_final_line_90_20260923'
B0=ROOT/'results/baseline/B0_full'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):
 p=EXP/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
def b0hash():return {str(p.relative_to(B0)):sha(p) for p in B0.rglob('*') if p.is_file()}
os.chdir(ROOT)
lock=open(ROOT/'logs/prompt_compare/evaluation.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
source=ROOT/'baseline_tools/baseline_eval.py'
assert sha(source)=='5677e27c29b54b0310b8c4e23d285499d339f4fa0ee8c9734ea1d12933462b21','B0 code changed'
assert not OUT.exists(),'Output exists'
spec=importlib.util.spec_from_file_location('b0_eval',source);b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
MC=b.MC_INSTRUCTION + "\n" + "End your response with a separate final line that starts with \"Final Answer: \" followed by only your chosen uppercase option letter in parentheses. Do not write anything after that line."
OPEN=b.OPEN_INSTRUCTION
manifest,all_samples=b.load_prepared(DATA)
groups=defaultdict(list)
for s in all_samples:groups[s['subject']].append(s)
key=lambda s:hashlib.sha256(('prompt-pilot-v1|'+s['id']).encode()).hexdigest()
selected=[s for subject in sorted(groups) for s in sorted(groups[subject],key=key)[:3]]
selected=sorted(selected,key=lambda s:(s['subject'],int(s['id'].rsplit('_',1)[1])))
assert len(selected)==90 and len(groups)==30
prior=json.loads((ROOT/'logs/prompt_compare/B1_answer_only_90_20260923/experiment.json').read_text())
assert [s['id'] for s in selected]==prior['selected_ids'], 'Pilot IDs differ'
before=b0hash()
record=json.loads((ROOT/'logs/baseline_server.json').read_text())
baseline_config=json.loads((B0/'config.json').read_text())
assert record['settings']==baseline_config['identity']['server_settings'], 'Server settings differ from B0'
assert record['model_hashes']==baseline_config['identity']['model_hashes'], 'Model identity differs from B0'
model=Path(record['model_path'])
for name,meta in record['model_hashes'].items():assert sha(model/name)==meta['sha256'], 'model hash mismatch '+name
api=b.LocalAPI('http://127.0.0.1:8000',300)
api.request('/health')
metrics=api.opener.open('http://127.0.0.1:8000/metrics',timeout=10).read().decode()
for line in metrics.splitlines():
 if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')):assert float(line.rsplit(' ',1)[1])==0,'Server busy'
original_prompt=b.prompt_text
b.MC_INSTRUCTION=MC;b.OPEN_INSTRUCTION=OPEN;b.PROMPT_VERSION='B2_final_line_contract_v1'
b.choose_samples=lambda samples,args:selected
save('experiment.json',{'status':'preflight','pid':os.getpid(),'started_at':b.utc(),'selected_ids':[s['id'] for s in selected],'selection':'3 per subject: ascending SHA256(prompt-pilot-v1|ID), independent of answer/correctness','types':dict(Counter(s['question_type'] for s in selected)),'mc_instruction':MC,'open_instruction':OPEN,'sampling':b.SAMPLING,'server_record_sha256':sha(ROOT/'logs/baseline_server.json'),'base_script_sha256':sha(source),'wrapper_sha256':sha(Path(__file__)),'b0_hashes_before':before,'model_file_hashes_verified':True,'caveats':['single seed exploratory pilot','Only MC final-line format instruction added; open prompt unchanged; no explicit reasoning-length change','B0 selected scores retained from original 900 scoring; subset fallback RNG may differ if B2 falls back']})
save('state.json',{'status':'running','pid':os.getpid(),'out':str(OUT),'started_at':b.utc()})
exit_code=0
try:
 import argparse
 b.run(argparse.Namespace(data_path=str(DATA),out=str(OUT),server_config=str(ROOT/'logs/baseline_server.json'),base_url='http://127.0.0.1:8000',timeout=300,subject=None,smoke=0,limit=0,resume=False,preflight_only=False))
except BaseException:
 exit_code=1;traceback.print_exc()
finally:
 after=b0hash();save('b0_integrity.json',{'unchanged':before==after,'before':before,'after':after})
 if before!=after:exit_code=2
 save('state.json',{'status':'completed' if exit_code==0 else 'failed','pid':os.getpid(),'exit_code':exit_code,'ended_at':b.utc(),'out':str(OUT),'b0_unchanged':before==after})
sys.exit(exit_code)
