from pathlib import Path
import sys,json,hashlib,io,urllib.request,importlib.util
ROOT=Path('C:/project/MMDL');TMP=ROOT/'tmp/era_pilot_20261027';TMP.mkdir(exist_ok=True,parents=True)
sys.path.insert(0,str(ROOT/'.runtime/era_review_deps'))
import pyarrow.parquet as pq
from PIL import Image
spec=importlib.util.spec_from_file_location('selector',ROOT/'reports/era_selection_kit_20261015/select_questions.py')
sel=importlib.util.module_from_spec(spec);spec.loader.exec_module(sel)
science=json.loads((ROOT/'reports/era_dataset_examples_20261014/scienceqa_problems.json').read_text(encoding='utf-8'))
train=json.loads((ROOT/'reports/era_dataset_examples_20261014/scienceqa_splits.json').read_text(encoding='utf-8'))['train']
cache=ROOT/'tmp/era_prereview_20261021/scienceqa_train.parquet'
assert hashlib.file_digest(cache.open('rb'),'sha256').hexdigest()=='62c90a28e3fb1bc0ad7bbcab1ac62b483ae6758291a655944d8f494bf6445745'
table=pq.read_table(cache)
out=[]
for owner,lo,hi in [('A',105,105),('B',106,106),('C',195,223),('D',198,202)]:
 rows,fp=sel.load_sources(ROOT/'reports/era_dataset_examples_20261014',owner)
 for n,path,batch in sel.grouped(rows):
  if not lo<=n<=hi or sel.assigned_owner(batch[0]['dataset'],n)!=owner:continue
  for r in batch:
   r=dict(r,owner=owner,image_group_no=n,source_fingerprint=fp,selection_version=sel.VERSION)
   sid=r['source_id'];p=TMP/'candidate_images'/owner/Path(path).name
   if owner in 'CD':p=TMP/'candidate_images'/owner/f'{sid}.png'
   p.parent.mkdir(exist_ok=True,parents=True)
   if owner in 'CD':
    row_index=[str(x) for x in train].index(sid);mirror=table.slice(row_index,1).to_pylist()[0]
    assert all(mirror[k]==science[sid][k] for k in ['question','choices','answer','hint','lecture','solution','task','grade','subject','topic','category','skill'])
    p.write_bytes(mirror['image']['bytes'])
    r['asset_provenance']={'repository':'derek-thomas/ScienceQA','revision':'f18b0a70359ebfb41f658fd564208d0355b013f4','parquet_sha256':'62c90a28e3fb1bc0ad7bbcab1ac62b483ae6758291a655944d8f494bf6445745','row_index':row_index,'official_metadata_exact_match':True,'bytes':'original embedded PNG, no resizing'}
   else:
    url='https://raw.githubusercontent.com/vis-nlp/ChartQA/044eabfc306abfe9340c5741f0093aefc5973d06/ChartQA%20Dataset/'+path
    if not p.exists():
     with urllib.request.urlopen(url,timeout=45) as f:p.write_bytes(f.read())
    r['asset_provenance']={'url':url,'revision':'044eabfc306abfe9340c5741f0093aefc5973d06','bytes':'original downloaded PNG, no resizing'}
   r['local_image']=str(p);r['image_sha256']=hashlib.sha256(p.read_bytes()).hexdigest();r['image_size']=Image.open(p).size
   out.append(r)
(TMP/'candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{'id':r['source_id'],'owner':r['owner'],'group':r['image_group_no'],'question':r['question'],'options':r['original_options']} for r in out],ensure_ascii=False))
