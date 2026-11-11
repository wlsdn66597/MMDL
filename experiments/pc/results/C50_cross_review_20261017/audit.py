import ast, csv, hashlib, itertools, json, math, re, statistics, string, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
ROOT=Path('C:/project/MMDL'); OUT=Path(__file__).resolve().parent; SRC=OUT/'source_C'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def write(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(s):return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
def info(path):
    with Image.open(path) as im:
        im.load(); rgb=im.convert('RGB'); sm=list(rgb.convert('L').resize((9,8),Image.Resampling.BICUBIC).getdata()); h=0
        for y in range(8):
            for x in range(8):h=(h<<1)|int(sm[y*9+x]>sm[y*9+x+1])
        return {'file_sha256':sha(path),'size':list(rgb.size),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'dhash':h,'format':im.format}
C=rows(SRC/'paired_train.draft.jsonl'); D=rows(ROOT/'results/MMDL_ERA_Dataset_D50_package/paired_train.draft.jsonl')
source_path=ROOT/'reports/era_dataset_examples_20261014/scienceqa_problems.json'; source=read(source_path)
splits=read(ROOT/'reports/era_dataset_examples_20261014/scienceqa_splits.json')
assignment={r['source_id']:r for r in rows(ROOT/'reports/era_image_assignment_20261015/source_manifest.jsonl') if r['source_dataset']=='ScienceQA'}
sys.path.insert(0,str(ROOT/'.runtime/tokenizers'))
from tokenizers import Tokenizer
tokenizer_path=ROOT/'.runtime/qwen_tokenizer/tokenizer.json';tok=Tokenizer.from_file(str(tokenizer_path))
tree=ast.parse((ROOT/'code/evaluation/mmdl-visual-token-analysis/src/evaluate.py').read_text(encoding='utf-8'))
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prompt_for');ns={'ast':ast,'math':math,'string':string}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'baseline_prompt','exec'),ns)
config=read(ROOT/'results/era_D_50_20261016/provenance/baseline_config.json')
blind={r[0]:r for r in read(OUT/'blind_observations.json')['rows']}
images={}; checks=[]
for p in C:
    s=source[p['source_id']]; a=assignment[p['source_id']]; options=list(p['options'].values()); im=info(SRC/'images'/f"{p['source_id']}.png");images[p['sample_id']]=im
    checks_source={'question':p['question']==s['question'],'hint':p['hint']==s['hint'],'options_and_order':options==s['choices'],'answer_letter':p['answer_letter']==string.ascii_uppercase[s['answer']],'answer_value':p['answer_value']==s['choices'][s['answer']],'train_split':p['source_id'] in splits['train'] and p['source_split']=='train','revision':p['source_revision']==a['source_revision'],'assignment':p['image_group_no']==a['image_group_no'] and p['image_group_no']%2==1 and p['owner']=='C' and p['reviewer']=='D','image_declared_hash':im['file_sha256']==p['image_sha256']}
    canonical=ns['prompt_for'](dict(question=p['question'],hint=p['hint'],options=options,question_type='multiple-choice'),config)
    r={'sample_id':p['sample_id'],'source_checks':checks_source,'source_all_pass':all(checks_source.values()),'declared_path_exists':(SRC/p['image_path']).exists(),'resolved_path':f"images/{p['source_id']}.png",'baseline_prompt_exact':p['input_prompt']==canonical,'baseline_only_empty_hint_difference':not p['hint'] and p['input_prompt']==canonical.removeprefix('Hint: \n'),'blind_answer':blind[p['source_id']][1],'blind_answer_matches':blind[p['source_id']][1]==p['answer_letter'],'response_checks':{},'tokens':{}}
    for k in ['general','era']:
        t=p['response_'+k];body=t.rsplit('\nFinal answer:',1)[0]
        if k=='era':body=re.sub(r'(?m)^(Evidence|Rule|Application):\s*','',body)
        r['response_checks'][k]={'final_answer':t.splitlines()[-1]=='Final answer: '+p['answer_letter'],'one_final_answer':len(re.findall(r'(?m)^Final answer:',t))==1,'no_korean':not re.search('[가-힣]',t),'labels':re.findall(r'(?m)^(Evidence|Rule|Application):',t)==(['Evidence','Rule','Application'] if k=='era' else [])}
        r['tokens'][k+'_total']=len(tok.encode(t,add_special_tokens=False).ids);r['tokens'][k+'_body']=len(tok.encode(body,add_special_tokens=False).ids)
    ll=[r['tokens']['general_body'],r['tokens']['era_body']];r['tokens']['body_relative_difference']=(max(ll)-min(ll))/min(ll)
    r['tokens']['over512']=max(r['tokens']['general_total'],r['tokens']['era_total'])>512
    checks.append(r)
for p in D:images[p['sample_id']]=info(ROOT/'results/MMDL_ERA_Dataset_D50_package'/p['image_path'])
exact_c=[];exact_cd=[];near=[]
for left,right in itertools.combinations(C+D,2):
    if left in D and right in D:continue
    a=images[left['sample_id']];b=images[right['sample_id']]
    rec={'left':left['sample_id'],'right':right['sample_id'],'dhash_distance':(a['dhash']^b['dhash']).bit_count(),'same_bytes':a['file_sha256']==b['file_sha256']}
    if (a['size'],a['rgb_sha256'])==(b['size'],b['rgb_sha256']):
        (exact_c if left['owner']==right['owner'] else exact_cd).append(rec)
    elif rec['dhash_distance']<=4:near.append(rec)
# A low dHash distance in maps can just reflect a shared background; it is not a duplicate verdict.
groups=defaultdict(list)
for p in C:groups[(tuple(images[p['sample_id']]['size']),images[p['sample_id']]['rgb_sha256'])].append(p['sample_id'])
baseline=ROOT/'results/mmmu_first_baseline_20261009';ref=read(baseline/'baseline_seed3407/results.json')
mm={'dataset':ref['manifest']['configuration']['dataset'],'split':'validation','questions':len(ref['records']),'images':0,'missing':[],'exact_text':[],'near_text':[],'exact_images':[],'near_images':[],'reference_results_sha256':sha(baseline/'baseline_seed3407/results.json')}
for r in ref['records']:
    words=set(norm(r['question']).split())
    for p in C:
        qw=set(norm(p['question']).split());j=len(words&qw)/len(words|qw) if words|qw else 0
        if norm(r['question'])==norm(p['question']):mm['exact_text'].append([p['sample_id'],r['id']])
        if j>=.8:mm['near_text'].append([p['sample_id'],r['id'],j])
    for field in r['image_fields']:
        path=baseline/'analysis/images'/f"{r['id']}_{field}.png"
        if not path.exists():mm['missing'].append(str(path));continue
        a=info(path);mm['images']+=1
        for p in C:
            b=images[p['sample_id']];dist=(a['dhash']^b['dhash']).bit_count()
            if (a['size'],a['rgb_sha256'])==(b['size'],b['rgb_sha256']):mm['exact_images'].append([p['sample_id'],path.name])
            if dist<=4:mm['near_images'].append([p['sample_id'],path.name,dist])
logs=list(csv.DictReader((SRC/'image_log.csv').open(encoding='utf-8-sig',newline='')))
completed=[r for r in logs if r['상태']=='작성완료']
stats={'rows':len(C),'unique_ids':len(set(p['source_id'] for p in C)),'unique_pixel_images':len(groups),'source_all_pass':sum(r['source_all_pass'] for r in checks),'blind_answer_matches':sum(r['blind_answer_matches'] for r in checks),'path_errors':sum(not r['declared_path_exists'] for r in checks),'baseline_prompt_exact':sum(r['baseline_prompt_exact'] for r in checks),'empty_hint_differences':sum(r['baseline_only_empty_hint_difference'] for r in checks),'token_length_flags':[r['sample_id'] for r in checks if r['tokens']['body_relative_difference']>.15],'over512':[r['sample_id'] for r in checks if r['tokens']['over512']],'max_total_tokens':max(r['tokens'][k+'_total'] for r in checks for k in ['general','era']),'image_log':{'total':len(logs),'statuses':dict(Counter(r['상태'] for r in logs)),'completed_ids_match':set(r['원본문항ID'] for r in completed)==set(p['source_id'] for p in C)}}
report={'summary':stats,'source_metadata_path':str(source_path),'source_metadata_sha256':sha(source_path),'tokenizer_sha256':sha(tokenizer_path),'tokenizer_revision':'ebb281ec70b05090aa6165b016eac8ec08e71b17','checks':checks,'C_exact_pixel_groups':[g for g in groups.values() if len(g)>1],'C_exact_pixel_pairs':exact_c,'CD_exact_pixel_pairs':exact_cd,'CD_source_id_overlap':sorted(set(p['source_id'] for p in C)&set(p['source_id'] for p in D)),'near_pairs_dhash_le4':near,'mmmu_independent_rerun':mm,'mmmu_author_claim_not_independently_rerun':{'dev':150,'test':10500,'file':'source_C/mmmu_overlap.json'},'not_covered':['A/B full submissions unavailable locally','separate team development 200 and final evaluation 300 lists not supplied','official ScienceQA image archive not available for independent byte comparison','human cross-review final approval pending'],'image_details':images}
write(OUT/'automated_audit.json',report)
print(json.dumps({'summary':stats,'C_groups':report['C_exact_pixel_groups'],'CD_exact':exact_cd,'near_pairs':len(near),'MMMU':mm},ensure_ascii=False,indent=2))
# Build compact, full solution review batches only after blind observations were saved.
for start in range(0,50,10):
    parts=[]
    for p in C[start:start+10]:
        s=source[p['source_id']]
        parts.append(f"\n{p['sample_id']} | answer {p['answer_letter']} = {p['answer_value']}\nERA: {p['response_era']}\nGENERAL: {p['response_general']}\nCOMMON: {json.dumps(p['common_content'],ensure_ascii=False)}\nSOURCE SOLUTION: {s['solution']}\nSOURCE LECTURE: {s['lecture']}")
    (OUT/f'solutions_{start+1:02d}_{start+10:02d}.txt').write_text('\n'.join(parts),encoding='utf-8')
