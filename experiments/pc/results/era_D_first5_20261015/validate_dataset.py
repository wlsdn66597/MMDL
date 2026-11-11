"""Validate the paired draft against locked originals and the existing MMMU baseline."""
import ast, hashlib, json, math, re, string, unicodedata, statistics
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def normalize(s): return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
def image_info(p):
    with Image.open(p) as im:
        rgb=im.convert('RGB')
        pixel=hashlib.sha256(rgb.tobytes()).hexdigest()
        small=list(rgb.convert('L').resize((9,8)).get_flattened_data())
        dh=0
        for y in range(8):
            for x in range(8): dh=(dh<<1)|int(small[y*9+x]>small[y*9+x+1])
        return tuple(rgb.size),pixel,dh

paired=rows(ROOT/'paired_train.draft.jsonl')
plain=rows(ROOT/'train_plain.draft.jsonl')
era=rows(ROOT/'train_era.draft.jsonl')
source=read(PROJECT/'reports/era_dataset_examples_20261014/scienceqa_problems.json')
splits=read(PROJECT/'reports/era_dataset_examples_20261014/scienceqa_splits.json')
config=read(ROOT/'provenance/baseline_config.json')
# Execute the exact existing prompt function, without importing GPU/evaluation dependencies.
tree=ast.parse((PROJECT/'code/evaluation/mmdl-visual-token-analysis/src/evaluate.py').read_text(encoding='utf-8'))
fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='prompt_for')
ns={'ast':ast,'string':string,'math':math}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'existing_evaluate_prompt_for','exec'),ns)
assert len(paired)==len(plain)==len(era)==5
assert len({r['source_id'] for r in paired})==5
assert len({r['image_group_no'] for r in paired})==5
checks=[]
images=[]
for p,g,e in zip(paired,plain,era):
    s=source[p['source_id']]
    assert p['owner']=='D' and p['image_group_no']%2==0
    assert p['source_id'] in splits['train'] and p['source_split']=='train'
    assert p['quality_status']=='draft'
    assert p['question']==p['original_question']==s['question']
    assert p['options']==p['original_options']==s['choices']
    assert p['hint']==s['hint'] and p['original_answer']==s['answer']
    assert p['answer_letter']==string.ascii_uppercase[s['answer']]
    assert p['answer_value']==s['choices'][s['answer']]
    assert p['input_prompt']==ns['prompt_for'](dict(question=p['question'],hint=p['hint'],options=p['options'],question_type='multiple-choice'),config)
    assert g['sample_id']==e['sample_id']==p['sample_id']
    assert g['messages'][:-1]==e['messages'][:-1]
    assert len(g['messages'])==2 and [m['role'] for m in g['messages']]==['user','assistant']
    assert g['messages'][0]['content']==[{'type':'image','path':p['image_path']},{'type':'text','text':p['input_prompt']}]
    assert g['messages'][-1]['content']==p['response_general']
    assert e['messages'][-1]['content']==p['response_era']
    for condition in ['general','era']:
        text=p['response_'+condition]
        assert text.splitlines()[-1]=='Final answer: '+p['answer_letter']
        assert len(re.findall(r'(?m)^Final answer:',text))==1
        assert not any('\uac00'<=c<='\ud7a3' for c in text)
    assert not re.search(r'(?m)^(Evidence|Rule|Application):',p['response_general'])
    assert re.findall(r'(?m)^(Evidence|Rule|Application):',p['response_era'])==['Evidence','Rule','Application']
    for unit in p['semantic_units']:
        for condition in ['general','era']:
            assert unit[condition+'_excerpt'] in p['response_'+condition],(p['source_id'],unit['id'],condition)
    image=ROOT/p['image_path']
    assert sha(image)==p['image_sha256']
    info=image_info(image); images.append((p,info))
    checks.append({'sample_id':p['sample_id'],'source_and_options':'pass','input_pair_and_baseline_template':'pass',
        'final_answer_and_era_headers':'pass','image_file_and_hash':'pass','semantic_excerpt_coverage':'pass',
        'semantic_equivalence':'author_model_self_review; independent human pending'})
assert len({i[1][:2] for i in images})==5
base=PROJECT/'results/mmmu_first_baseline_20261009'
ref=read(base/'baseline_seed3407/results.json')
exact_text=[]; near_text=[]; exact_image=[]; near_image=[]; missing=[]; count=0
for rec in ref['records']:
    text=normalize(rec['question']); words=set(text.split())
    for p,_ in images:
        target=normalize(p['question']); tw=set(target.split())
        if target==text: exact_text.append([p['source_id'],rec['id']])
        score=len(words&tw)/len(words|tw) if words|tw else 0
        if score>=0.8: near_text.append({'source_id':p['source_id'],'mmmu_id':rec['id'],'jaccard':score})
    for field in rec['image_fields']:
        path=base/'analysis/images'/f"{rec['id']}_{field}.png"
        if not path.exists(): missing.append(str(path)); continue
        info=image_info(path); count+=1
        for p,target in images:
            if info[:2]==target[:2]: exact_image.append([p['source_id'],path.name])
            distance=(info[2]^target[2]).bit_count()
            if distance<=4: near_image.append({'source_id':p['source_id'],'mmmu_image':path.name,'dhash_distance':distance})
tokens=read(ROOT/'token_report.json') if (ROOT/'token_report.json').exists() else None
if tokens:
    tokens['summary']={}
    for condition in ['general','era']:
        for part in ['total','body']:
            vals=[r[condition+'_'+part] for r in tokens['counts']]
            tokens['summary'][condition+'_'+part]={'sum':sum(vals),'min':min(vals),'max':max(vals),'mean':statistics.mean(vals),'median':statistics.median(vals)}
report={'schema_version':'era-draft-validation-v1','unique_questions':5,'responses':10,'all_train':True,
    'quality_status':'draft','final_release_ready':False,'checks':checks,
    'internal_exact_image_duplicates':0,'mmmu_overlap':{'dataset':ref['manifest']['configuration']['dataset'],
        'questions_compared':len(ref['records']),'images_compared':count,'missing_images':missing,
        'reference_results_sha256':sha(base/'baseline_seed3407/results.json'),
        'methods':['normalized exact question','word Jaccard >= 0.8','RGB size+pixel SHA256','64-bit dHash distance <= 4'],
        'exact_text_matches':exact_text,'near_text_candidates':near_text,'exact_image_matches':exact_image,'near_image_candidates':near_image,
        'limitation':'Heuristic similarity is a screening method, not proof that related problem families do not overlap.'},
    'tokens':tokens,'pending':['C reviewer human cross-review','other team drafts cross-owner and duplicate audit','separate development/evaluation lists and overlap audit','Notion publication destination','official image archive byte comparison'],
    'files_sha256':{p.name:sha(p) for p in ROOT.glob('*.jsonl')}}
(ROOT/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'questions':5,'responses':10,'structural_checks':'passed','mmmu_questions':len(ref['records']),
    'mmmu_images':count,'exact_text_matches':exact_text,'exact_image_matches':exact_image,
    'near_image_candidates':near_image,'near_text_candidates':near_text,'token_measurement':bool(tokens)},ensure_ascii=False))
