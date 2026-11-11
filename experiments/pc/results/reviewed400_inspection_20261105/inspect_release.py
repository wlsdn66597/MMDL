from pathlib import Path
import argparse, collections as C, hashlib, io, json, re, zipfile, unicodedata
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument("zip", type=Path)
args = p.parse_args()
def sha(b): return hashlib.sha256(b).hexdigest()
def rows(z, name): return list(map(json.loads, z.read(name).splitlines()))
def byid(rs):
    assert len(rs)==len({x['id'] for x in rs})
    return {x['id']: x for x in rs}
def norm(s): return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def pixel(im):
    rgb=im.convert('RGB')
    return (rgb.size, sha(rgb.tobytes()))
def question(r):
    # Extract the stem before any choices or trailing answer-format instruction.
    s=r['input_prompt']
    if 'Question: ' in s: s=s.split('Question: ',1)[1]
    return norm(re.split(r'\n(?:Options:|On the last line,)',s,maxsplit=1)[0])
def count(rs,k): return dict(sorted(C.Counter(r[k] for r in rs).items()))
def buckets(rs, key):
    out=C.defaultdict(list)
    for r in rs: out[key(r)].append(r)
    return out
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def without_headings(t):
    return ' '.join(re.sub(r'^(?:Evidence|Reasoning):\s*','',t,flags=re.M).split())

with zipfile.ZipFile(args.zip) as z:
    assert z.testzip() is None
    mf=json.loads(z.read('manifest.json'))
    assert len(set(z.namelist()))==len(z.namelist()) and set(mf)|{'manifest.json'}==set(z.namelist())
    assert all(sha(z.read(n))==digest for n,digest in mf.items())
    er=byid(rows(z,'data/d_era.jsonl')); ge=byid(rows(z,'data/d_general.jsonl')); rr=byid(rows(z,'data/r_original.jsonl'))
    pv=byid(rows(z,'metadata/provenance.jsonl')); der=byid(rows(z,'metadata/format_derivations.jsonl'))
    assert er.keys()==ge.keys()==der.keys() and not (er.keys()&rr.keys()) and er.keys()|rr.keys()==pv.keys()
    unique=list(er.values())+list(rr.values())
    assert len({(r['input_prompt'],tuple(r['images'])) for r in unique})==len(unique)
    for r in list(ge.values())+unique:
        assert sha(r['target'].encode())==r['target_sha256']
        assert r['images']==pv[r['id']]['images'] and [mf[n] for n in r['images']]==pv[r['id']]['images_sha256']
    for i,e in er.items():
        g=ge[i]
        assert all(e[k]==g[k] for k in ('id','source_group','source_slug','images','input_prompt','evidence','reasoning','answer'))
        assert all(e[k] and e[k].strip() for k in ('evidence','reasoning','answer'))
        assert e['target']=='Evidence: '+e['evidence']+'\nReasoning: '+e['reasoning']+'\nFinal answer: '+e['answer']
        assert without_headings(e['target'])==without_headings(g['target'])
        assert g['target_sha256']==pv[i]['target_sha256']==e['original_target_sha256']==der[i]['source_target_sha256']
        assert e['target_sha256']==der[i]['era_target_sha256']
    for i,r in rr.items():
        assert all(r[k] is None for k in ('evidence','reasoning','answer'))
        assert r['target_sha256']==r['original_target_sha256']==pv[i]['target_sha256']==pv[i]['source_evidence']['original_base_response_sha256']
    imgs={im for r in unique for im in r['images']};assert imgs=={n for n in mf if n.startswith('images/')}
    pixels={}
    for n in imgs:
        with Image.open(io.BytesIO(z.read(n))) as im: pixels[n]=pixel(im)
    pre=json.loads(z.read('metadata/era_processor_preflight.json'))
    assert pre['rows_sha256']==mf['data/d_era.jsonl'] and set(er)=={r['id'] for r in pre['per_row']}
    pilotdir=ROOT/'results/era_pilot_200_20261028_r2'
    old=list(map(json.loads,(pilotdir/'paired_train.pilot.jsonl').read_text(encoding='utf-8-sig').splitlines()))
    oldpix=set()
    for r in old:
        with Image.open(pilotdir/r['image_path']) as im: oldpix.add(pixel(im))
    im_index=load(ROOT/'results/mmmu_first_baseline_20261009/analysis/images/index.json')
    evalpix={(tuple(im['size']),im['rgb_bytes_sha256']) for ims in im_index.values() for im in ims}
    ev=load(ROOT/'results/mmmu_first_baseline_20261009/analysis/review_records.json')
    oq={norm(r[k]) for r in old for k in ('question','original_question')}
    eq={norm(r['question']) for r in ev}
    def matches(stems):return [r['id'] for r in unique if question(r) in stems]
    sq={pv[i]['source_evidence']['original_question_record']['metadata']['original_id'] for i in rr if rr[i]['source_slug']=='scienceqa'}
    oldsq={r['source_id'] for r in old if r['source_dataset']=='ScienceQA'}
    ds=list(er.values()); di=buckets(ds,lambda r:r['images'][0]); dq=buckets(ds,lambda r:r['input_prompt'])
    qvary=[v for v in dq.values() if len({r['answer'] for r in v})>1 and len({tuple(r['images']) for r in v})>1]
    group_used=buckets([r for r in unique if r['images']],lambda r:pixels[r['images'][0]])
    many_groups=[sorted({r['source_group'] for r in v}) for v in group_used.values() if len({r['source_group'] for r in v})>1]
    out={'archive_sha256':sha(args.zip.read_bytes()),'archive_size_bytes':args.zip.stat().st_size,
    'manifest_files_verified':len(mf),'unique_questions':len(unique),'unique_input_image_pairs':len({(r['input_prompt'],tuple(r['images'])) for r in unique}),
    'role_counts':count(unique,'role'),'with_image':sum(bool(r['images']) for r in unique),'unique_image_files':len(imgs),'unique_image_pixels':len(set(pixels.values())),
    'd_pairs_verified':len(er),'r_original_hashes_matched_to_provenance':len(rr),'r_without_separate_evidence_reasoning_answer':len(rr),
    'derivations':count(list(der.values()),'derivation'),'sources':{s:count(v,'role') for s,v in sorted(buckets(unique,lambda r:r['source_slug']).items())},
    'd_source_groups':len({r['source_group'] for r in ds}),'d_image_use_histogram':dict(C.Counter(map(len,di.values()))),
    'd_images_crossing_source_groups_in_all_roles':many_groups,'d_mc_options_prompts':sum('\nOptions:\n' in r['input_prompt'] for r in ds),
    'd_distinct_prompts':len(dq),'d_same_prompt_different_answer_clusters':len(qvary),'d_rows_in_these_clusters':sum(map(len,qvary)),
    'variable_prompt_clusters':[{'source':v[0]['source_slug'],'task':v[0]['task'],'n':len(v),'distinct_answers':len({r['answer'] for r in v}),'ids':[r['id'] for r in v]} for v in qvary],
    'original_package_era_processor_log':{'max_total_tokens':max(r['total_tokens'] for r in pre['per_row']),'rows':len(pre['per_row']),'skipped':pre['skipped'],'truncated':pre['truncated'],'above_4096':sum(r['total_tokens']>4096 for r in pre['per_row'])},
    'overlap':{'exact_pixel_key':'stored image orientation, width/height and decoded RGB bytes SHA-256','pilot_rows':len(old),'pilot_unique_pixels':len(oldpix),'pilot_common_images':len(set(pixels.values())&oldpix),'mmmu_rows':len(ev),'mmmu_image_records':sum(map(len,im_index.values())),'mmmu_unique_pixels':len(evalpix),'mmmu_common_images':len(set(pixels.values())&evalpix),
    'new_scienceqa_original_ids':len(sq),'old_scienceqa_original_ids':len(oldsq),'scienceqa_shared_ids':sorted(sq&oldsq),
    'question_stem_matching_caveat':'Normalized exact stems, ignoring hints, options, and images; same stems need not mean same item.',
    'pilot_matching_stem_ids':matches(oq),'mmmu_matching_stem_ids':matches(eq),'scope':'Exact local checks only; does not rule out crops, near duplicates, templates, or semantic similarity.'},
    'visual_spot_checks':{'accounting':['da_1c1efd1b44146e78','ta_b27155f288bf2495602e'],'music':['ta_67634ac6bc68f5b2807d','ta_8d06ebc0c254d46b6e73']}}
file=Path(__file__).with_name('audit.json')
file.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='variable_prompt_clusters'},ensure_ascii=False))
