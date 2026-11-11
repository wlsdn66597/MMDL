import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/era_astra_review_20261021'
BASE=ROOT/'results/era_abcd_prereview_20261021'
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def object_sha(r):return hashlib.sha256(json.dumps(r,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def csvrows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

records=rows(BASE/'paired_train.draft.jsonl');ids={r['sample_id'] for r in records}
reviews=rows(OUT/'astra_review_results.jsonl');blinds=rows(OUT/'astra_blind_results.jsonl')
statuses=rows(OUT/'review_status.jsonl');tasks=csvrows(OUT/'human_review_tasks.csv')
assert len(records)==len(reviews)==len(blinds)==len(statuses)==len(ids)==200
assert {r['sample_id'] for r in reviews}=={r['sample_id'] for r in blinds}==ids
byid={r['sample_id']:r for r in records};rev={r['sample_id']:r for r in reviews}
assert all(v['record_sha256']==object_sha(byid[v['sample_id']]) for v in reviews)
assert all(b['image_inspected'] for b in blinds)
assert sha(BASE/'paired_train.draft.jsonl')==sha(OUT/'reviewed_snapshot/paired_train.draft.jsonl')
for r in records:
    assert sha(BASE/r['image_path'])==sha(OUT/'reviewed_snapshot'/r['image_path'])
for manifest in (OUT/'astra').glob('*/review_manifest.json'):
    m=read(manifest)
    assert m['blind_results_sha256']==sha(manifest.parent/'blind_results.jsonl')
    assert m['human_review'] is False
taskids={r['sample_id'] for r in tasks}
assert len(tasks)==len(taskids)==73
assert {r['sample_id'] for r in reviews if r['verdict']!='pass'}<=taskids
assert all(not r['final_acceptance'] and r['human_status'] in {'pending','not_selected'} for r in statuses)
assert all(r['status']=='pending' and not any(r[k] for k in ['human_assessment','human_answer','action','notes','family_decision','reviewer_name','reviewed_at','revision_recheck']) for r in tasks)
per_reviewer=[r for who in 'ABCD' for r in csvrows(OUT/f'reviewer_{who}/checklist.csv')]
assert sorted(per_reviewer,key=lambda r:r['sample_id'])==sorted(tasks,key=lambda r:r['sample_id'])
for s in read(OUT/'sampling_manifest.json')['strata']:
    ranked=sorted(s['pool_ids'],key=lambda sid:hashlib.sha256(('20261021|'+sid).encode()).hexdigest())
    assert s['selected_ids']==ranked[:math.ceil(.2*len(ranked))]
    assert all(rev[sid]['verdict']=='pass' for sid in ranked)

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.cards=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        for k in ['href','src']:
            if k in a:self.links.append(a[k])
        if tag=='article':self.cards.append(a['id'])
        if tag=='img':self.images.append(a['src'])
p=Links();p.feed((OUT/'review.html').read_text(encoding='utf-8'))
assert len(p.cards)==len(p.images)==200 and set(p.cards)==ids
for link in p.links:
    if link.startswith('#'):assert link[1:] in ids
    else:assert (OUT/unquote(link)).is_file(),link

# Synthetic fixtures live only in a disposable temp folder, never in deliverables.
test_results={}
with tempfile.TemporaryDirectory(prefix='synthetic_merge_test_',dir=ROOT/'tmp') as tmp:
    dest=Path(tmp)
    shutil.copy2(OUT/'merge_checklists.py',dest/'merge_checklists.py')
    shutil.copy2(OUT/'human_review_tasks.csv',dest/'human_review_tasks.csv')
    for who in 'ABCD':shutil.copytree(OUT/f'reviewer_{who}',dest/f'reviewer_{who}')
    def run():
        result=subprocess.run([sys.executable,str(dest/'merge_checklists.py')],capture_output=True)
        audit=read(dest/'human_merge_audit.json')
        assert not audit['final_training_release']
        return result.returncode,audit
    code,audit=run();assert code==1 and audit['valid_completed']==0 and not audit['errors']
    test_results['pending_not_approved']='pass'
    def save_fixture(rows):
        for who in 'ABCD':
            with (dest/f'reviewer_{who}/checklist.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(r for r in rows if r['reviewer']==who)
    fixture=[dict(r,human_assessment='no_issue',human_answer=byid[r['sample_id']]['answer_letter'],action='keep',notes='SYNTHETIC TEST ONLY',family_decision='SYNTHETIC TEST ONLY',reviewer_name='SYNTHETIC TEST ONLY',reviewed_at='2026-10-21T12:00:00+09:00',status='reviewed') for r in tasks]
    save_fixture(fixture);code,audit=run();assert code==0 and audit['valid_completed']==73
    test_results['complete_fixture_never_releases_dataset']='pass'
    sample=[r for r in fixture if 'random_sample_20pct' in r['review_reason']][:2]
    for r in sample:r.update(human_assessment='confirmed_issue',action='revise')
    save_fixture(fixture);code,audit=run();assert audit['expansion']=='full_remaining_model_pass_review_required'
    test_results['sample_errors_require_expansion']='pass'
    fixture[0]['record_sha256']='stale-fixture-hash';save_fixture(fixture);code,audit=run()
    assert code==1 and any('record_sha256' in e for e in audit['errors'])
    test_results['stale_review_version_rejected']='pass'

audit=dict(result='pass',unique_model_reviewed=200,original_image_hashes_verified=200,record_hashes_verified=200,human_queue=73,human_fields_blank=True,original_paired_file_unchanged=True,html_cards=200,html_images=200,html_local_links_verified=len(p.links),sampling_reproduced=True,merge_tool_synthetic_tests=test_results,browser_rendering='NOT_RUN_local_file_browser_policy',final_training_release=False)
(OUT/'artifact_validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
