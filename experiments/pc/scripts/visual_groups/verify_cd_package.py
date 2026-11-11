"""Verify the real C/D output and exercise hypothetical splits without splitting training data."""
import copy
import json
from pathlib import Path
import visual_groups as v

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/CD_visual_groups_v1_20261017'
SOURCES={'C':ROOT/'results/C50_cross_review_20261017/source_C/paired_train.draft.jsonl',
         'D':ROOT/'results/MMDL_ERA_Dataset_D50_package/paired_train.draft.jsonl'}
reg=v.load(OUT/'visual_group_registry.json')
assert len(reg['images'])==100 and len(reg['groups'])==89 and reg['unresolved_pairs']==0
assert sum(len(g['member_ids'])>1 for g in reg['groups'])==9
source_rows={name:v.rows(path) for name,path in SOURCES.items()}
protected=['sample_id','source_id','source_dataset','source_revision','source_split','owner','reviewer','image_group_no',
           'question','hint','options','answer_letter','answer_value','input_prompt','response_general','response_era',
           'quality_status','common_content','semantic_units','image_sha256']
all_rows=[]
for name,path in SOURCES.items():
    new=v.rows(OUT/name/'paired_train.draft.jsonl')
    assert len(new)==50
    assert reg['source_manifest_sha256'][name]==v.sha(path.read_bytes())
    plain=v.rows(OUT/name/'train_plain.draft.jsonl');era=v.rows(OUT/name/'train_era.draft.jsonl')
    for old,row,p,e in zip(source_rows[name],new,plain,era):
        for field in protected:assert old.get(field)==row.get(field),(name,row['sample_id'],field)
        assert v.fingerprint(v.image_path(row,OUT/name/'paired_train.draft.jsonl'))['file_sha256']==old['image_sha256']
        assert p['visual_group_id']==e['visual_group_id']==row['visual_group_id']
        assert p['messages'][0]==e['messages'][0]
        assert p['messages'][0]['content'][1]['text']==old['input_prompt']
        assert p['messages'][1]['content']==old['response_general']
        assert e['messages'][1]['content']==old['response_era']
        all_rows.append(dict(row,image_path=name+'/'+row['image_path']))
    if name=='D':
        for filename in ['train_plain.draft.jsonl','train_era.draft.jsonl']:
            original=v.rows(path.parent/filename);exported=v.rows(OUT/name/filename)
            assert [x['messages'] for x in original]==[x['messages'] for x in exported]

# Treating C and D as separate splits must detect the known shared images.
report=v.check_records({name:OUT/name/'paired_train.draft.jsonl' for name in SOURCES},reg)
assert report['status']=='leak' and len(report['leaks'])==9 and not report['review_required']
report['test_only']='C/D are both training submissions; this hypothetical separation tests the checker, not an actual train/test split.'
v.dump(OUT/'verification/cd_separation_expected_leaks.json',report)

# A hypothetical group-aware 3-way partition must pass. Temporary manifests are removed afterwards.
group_splits={g['visual_group_id']:['train','dev','test'][i%3] for i,g in enumerate(reg['groups'])}
test_paths={s:OUT/f'_verification_{s}.jsonl' for s in ['train','dev','test']}
assert all(not p.exists() for p in test_paths.values())
try:
    for split,path in test_paths.items():v.jsonl(path,[r for r in all_rows if group_splits[r['visual_group_id']]==split])
    clean=v.check_records(test_paths,reg)
    assert clean['status']=='pass',clean
    clean['test_only']='Hypothetical group-based split for verification. No actual training/evaluation assignments were changed.'
    v.dump(OUT/'verification/group_aware_split_pass.json',clean)
finally:
    for path in test_paths.values():
        if path.exists():path.unlink()

summary={'source_rows':100,'output_rows':100,'visual_groups':89,'shared_groups':9,'unresolved_candidates':0,
         'preserved_assignment_numbers':True,'preserved_question_answers_and_responses':True,'identical_image_bytes':True,
         'general_ERA_same_user_input_and_group':True,'unchanged_D_original_export_messages':True,
         'hypothetical_C_D_separation_leak_pairs':len(report['leaks']),'hypothetical_group_aware_split':'pass',
         'actual_split_performed':False,'registry_id':reg['registry_id']}
v.dump(OUT/'verification/summary.json',summary)
print(json.dumps(summary,ensure_ascii=False,indent=2))
