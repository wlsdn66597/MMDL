from pathlib import Path
import sys,json,itertools
ROOT=Path('C:/project/MMDL');TMP=ROOT/'tmp/era_pilot_20261027';BASE=ROOT/'results/ERA_검토_200문항/era_astra_review_20261021/reviewed_snapshot'
sys.path.insert(0,str(ROOT/'scripts/visual_groups'))
from visual_groups import fingerprint,is_candidate,distance,panels
old=json.loads((BASE/'visual_group_registry.json').read_text(encoding='utf-8'))
data=[]
for r in old['images']:
 r=dict(r,image_path=str(BASE/r['image_path']));data.append(r)
cand={r['source_id']:r for r in json.loads((TMP/'candidates.json').read_text(encoding='utf-8'))}
content=json.loads((TMP/'new_pair_content.json').read_text(encoding='utf-8'));new=[]
for sid,s in content.items():
 r=cand[s['source_id']];info=dict(sample_id=sid,image_path=r['local_image'],source_id=r['source_id'],package=s['owner'],image_group_no=r['image_group_no'],**fingerprint(r['local_image']));new.append(info)
pairs=[]
for a,b in itertools.combinations(data+new,2):
 if a in data and b in data:continue
 if a['pixel_sha256']==b['pixel_sha256'] or is_candidate(a,b):pairs.append(dict(left=a['sample_id'],right=b['sample_id'],**distance(a,b)))
(TMP/'new_fingerprints.json').write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
(TMP/'new_image_pairs.json').write_text(json.dumps(pairs,ensure_ascii=False,indent=2),encoding='utf-8')
panels(data+new,pairs,TMP/'new_pair_panels')
refs=json.loads((ROOT/'reports/era_abcd_review_20261020/mmmu_reference_fingerprints.json').read_text(encoding='utf-8'))
mm=[]
for a in new:
 for b in refs:
  if a['pixel_sha256']==b['pixel_sha256'] or is_candidate(a,b):mm.append(dict(sample_id=a['sample_id'],reference_id=b['id'],**distance(a,b)))
(TMP/'new_mmmu_candidates.json').write_text(json.dumps(mm,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'new_images':len(new),'new_pairs':len(pairs),'mmmu_candidates':len(mm)}));print(json.dumps(pairs,ensure_ascii=False))
