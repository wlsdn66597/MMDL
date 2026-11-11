"""Serialize image-bound visual adjudications completed in the 2026-10-17 review.

The new 65 candidate comparisons were all visually inspected in new_panels/001-065.
This script encodes those judgments; it does NOT classify future pairs automatically.
"""
from pathlib import Path
import visual_groups as v

ROOT=Path(__file__).resolve().parents[2]
scan=ROOT/'results/CD_visual_groups_scan_20261017'
review=ROOT/'results/C50_cross_review_20261017'
inventory=v.load(scan/'inventory.json')['images'];by_id={r['sample_id']:r for r in inventory}
old=v.load(review/'duplicate_pair_review.json')
old_info=v.load(review/'automated_audit.json')['image_details']
new=v.load(scan/'new_candidates.json')
assert len(old)==60 and len(new)==65
decisions=[]
for pair in old:
    a,b=by_id[pair['left']],by_id[pair['right']]
    assert a['file_sha256']==old_info[a['sample_id']]['file_sha256']
    assert b['file_sha256']==old_info[b['sample_id']]['file_sha256']
    decisions.append(dict(left=a['sample_id'],right=b['sample_id'],left_sha256=a['file_sha256'],right_sha256=b['file_sha256'],
        verdict='same' if pair['group_id'] else 'different',reviewer='Codex AI visual review; human final approval pending',
        reason='같은 지도·강조 대상·표기 내용이며 미세한 픽셀 차이만 있음' if pair['group_id'] else '같은 양식이지만 강조 국가·주 이름·입자 수·유전자형 등 문제에 필요한 시각 내용이 다름',
        evidence_file='results/C50_cross_review_20261017/'+pair['comparison_panel'],review_date='2026-10-17'))
for i,pair in enumerate(new,1):
    a,b=by_id[pair['left']],by_id[pair['right']]
    if i<=5 or i in [12,13,61,62]:reason='동일 바탕 지도지만 초록 강조 영역·테두리의 위치가 다름. 동일 이미지로 묶지 않음.'
    elif i>=63:reason='용액 그림의 입자 수·색·부피 표기가 다름. 25/35/50mL와 두 용액의 입자 배치를 대조함.'
    else:reason='같은 미국 지도 바탕이지만 표시된 주 이름과 연결선의 대상이 다름. 라벨은 문제 해결에 필요한 시각 내용이므로 다른 이미지로 관리함.'
    decisions.append(dict(left=a['sample_id'],right=b['sample_id'],left_sha256=a['file_sha256'],right_sha256=b['file_sha256'],
        verdict='different',reviewer='Codex AI visual review; human final approval pending',reason=reason,
        evidence_file=f'results/CD_visual_groups_scan_20261017/new_panels/{(i-1)//4*4+1:03d}-{min((i-1)//4*4+4,65):03d}.jpg',review_date='2026-10-17'))
assert len({v.key(d['left'],d['right']) for d in decisions})==125
v.dump(Path(__file__).with_name('cd_reviewed_pairs_20261017.json'),decisions)
print('Saved 125 image-bound pair decisions (14 same; 111 different).')
