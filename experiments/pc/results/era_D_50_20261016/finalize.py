"""Audit final files, record review limitations, and make a curated local share bundle."""
import csv,hashlib,json,math,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
R=Path(__file__).resolve().parent
OLD=R.parent/'era_D_first5_20261015'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=rows(R/'paired_train.draft.jsonl');t=read(R/'token_report.json');v=read(R/'validation_report.json')
assert len(p)==50 and p[:5]==rows(OLD/'paired_train.draft.jsonl')
assert t['paired_file_sha256']==sha(R/'paired_train.draft.jsonl')
assert len(t['counts'])==50 and not any(x['review_length_difference'] or x['review_over_512'] for x in t['counts'])
with (R/'dataset_review.csv').open(encoding='utf-8-sig',newline='') as f:reviewcsv=list(csv.DictReader(f))
assert len(reviewcsv)==50
for q,c in zip(p,reviewcsv):
 assert json.loads(c['options_json'])==q['options']
 assert all(c[k]==q[k] for k in ['question','hint','response_general','response_era','answer_letter'])
with (R/'image_log.csv').open(encoding='utf-8-sig',newline='') as f:log=list(csv.DictReader(f))
assert len(log)==98
assert sum(x['상태']=='작성완료' for x in log)==50 and sum(x['상태']=='제외' for x in log)==48
assert {x['원본문항ID'] for x in log if x['상태']=='작성완료'}=={x['source_id'] for x in p}
# Independent code count of diagram particles (yellow/green connected components).
def particles(sid,color):
 im=Image.open(R/'images'/f'{sid}.png').convert('RGB');pix=im.load();w,h=im.size
 def ok(x,y):
  rr,g,b=pix[x,y]
  return (rr>220 and 130<g<220 and b<70) if color=='yellow' else (g>95 and rr<55 and b<140)
 seen=set();components=[]
 for y in range(h):
  for x in range(w):
   if (x,y) in seen or not ok(x,y):continue
   stack=[(x,y)];seen.add((x,y));pts=[]
   while stack:
    xx,yy=stack.pop();pts.append((xx,yy))
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
     nx,ny=xx+dx,yy+dy
     if 0<=nx<w and 0<=ny<h and (nx,ny) not in seen and ok(nx,ny):seen.add((nx,ny));stack.append((nx,ny))
   if len(pts)>70:components.append((sum(z[0] for z in pts)/len(pts),sum(z[1] for z in pts)/len(pts)))
 return components,w,h
yellow,w,h=particles('10177','yellow');green,wg,hg=particles('10296','green');diff,wd,hd=particles('1026','yellow')
yellow_counts=[sum(x<w/2 for x,y in yellow),sum(x>=w/2 for x,y in yellow)]
green_counts=[sum(x<wg/2 for x,y in green),sum(x>=wg/2 for x,y in green)]
diffcounts=[sum((x<wd/2)==left and (y<hd/2)==top for x,y in diff) for top,left in [(True,True),(True,False),(False,True),(False,False)]]
assert yellow_counts==[11,5] and green_counts==[12,11] and diffcounts==[3,5,4,4],(yellow_counts,green_counts,diffcounts)
checks=[
 dict(source_id='10177',method='visual equal-volume comparison plus independent pixel-component count',counts=yellow_counts,volume_mL=[50,50],particles_per_mL=[11/50,5/50],result='A > B'),
 dict(source_id='10296',method='visual equal-volume comparison plus independent pixel-component count',counts=green_counts,volume_mL=[50,50],particles_per_mL=[12/50,11/50],result='A > B'),
 dict(source_id='1026',method='visual count plus independent pixel-component count',initial=[3,5],final=[4,4],net_change=[1,-1],result='net leftward; individual crossings not inferred'),
 dict(source_id='10186',rock_volumes_billion_km3={'Mercury':60,'Venus':930,'Earth':1090,'Mars':160},result='max = Earth'),
 dict(source_id='10484',parent_alleles=[['F','F'],['f','f']],offspring=[x+y for x in ['F','F'] for y in ['f','f']],heterozygous_probability='4/4 = 1'),
 dict(source_id='10405',gaps_mm=[21.4,45.8],result='Pair 2 larger gap -> weaker in source model'),
 dict(source_id='10593',gaps_in=[3.75,1.75],result='Pair 2 smaller gap -> stronger in source model')]
write(R/'arithmetic_checks.json',dict(status='passed',checks=checks,scope='Counts and arithmetic supplement author visual review; no independent human validation.'))
near=v['mmmu_overlap']['near_image_candidates']
assert near==[dict(source_id='10590',mmmu_image='validation_Literature_6_image_1.png',dhash_distance=4)],near
decision=dict(source_id='10590',reference='validation_Literature_6_image_1.png',decision='false_positive_after_visual_self_review',
 evidence='ScienceQA shows a fish fossil with fins; MMMU image is a single black horizontal line on white. Different visual content despite dHash distance 4.',independent_human_review=False)
v['mmmu_overlap']['candidate_reviews']=[decision]
v['mmmu_overlap']['unresolved_candidates_in_this_screen']=0
v['csv_roundtrip']='50 rows match JSONL, including multiline responses and variable option counts'
v['history_counts']={'completed_drafts':50,'excluded':48,'total':98}
v['original_first5_unchanged']=True
v['arithmetic_checks']='arithmetic_checks.json'
v['token_report_sha256']=sha(R/'token_report.json')
v['production_status']='draft complete; human and integrated audit pending'
write(R/'validation_report.json',v)
audit=[dict(stage='MMMU similarity candidate review',**decision),dict(stage='completion_length_and_semantic_review',source_ids=[x['sample_id'].split('_')[-1] for x in read(R/'revisions/before_length_review/token_report.json')['counts'] if x['review_length_difference']],
 note='14 ERA responses had redundant wording shortened; natural general text unchanged. All final body differences <=15%. Same author model reviewed meaning, not independent human review.')]
reviews=[x for x in rows(R/'review_log.jsonl') if x.get('stage') not in ['MMMU similarity candidate review','completion_length_and_semantic_review'] and not x.get('note','').startswith('14 ERA responses')]
with (R/'review_log.jsonl').open('w',encoding='utf-8') as f:
 for x in reviews+audit:f.write(json.dumps(x,ensure_ascii=False)+'\n')
v['files_sha256']={f.name:sha(f) for f in R.glob('*.jsonl')}
write(R/'validation_report.json',v)
# File-start/end wall time is explicitly not person-hours or a future training estimate.
start=datetime.fromtimestamp((R/'progress.json').stat().st_ctime,timezone.utc)
end=datetime.now(timezone.utc);mins=round((end-start).total_seconds()/60)
write(R/'work_timing.json',dict(measurement='wall clock from initial progress-file creation to final packaging audit',start=start.isoformat(),end=end.isoformat(),elapsed_minutes_rounded=mins,
 excludes=['earlier first-five production','future human cross-review','future integrated merge audit','model training'],note='Approximate file-record interval; not human labor time.'))
maxdiff=max(x['body_relative_difference'] for x in t['counts'])
summary=dict(unique_questions=50,existing_preserved=5,new_questions=45,plain_responses=50,era_responses=50,split='train',quality_status='draft',
 option_count_distribution={str(n):sum(len(x['options'])==n for x in p) for n in sorted({len(x['options']) for x in p})},
 body_length_max_relative_difference=maxdiff,over_512=0,length_flags_over_15_percent=0,
 max_response_tokens=max(max(x['general_total'],x['era_total']) for x in t['counts']),
 selection_history={'rows':98,'completed':50,'excluded':48},source_solution_cautions=['10126','10168'],
 mmmu_screen={'questions':900,'images':982,'exact_matches':0,'heuristic_candidates':1,'candidate_review':'visually different / false positive'},
 independent_human_review=False,reviewer='C',final_accepted=False)
write(R/'검증요약.json',summary)
readme=f'''D 담당 ScienceQA 50문항 공유용 초안

기존 5문항을 그대로 보존하고 45문항을 추가했습니다. 고유 문제 50개에 일반 풀이 50개와 ERA 풀이 50개가 대응합니다. 풀이 100개를 고유 문항 100개로 세지 않습니다. 전부 학습용이며 개발·평가셋은 포함하지 않습니다.

확인 순서
1. ZIP을 풀고 review.html을 열면 이미지·원문·보기·두 풀이를 함께 볼 수 있습니다.
2. dataset_review.csv: 실제 작성된 50문항과 두 풀이. UTF-8 BOM, 원본 가변 보기 수를 options_json에 보존.
3. image_log.csv: 공통 선별 형식의 이력 98행 = 작성완료 50 + 제외 48. 작성완료는 초안 작성 완료이며 accepted가 아닙니다. 제외를 삭제하거나 그룹 번호를 다시 매기지 마세요.
4. paired_train.draft.jsonl: 출처·제작 조건·공통 내용·검토 메모가 포함된 기준 데이터.
5. train_plain.draft.jsonl / train_era.draft.jsonl: 각각 50행의 로컬 학습 준비용 메시지. 이미지 경로는 이 폴더 기준입니다. images/ 50개를 함께 보관하세요.
6. selected_questions.jsonl의 lecture·solution·answer는 교사/검수 자료입니다. 학생 입력에 합치지 마세요.

검증 결과
원본 ScienceQA train/짝수 이미지 그룹·원문·hint·보기·정답·동일 학생 입력·이미지 해시 검증 통과. 원본 보기를 변경하지 않았고 기존 5문항 레코드도 변경하지 않았습니다. 실제 답변 토큰 최대 {summary['max_response_tokens']}개, 512 초과 0개. ERA 라벨과 마지막 정답 줄을 뺀 본문 길이 차이 최대 {maxdiff:.2%}, 15% 초과 0개. 로컬 공개 토크나이저의 첫 5문항 계수는 이전 실습 PC Transformers 계수와 정확히 일치했습니다.
MMMU 기존 기준 900문항/982이미지 대조에서 정확한 중복 없음. dHash 후보 1건은 생선 화석 대 검은 가로선 그림으로 직접 비교해 서로 다른 내용으로 판정했습니다. 이 검사는 일반적인 주제·문제 유형의 중첩까지 배제하는 증명이 아닙니다.
원본 해설 10126·10168에 모순/깨진 문구가 있어 이미지·정답 대조 근거를 메모했습니다. 10448의 배경 연대 등 답에 쓰지 않은 모든 원문 사실까지 외부 검증한 것은 아닙니다.

남은 확인
모든 레코드는 draft이며 C 담당 사람 교차 검수 전입니다. 실제 이미지 의존성, 일반/ERA 정보·수치·조건 동등성, 지도·도표 판독, 정답 및 원문 결함 메모를 확인하세요. 타 담당 문항과 별도 개발·평가셋을 모은 뒤 중복·누출을 다시 확인해야 합니다. 수정 시 두 풀이를 함께 검토하고 토큰 수를 다시 측정하세요. 모델 자가 검토를 사람 검수로 표시하지 마세요.

제작 방법 및 시간
Codex 시각 판독·모델 작성/자가 검토 + Python 직렬화·공통 선별·산술/해시 검증. 반복 유형에 코드 도우미를 사용했습니다. 정확한 serving model ID, revision, sampling 설정은 노출되지 않아 null이며 사람 수정은 없습니다. GPU 생성/학습은 하지 않았습니다. 이번 추가분 파일 생성~검증/패키징 약 {mins}분(파일 기록 기준, 앞선 5문항 제작 및 사람 검수 시간 제외).

공개/학습 형식
Hugging Face 업로드용 Arrow/Parquet 형식으로 최종 변환한 패키지가 아닙니다. 특히 messages.content의 사용자 멀티모달 배열과 assistant 문자열은 통합 시 정규화가 필요합니다. 전체 200문항 통합 때 이미지 특징·스키마·데이터카드·학습/별도 평가 분리 및 라이선스를 확정하세요. 이 작업에서 외부 업로드나 팀 메시지 전송은 하지 않았습니다.
공식 데이터 라이선스 CC-BY-NC-SA-4.0을 LICENSE-DATA.txt로 포함했습니다. 이미지 출처는 고정된 공식 연결 HF 미러의 원본 PNG이며 공식 Google Drive 이미지 아카이브와의 바이트 비교는 수행하지 않았습니다. 세부 출처는 image_manifest.json과 generation_provenance.json을 보세요.
'''
(R/'README_공유.txt').write_text(readme,encoding='utf-8')
message=f'''D 담당 ScienceQA 50문항 초안 공유합니다. 기존 5문항에 45문항 추가했고, 전부 학습용으로 일반·ERA 풀이를 각각 작성했습니다.

ZIP 풀고 review.html로 보면 이미지와 두 풀이를 같이 확인할 수 있어요. 작성 문항 CSV는 50행, 선별 이력 CSV는 제외 사유까지 포함해 98행입니다.

추가 제작·검증은 파일 기록 기준 약 {mins}분 걸렸고, 원문·정답·형식·길이 검사는 통과했습니다. 아직 사람 교차 검수 전이라 이미지가 꼭 필요한지, 두 풀이의 정보가 같은지, 원본 해설 오류 메모를 확인 부탁드립니다. 전체 200문항을 모을 때 팀 문항·별도 평가셋과 중복도 다시 확인하면 됩니다.'''
(R/'카톡_공유메시지.txt').write_text(message+'\n',encoding='utf-8')
progress=read(R/'progress.json');progress.update(status='draft_complete',completed=50,remaining=0,human_review='pending',elapsed_minutes_rounded=mins)
write(R/'progress.json',progress)
# Curate only useful delivery files. Source scripts, candidate downloads and cache stay in work folder.
share=R.parent/'era_D_50_team_share_20261016';share.mkdir(exist_ok=True)
names=['paired_train.draft.jsonl','train_plain.draft.jsonl','train_era.draft.jsonl','common_content.jsonl','review_log.jsonl','image_log.csv','dataset_review.csv','selected_questions.jsonl','review.html','README_공유.txt','카톡_공유메시지.txt','검증요약.json','validation_report.json','token_report.json','arithmetic_checks.json','image_manifest.json','generation_provenance.json','authoring_instructions.txt','work_timing.json']
for name in names:shutil.copy2(R/name,share/name)
shutil.copytree(R/'provenance',share/'provenance',dirs_exist_ok=True)
(share/'images').mkdir(exist_ok=True)
for q in p:shutil.copy2(R/q['image_path'],share/q['image_path'])
shutil.copy2(R/'provenance/ScienceQA_LICENSE-DATA.txt',share/'LICENSE-DATA.txt')
write(share/'package_manifest.json',dict(unique_questions=50,quality_status='draft',files={str(f.relative_to(share)).replace('\\','/'):sha(f) for f in sorted(share.rglob('*')) if f.is_file() and f.name!='package_manifest.json'}))
archive=share.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
 for f in sorted(share.rglob('*')):
  if f.is_file():z.write(f,str(f.relative_to(share)).replace('\\','/'))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 manifest=json.loads(z.read('package_manifest.json'))
 for n,digest in manifest['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==digest,n
 assert sum(n.startswith('images/') for n in z.namelist())==50
print(json.dumps(dict(archive=str(archive),files=len(manifest['files'])+1,summary=summary,elapsed_minutes=mins),ensure_ascii=False))
