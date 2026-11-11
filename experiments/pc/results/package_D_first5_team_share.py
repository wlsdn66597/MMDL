"""Create a review-only handoff bundle without modifying the working dataset."""
import csv, hashlib, json, re, shutil, zipfile
from pathlib import Path

BASE=Path(__file__).resolve().parent
SRC=BASE/'era_D_first5_20261015'
OUT=BASE/'era_D_first5_team_share_20261015'
OUT.mkdir(exist_ok=False)
(OUT/'images').mkdir()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj): p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

pairs=[json.loads(x) for x in (SRC/'paired_train.draft.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(pairs)==5
for name in ['review.html','paired_train.draft.jsonl','train_plain.draft.jsonl','train_era.draft.jsonl',
             'first5_selected_questions.jsonl','image_log.csv','run_info.json','review_log.jsonl']:
    shutil.copy2(SRC/name,OUT/name)
for p in pairs: shutil.copy2(SRC/p['image_path'],OUT/p['image_path'])
shutil.copy2(SRC/'provenance/ScienceQA_LICENSE-DATA.txt',OUT/'ScienceQA_LICENSE-DATA.txt')
v=read(SRC/'validation_report.json')
summary={k:v[k] for k in ['schema_version','unique_questions','responses','all_train','quality_status',
    'final_release_ready','checks','internal_exact_image_duplicates','mmmu_overlap','pending']}
summary['token_measurement']={k:v['tokens'][k] for k in ['model_id','revision','tokenizer_sha256','add_special_tokens','body_definition','counts','summary']}
summary['original_validation_report_sha256']=digest(SRC/'validation_report.json')
write(OUT/'validation_summary.json',summary)

message='''D 담당 ScienceQA 첫 5문항 초안 공유합니다! 같은 문제에 일반·ERA 풀이를 각각 작성했고, ZIP 풀고 review.html 열면 이미지랑 풀이를 같이 볼 수 있어요.

후보 선별부터 이미지 확보, 풀이 작성, 자체 검증·파일 정리까지 기록상 약 16분 걸렸습니다. 사람 교차 검수 시간은 제외예요.

데이터셋 만들면서 아래 항목을 확인하면 될 것 같습니다.
- 그림이 실제로 필요한 문제인지
- 이미지 근거·풀이·정답이 맞는지
- 일반·ERA 풀이에 같은 정보가 들어가는지
- 원본 질문·hint·보기 순서가 유지됐는지
- 담당 간 중복이나 평가셋과의 중복이 없는지
- 제작 방법·모델·제외 사유가 기록됐는지

현재는 draft이고, C 교차 검수와 통합 중복 검사는 남아 있습니다. 우선 이 5문항으로 작성 방향이 맞는지 봐주세요!
'''
(OUT/'공유_메시지.txt').write_text(message,encoding='utf-8-sig')

guide='''D 담당 ScienceQA 첫 5문항 팀 검토 패키지
2026-10-15 | 전부 학습용 | 고유 문항 5개 / 일반 5개 + ERA 5개 | draft

먼저 ZIP을 압축 해제하고 review.html을 브라우저로 열어주세요.
HTML만 따로 옮기면 이미지가 표시되지 않으므로 images 폴더와 함께 유지해주세요.
공유_메시지.txt는 팀 채팅이나 노션에 붙여 넣을 설명입니다.

검토 대상
10008 / 짝수 그룹 4 / 지도에서 가장 북쪽인 주 / D, New Jersey
10066 / 짝수 그룹 14 / 강조된 식민지 / A, Georgia
10076 / 짝수 그룹 18 / 마주 보는 자석 극 / A, attract
10098 / 짝수 그룹 24 / 물체의 색 / A, colorful
10111 / 짝수 그룹 28 / 물체의 투명성 / B, transparent

파일 안내
- review.html + images/: 이미지·학생 입력·두 풀이·공통 내용표 검토 화면. 이미지 5개만 포함.
- paired_train.draft.jsonl: 동일 5문항의 출처, 공통 내용표, 의미 단위 대응, 일반·ERA 풀이, 제작 조건, 자체 검토 기록.
- train_plain.draft.jsonl / train_era.draft.jsonl: 동일한 5개 입력에 각각 일반/ERA 응답을 연결한 파일. 합쳐서 고유 10문항으로 취급하지 않음.
- first5_selected_questions.jsonl: 해당 5문항의 원문·정답·원본 보기·lecture·solution 등 제작 참고 자료. 학생 입력에 넣지 않음.
- image_log.csv: 공통 코드 헤더를 유지한 최신 관리 CSV. 작성완료 5 + 예정 45 + 제외 9 = 59행. 예정 45문항은 풀이 제작·이미지 검토를 완료한 것이 아님. 제외 행을 삭제하지 않음.
- run_info.json: 원본 지문·선별 버전·후보 50개 유지·최초 5개 작성 범위. selected=50은 후보 수이며 완성 문항 수가 아님.
- review_log.jsonl: 제외 사유와 작성 모델의 자체 검토·길이 수정 이력.
- validation_summary.json: 실제 형식·MMMU 중복 검사 범위와 결과, Qwen 토큰 측정 결과.
- ScienceQA_LICENSE-DATA.txt: 공식 데이터 라이선스 원문.
- package_manifest.json: 공유 파일별 SHA256.

이번 공유에서 맞출 부분
이미지 필요성, 근거·규칙·적용의 정확성, 두 응답의 정보 동등성, 일반 풀이의 자연스러움, 입력·저장 형식을 봐주세요. 문제 ID와 수정이 필요한 문장, 근거를 함께 알려주시면 반영하기 쉽습니다. 특히 10008은 지도 내 상대 위치를 근거로 채택했으며 이미지 필요성 판단도 함께 검토할 수 있습니다.

작성·검증 상태
같은 Codex 모델이 같은 이미지·공통 자료로 두 응답을 작성했습니다. 사람 교차 검수 전이며, 정확한 serving model ID·revision·sampling 설정은 제공되지 않아 추정하지 않고 null/미노출로 기록했습니다. Qwen은 응답 생성이 아니라 토큰 측정에 사용했습니다.
모든 응답은 512토큰 이내이고, 표제·최종 답 줄을 제외한 본문 상대 길이 차이는 최대 13.11%입니다. 일반·ERA 토큰 수가 완전히 같은 것은 아닙니다.
기존 MMMU validation 900문항·982개 이미지에 대해 정규화 질문 일치, RGB 픽셀 해시, 단어 Jaccard >= 0.8, dHash 거리 <= 4로 검사했으며 일치·유사 후보는 없었습니다. 해당 검사 범위를 넘어 누출이 없다고 확정한 것은 아닙니다.
C 담당자의 사람 교차 검수, 다른 담당 초안과의 통합 검사, 별도 개발·최종 평가셋과의 중복 검사는 미완료입니다. accepted나 최종 200문항 확정본이 아닙니다.

입력·이미지 경로
ScienceQA 원문 질문·hint·보기·정답을 유지했습니다. 두 학생용 파일의 입력은 동일하며 기존 baseline 프롬프트를 적용했습니다. 정답·lecture·solution·검수 메모를 학생 입력에 넣지 않았습니다.
paired 및 학생용 JSONL의 images/... 경로는 이 패키지 루트 기준입니다. image_log.csv의 train/문항ID/image.png는 공통 코드가 관리하는 원본 이미지 상대 경로이며 변경하지 않았습니다. 둘 사이의 대응은 paired 파일의 original_image_path와 image_path에서 확인할 수 있습니다.
현재 JSONL은 제작·취합용입니다. 특정 학습기의 즉시 실행 형식이나 Hugging Face 업로드 형식으로 변환한 결과는 아닙니다. 200문항 취합·검수 후 변환하면 됩니다.

출처
공식 ScienceQA revision: 2cbf8318e07b9ece895bb2ae605e71e38d623264
https://github.com/lupantech/ScienceQA/tree/2cbf8318e07b9ece895bb2ae605e71e38d623264
이미지는 공식 저장소가 안내하는 derek-thomas/ScienceQA 미러의 revision f18b0a70359ebfb41f658fd564208d0355b013f4에서 PNG 바이트 그대로 추출했습니다. 질문·보기·정답·hint 등 원본 메타데이터를 대조했습니다. 공식 이미지 ZIP과 별도의 바이트 대조는 하지 않았습니다. 파일·픽셀 해시와 출처 정보는 paired 파일에 보존되어 있습니다.
공식 LICENSE-DATA의 CC BY-NC-SA 4.0 조건과 출처를 유지합니다.

범위
불필요한 실행 코드, 원격 환경 정보 파일, 이전 수정본, 전체 후보 50개의 상세 JSONL, 제외된 9개 이미지, 중복 압축파일은 넣지 않았습니다. 최신 관리 CSV에는 이력 보존을 위해 예정·제외 행이 남아 있습니다. 실제 팀 전송이나 노션 게시를 수행한 것은 아닙니다.
'''
(OUT/'먼저_읽어주세요.txt').write_text(guide,encoding='utf-8-sig')

assert len(list((OUT/'images').glob('*.png')))==5
for name in ['paired_train.draft.jsonl','train_plain.draft.jsonl','train_era.draft.jsonl','first5_selected_questions.jsonl']:
    assert len((OUT/name).read_text(encoding='utf-8').splitlines())==5
page=(OUT/'review.html').read_text(encoding='utf-8')
assert len(re.findall(r'<article ',page))==5
for name in re.findall(r'<img src="([^"]+)"',page): assert (OUT/name).is_file()
with (OUT/'image_log.csv').open(encoding='utf-8-sig',newline='') as f: log=list(csv.DictReader(f))
assert len(log)==59 and sum(r['상태']=='작성완료' for r in log)==5
assert {r['원본문항ID'] for r in log if r['상태']=='작성완료'}=={r['source_id'] for r in pairs}
files={p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file()}
write(OUT/'package_manifest.json',{'package':'D first five team review draft','questions':5,'responses':10,'files_sha256':files})
archive=BASE/(OUT.name+'.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file(): z.write(p,(Path(OUT.name)/p.relative_to(OUT)).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for rel,h in files.items(): assert hashlib.sha256(z.read(OUT.name+'/'+rel)).hexdigest()==h
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'files':len(files)+1,'questions':5,'responses':10,'images':5,'csv_rows':len(log),'checks':'passed'},ensure_ascii=True))
