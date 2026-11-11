"""Prepare two upload files without publishing or altering the 50 question pairs."""
import csv, hashlib, json, re, shutil, zipfile
from pathlib import Path

BASE=Path(__file__).resolve().parent
SRC=BASE/'era_D_50_team_share_20261016'
OUT=BASE/'MMDL_ERA_Dataset_upload'
PACKAGE=BASE/'MMDL_ERA_Dataset_D50_package'
OUT.mkdir(exist_ok=True); PACKAGE.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]

names=['paired_train.draft.jsonl','train_plain.draft.jsonl','train_era.draft.jsonl',
 'common_content.jsonl','selected_questions.jsonl','dataset_review.csv','image_log.csv',
 'image_manifest.json','review.html','review_log.jsonl','arithmetic_checks.json',
 '검증요약.json','LICENSE-DATA.txt','work_timing.json']
for name in names:shutil.copy2(SRC/name,PACKAGE/name)
paired=rows(PACKAGE/'paired_train.draft.jsonl')
assert len(paired)==50
(PACKAGE/'images').mkdir(exist_ok=True)
for p in paired:shutil.copy2(SRC/p['image_path'],PACKAGE/p['image_path'])

token=read(SRC/'token_report.json')
token.pop('tokenizer_path',None)
token['tokenizer_source']='https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/'+token['revision']+'/tokenizer.json'
write(PACKAGE/'token_report.json',token)
validation=read(SRC/'validation_report.json')
validation['tokens'].pop('tokenizer_path',None)
validation['tokens']['tokenizer_source']=token['tokenizer_source']
validation['token_report_sha256']=sha(PACKAGE/'token_report.json')
validation['files_sha256']={p.name:sha(p) for p in PACKAGE.glob('*.jsonl')}
validation['distribution_note']='Only infrastructure paths were omitted from token metadata. Question content, paired responses, images and review decisions are unchanged.'
write(PACKAGE/'validation_report.json',validation)

generation=read(SRC/'generation_provenance.json')
generation.pop('rejected_action',None)
generation.pop('original_first5_provenance',None)
generation['authoring_inputs']=['selected_questions.jsonl','images/','common_content.jsonl','Team common rules summarized in the repository README.md']
generation['author_code_sha256']=generation.pop('files_sha256')
generation['author_code_note']='Hashes identify author-workspace scripts; scripts and internal team documents are not included in this distribution.'
first5=read(SRC/'provenance/first5_generation_provenance.json')
generation['original_first5_provenance']={k:first5[k] for k in ['prompt_version','model_id','model_revision','generation_settings','human_edits','disclosure','code_sha256','same_author_and_sources_for_both_conditions','independent_generation_or_review_model','gpu_used','remote_actions']}
generation['distribution_changes']=['Excluded private infrastructure addresses and absolute runtime paths','Omitted internal team planning documents and chat-specific messages','Preserved all 50 paired records and images byte-for-byte']
write(PACKAGE/'generation_provenance.json',generation)

package_readme='''D 담당 ScienceQA 50문항

이 폴더의 review.html을 브라우저로 열면 이미지·원문·보기·일반 풀이·ERA 풀이를 함께 볼 수 있습니다.

고유 50문항 / 일반 풀이 50개 / ERA 풀이 50개 / images 50개 / 전부 학습용.
기존 5문항과 추가 45문항을 합쳤습니다. 현재 사람 교차 검수 전 초안입니다.

파일 안내
- dataset_review.csv: 실제 50문항과 두 풀이. UTF-8 BOM. options_json에 원본 보기 배열 보존.
- image_log.csv: 공통 선별 이력 98행(작성완료 50 + 제외 48). 작성완료는 초안 작성 완료를 뜻합니다.
- paired_train.draft.jsonl: 문항, 공통 내용, 두 풀이, 출처, 제작 방법을 포함한 기준 데이터.
- train_plain.draft.jsonl / train_era.draft.jsonl: 각각 50행. 두 조건의 학생 입력은 같습니다.
- images/: 원본 이미지. 모든 이미지 경로는 이 압축을 푼 폴더 기준입니다.
- common_content.jsonl: 공통 내용과 두 풀이의 의미 대응.
- selected_questions.jsonl: 잠금 원본 자료. lecture·solution·answer는 검수 자료이며 학생 입력에 추가하지 않습니다.
- generation_provenance.json: 제작 방법과 알려진 모델 정보.
- validation_report.json / token_report.json / arithmetic_checks.json / 검증요약.json: 검사 결과.
- review_log.jsonl: 제외 사유와 작성 모델 자체 검토 기록.
- image_manifest.json: 이미지 출처와 해시.
- LICENSE-DATA.txt: ScienceQA 공식 데이터 라이선스.
- package_manifest.json: 배포 파일 SHA256 목록.

검증 및 남은 작업
원문·hint·보기·정답·입력 형식·이미지 해시 검사 통과. 최대 답변 길이 103토큰, 본문 길이 차이 최대 15%.
기존 MMMU 900문항/982이미지 대조에서 정확한 일치 없음. 유사 이미지 후보 1건은 직접 비교 후 다른 그림으로 판정했습니다. 전체 팀 문항·별도 개발/평가셋과의 통합 중복 확인은 남아 있습니다.
원본 해설 10126, 10168의 모순/깨진 문구를 검토 메모에 기록했습니다. 이미지 필요성, 판독·정답, 두 풀이의 정보 동등성을 C 담당자가 교차 검수해야 합니다.

제작은 Codex 모델 작성·자가 검토와 Python 검증을 사용했습니다. 정확한 serving model ID와 설정은 노출되지 않아 null로 남겼습니다. Qwen3-VL-4B는 길이 측정용 토크나이저이며 풀이 생성 모델이 아닙니다.

현재는 취합·검수용 배포 파일입니다. 학습 시 messages.content 형식 정규화와 이미지 로딩 처리가 필요합니다. 전체 200문항 취합 후 최종 학습 스키마를 구성합니다. 개발·평가셋은 별도 확보합니다.
출처: https://github.com/lupantech/ScienceQA
원본 데이터 라이선스: CC-BY-NC-SA-4.0. 원본 문제·이미지를 선별하고 일반·ERA 풀이 및 검토 메타데이터를 추가했습니다.
'''
(PACKAGE/'README.txt').write_text(package_readme,encoding='utf-8')

readme='''---
pretty_name: MMDL ERA Dataset
language:
- en
task_categories:
- visual-question-answering
size_categories:
- n<1K
tags:
- multimodal
- scienceqa
- era
- reasoning
---

# MMDL_ERA_Dataset

멀티모달 딥러닝 프로젝트의 일반 풀이와 ERA 풀이 비교를 위한 데이터셋입니다.
담당자별 자료를 취합해 **학습용 200문항**을 구성합니다. 현재 제공 파일은 **D 담당 ScienceQA 50문항**입니다.

## 제공 파일

| 파일 | 내용 | 상태 |
|---|---|---|
| [D_50.zip](./D_50.zip) | ScienceQA 고유 50문항, 일반 풀이 50개, ERA 풀이 50개, 이미지 50개 | 교차 검수 전 초안 |

일반·ERA는 동일한 문제에 대한 두 풀이입니다. 고유 문항 수는 100개가 아닌 50개입니다.
A·B·C 담당 자료는 준비되는 대로 추가합니다.

## 확인 방법

1. `D_50.zip`을 다운로드하고 압축을 풉니다.
2. `review.html`을 브라우저에서 열면 이미지·원문·보기·두 풀이를 함께 볼 수 있습니다.
3. `dataset_review.csv`에서 실제 작성된 50문항을 확인할 수 있습니다.
4. `image_log.csv`는 작성완료 50행과 제외 48행을 포함한 선별 이력입니다. 제외 행을 삭제하거나 이미지 그룹 번호를 다시 매기지 않습니다.

ZIP 안의 `README.txt`에 각 파일의 용도를 정리했습니다. 현재는 파일 다운로드를 통한 취합·검수용 구조이며, Dataset Viewer 및 `load_dataset()` 자동 로딩용 구조는 아직 구성하지 않았습니다.

## 공통 구성 원칙

- 최종 200문항은 전부 학습용입니다. 개발·평가셋은 별도로 확보합니다.
- 같은 문항에 일반 풀이와 ERA 풀이를 모두 작성합니다.
- 일반 풀이는 같은 정보를 자연스럽게 서술하고, ERA는 Evidence·Rule·Application으로 구분합니다.
- ScienceQA의 원본 질문·hint·보기 순서·정답을 유지합니다. ChartQA 자료는 취합 시 4지선다 변환 규칙을 적용합니다.
- 제작 방법을 기록하고, VIC 비율은 강제하지 않습니다.
- 초안 취합 후 중복·풀이 검증과 수정·보충을 거쳐 최종 200문항을 확정합니다.

## D 자료 및 검증

D 자료는 ScienceQA 원본 `train`의 짝수 이미지 그룹에서 선별했으며, 기존 5문항과 추가 45문항을 합쳤습니다. 원본 보기 수는 2~5개로 유지했습니다.

원문·정답·입력 형식·이미지 해시 검증을 통과했습니다. 두 조건의 학생 입력은 같으며, 답변은 모두 512토큰 이내이고 본문 길이 차이는 최대 15%입니다. 기존 MMMU 기준 자료 900문항·982이미지와의 정확한 일치는 발견되지 않았습니다. 전체 팀 자료 및 별도 개발·평가셋과의 중복 확인은 통합 단계에서 진행합니다.

현재 상태는 **초안**입니다. 이미지 필요성, 두 풀이의 정보 동등성, 이미지 판독과 정답을 교차 검수해야 합니다. 원본 해설의 모순·깨진 문구가 발견된 문항 `10126`, `10168`에는 메모를 남겼습니다.

제작 방법·모델 정보는 `generation_provenance.json`과 문항별 메타데이터에 기록했습니다. 작성 모델의 자체 검토와 사람 교차 검수를 구분합니다.

## 출처와 라이선스

- 원본: [ScienceQA](https://github.com/lupantech/ScienceQA)
- 고정 원본 버전: `2cbf8318e07b9ece895bb2ae605e71e38d623264`
- D 자료의 원본 데이터 라이선스: **CC-BY-NC-SA-4.0**. 공식 [LICENSE-DATA](https://github.com/lupantech/ScienceQA/blob/2cbf8318e07b9ece895bb2ae605e71e38d623264/LICENSE-DATA)를 ZIP에 포함했습니다.
- 변경 사항: 원본 문제·이미지를 선별하고 일반·ERA 풀이, 선별 및 검토 메타데이터를 추가했습니다.
- 이미지 출처·버전·해시는 ZIP의 `image_manifest.json`에 기록했습니다. 공식 이미지 아카이브와의 바이트 비교는 수행하지 않았습니다.

추후 추가되는 자료는 각 원본의 출처와 라이선스를 함께 기재합니다. 저장소 전체에 하나의 라이선스를 일괄 지정하지 않습니다.

## 팀 자료 추가

담당자는 `A_50.zip`, `B_50.zip`, `C_50.zip`, `D_50.zip`처럼 담당자를 구분해 올립니다. 각 파일에는 이미지·문항·두 풀이·선별 이력·출처·제작 기록을 포함합니다. 팀 자료를 합친 뒤 최종 학습 스키마와 이미지 로딩 형식을 정리합니다.
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')

# Check preserved content and distribute no local infrastructure identifiers.
assert sha(PACKAGE/'paired_train.draft.jsonl')==sha(SRC/'paired_train.draft.jsonl')
for name in ['train_plain.draft.jsonl','train_era.draft.jsonl']:
 assert len(rows(PACKAGE/name))==50 and sha(PACKAGE/name)==sha(SRC/name)
with (PACKAGE/'dataset_review.csv').open(encoding='utf-8-sig',newline='') as f:assert len(list(csv.DictReader(f)))==50
for p in paired:assert sha(PACKAGE/p['image_path'])==p['image_sha256']
pattern=re.compile(r'(?<![A-Za-z])[A-Za-z]:[\\/]|/home/|sunj0|rlatjswo|100\.109\.154\.30|aix408-40')
for f in list(PACKAGE.rglob('*'))+[OUT/'README.md']:
 if f.is_file() and f.suffix not in ['.png']:
  assert not pattern.search(f.read_text(encoding='utf-8-sig')),f
manifest={'unique_questions':50,'responses':100,'quality_status':'draft','files':{str(p.relative_to(PACKAGE)).replace('\\','/'):sha(p) for p in sorted(PACKAGE.rglob('*')) if p.is_file() and p.name!='package_manifest.json'}}
write(PACKAGE/'package_manifest.json',manifest)
with zipfile.ZipFile(OUT/'D_50.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(PACKAGE.rglob('*')):
  if p.is_file():z.write(p,str(p.relative_to(PACKAGE)).replace('\\','/'))
with zipfile.ZipFile(OUT/'D_50.zip') as z:
 assert z.testzip() is None
 for name,digest in manifest['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
 assert sum(n.startswith('images/') for n in z.namelist())==50
print(json.dumps({'upload_files':[str(OUT/'D_50.zip'),str(OUT/'README.md')],'zip_bytes':(OUT/'D_50.zip').stat().st_size,'archive_files':len(manifest['files'])+1,'questions_unchanged':50,'images':50,'private_infrastructure_scan':'passed'},ensure_ascii=False))
