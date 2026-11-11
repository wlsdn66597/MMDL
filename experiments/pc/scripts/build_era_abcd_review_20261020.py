"""Publish local review artifacts from completed audit outputs; no dataset edits."""
import csv
import html
import json
import os
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"reports/era_abcd_review_20261020"
def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))
def rows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def esc(x):
    return html.escape(str(x))
def href(path):
    return os.path.relpath(path,OUT).replace("\\","/")

audit=read(OUT/"audit_full.json")
data={o:rows(Path(audit["archive_identities"][o]["manifest"])) for o in "ABCD"}
byid={r["sample_id"]:r for ps in data.values() for r in ps}
checks={r["sample_id"]:r for r in audit["records"]}
images={r["sample_id"]:r for r in read(OUT/"image_fingerprints_full.json")}
old=read(OUT/"audit.json")
for o in "CD":
    assert audit["archive_identities"][o]["sha256"]==old["identities"][o]["archive_sha256"]
assert len(byid)==200
assert all(r["all_source_checks"] and r["answer_maps_to_option"] for r in audit["records"])
assert all(s["active_ids_match"] and s["group_sequence_covered"] and s["exclusions_have_reason"]
           and s["fingerprint_consistent"] for s in audit["selections"].values())
assert all(c["input_matches_paired_allowlist"] for e in audit["exports"].values() for c in e.get("checks",[]))

issues=[]
def issue(scope, kind, status, detail, action, basis):
    issues.append(dict(scope=scope,kind=kind,status=status,detail=detail,action=action,basis=basis))
issue("C 전체 50", "image_path", "필수 수정",
      "image_path는 train/ID/image.png지만 ZIP에는 images/ID.png만 존재한다. 이미지 파일 자체와 해시는 정상이다.",
      "원본 경로는 original_image_path에 보존하고, 학습 경로와 내보내기 경로를 패키지 기준으로 통일한다.",
      "현재 ZIP·50행·파일 SHA256 직접 검사")
issue("C 전체 50", "metadata_and_exports", "필수 보완",
      "paired 파일에 계획 공통 필드 16종이 빠져 있고 plain/ERA 개별 학습 파일이 없다. 출처·검수 자료 일부는 README·별도 파일에 존재하므로 전체 자료 부재와 구분한다.",
      "source_url/license, original_* 및 검수·태그·중복 상태를 명시적으로 연결하고 최종 공통 스키마로 두 조건을 내보낸다.",
      "필드 존재 검사와 ZIP 파일 목록")
issue("C 빈 hint 33 / D 빈 hint 21", "empty_hint_policy", "통합 시 정규화",
      "C는 빈 Hint 줄을 생략하고 D는 'Hint: ' 줄을 포함한다. A/B는 원출처에 hint가 없어 생략한다. 모두 같은 baseline 템플릿 함수로 표현할 수 있으나 빈 값 표현 정책이 다르다.",
      "누락·빈 hint 처리 정책 하나를 고정하고 두 조건에 동일하게 적용한다. 단순히 A/B의 100행을 잘못된 프롬프트로 분류하지 않는다.",
      "baseline prompt_for를 None 및 빈 문자열로 각각 비교")
issue("ABCD 전체 200", "human_cross_review", "최종 승인 대기",
      "모두 quality_status=draft. B의 작성자 자체 검수 완료는 A의 독립 교차 검수 완료가 아니다. 제출 자료에서 독립 사람 검수 완료 증빙은 확인되지 않았다.",
      "A↔B·C↔D 교차 검수와 수정 후 재검수 기록을 결합한 뒤 accepted 여부를 확정한다.",
      "제출 metadata·review log·validation report")
issue("ABCD 전체", "visual_families", "미완료",
      "정확한 파일·픽셀 중복은 없지만 기존 C/D 등록부에는 9개 복수문항 계열(100행→89계열)이 있다. A/B 관련 지각 해시 후보 346쌍 중 2쌍은 직접 대조하여 다른 차트임을 확인했고 나머지 344쌍은 미판정이다.",
      "원본 계열 등록부에 동일/상이 판정과 근거를 기록한다. 전부 학습용인 현재 반복을 평가 누출로 오인하지 말고, 계열 반복 허용·대체 정책을 명시한다.",
      "C/D 원본 manifest 해시 일치 + 전체 지각 해시 재검사")
issue("ABCD 전체", "external_overlap_and_preflight", "미완료",
      "고정 MMMU validation 900문항·982이미지에서는 정확 질문·정확 픽셀·dHash≤4 또는 pHash≤6 후보가 0개다. 별도 개발 200·평가 300 확정 목록과 processor/chat-template 검증은 없다.",
      "별도 dev/eval 목록으로 문항·계열 중복을 확인하고 실제 processor로 길이·EOS·loss mask를 검사한다.",
      "이번 재검사; 범위 밖 데이터에 대한 누출 부재를 보증하지 않음")
issue("B-chartqa-034", "pair_mismatch", "경미 수정 권고",
      "일반 풀이에는 'Although Iceland and Czechia run close together near the top'이라는 추가 관찰이 있고 ERA와 common_content에는 없다. 정답 C=5는 동일하며 이미지에서 두 선을 확인했다.",
      "추가 관찰을 일반 풀이에서 빼거나 공통 내용·ERA Evidence에도 동일하게 반영한 뒤 의미·길이를 재검사한다.",
      "두 응답 직접 대조 및 현재 이미지 확인")
issue("B-chartqa-068", "era_role", "선택적 보완",
      "1999년 약 687이라는 관찰이 ERA Evidence가 아닌 Application에서 처음 등장한다. 2000년이 최고라는 결론은 이미지·원본과 일치한다.",
      "1999년 관찰을 Evidence로 옮겨 단계 역할을 더 명확히 한다. 이 항목을 정답 오류로 집계하지 않는다.",
      "두 응답·이미지 표본 검토")
issue("B-chartqa-006", "length_exception", "기록된 예외 유지 가능",
      "본문 토큰 길이 차이는 15.5%로 운영 기준을 조금 넘는다. Fiji 값 반복에 따른 차이라는 사유가 제출 보고서에 이미 기록되어 있다.",
      "자동 탈락시키지 않고 사람 교차 검수에서 의미 동등성과 자연스러움을 재확인한다.",
      "동일 Qwen tokenizer 재측정 + B validation_report.length_exceptions")
for r in old["prior_AI_content_flags_still_applicable_to_identical_C_archive"]:
    issue(r["sample_id"],"prior_C_content",r["내용판정"],r["수정·보류사유"],
          "기존 제안에 따라 두 응답·공통 내용·의미 단위를 함께 수정/대체하고 독립 재검수한다.",
          "2026-10-17 AI 검토 재사용; 이번 ZIP SHA256 동일 확인; 새 사람 승인 아님")
issue("Hugging Face README / A·B 일부 기록", "stale_documentation", "수정 권고",
      "원격 README는 A·D 100문항 상태이고, A 첫5행에는 나머지45개 미작성이라는 과거 review_note가 남아 있다. B validation_report에는 과거 tokenizer 설명과 최신 공식 측정이 함께 있어 읽기 혼란이 있다.",
      "최신 취합 수량·검증 범위·미완료 항목과 실제 tokenizer 측정 기준으로 설명을 정리한다.",
      "이전 턴 원격 페이지 확인 + 이번 제출 metadata 읽기")

calculations = [
    dict(sample_id="A_chartqa_g00043", expression="(0.9 + 3.5 + 8.8) / 3",
         result=str((Decimal(".9")+Decimal("3.5")+Decimal("8.8"))/3), expected="4.4", basis="이미지의 Europe/Africa/Americas 값 직접 확인"),
    dict(sample_id="B-chartqa-042",expression="15.4 - 14.22",result=str(Decimal("15.4")-Decimal("14.22")),expected="1.18",basis="현재 두 bar label 직접 확인"),
    dict(sample_id="B-chartqa-092",expression="max(54,63,51,50,39,51,64) - min(38,27,42,42,55,41,27)",
         result=str(max(54,63,51,50,39,51,64)-min(38,27,42,42,55,41,27)),expected="37",basis="현재 이미지의 두 계열 모든 label 확인"),
    dict(sample_id="B-chartqa-096",expression="71-13;57-33;51-31;47-23",
         result=[71-13,57-33,51-31,47-23],expected=[58,24,20,24],basis="현재 이미지 네 국가 favorable/unfavorable 값 확인"),
    dict(sample_id="B-chartqa-016",expression="len([1978,1979,1980,1981,1986,1987,1988])",
         result=7,expected=7,basis="원본 이미지에서 120g 초과 점을 직접 세었음; 경계값은 원본 크기로 재확인 권장"),
]
for c in calculations:
    assert c["result"]==c["expected"]
manual_ids=["B-chartqa-016","B-chartqa-050","B-chartqa-034","B-chartqa-068","B-chartqa-092","B-chartqa-096",
            "A_chartqa_g00043","A_chartqa_g00101","B-chartqa-042","B-chartqa-044","A_chartqa_g00049","B-chartqa-070"]
pair_decisions=[
    dict(left="B-chartqa-042",right="B-chartqa-044",verdict="different",
         reason="Ghana child mortality 1985의 두 추정치 vs Laos/DR Congo chicken yield 1961. 주제·단위·범주·값이 다르고 공통 2-bar 배치만 유사."),
    dict(left="A_chartqa_g00049",right="B-chartqa-070",verdict="different",
         reason="France/Luxembourg/Estonia inequality reduction 2005 vs world school computer access 2016. 서로 다른 3-bar 차트."),
]
for p in pair_decisions:
    for side in ("left","right"):
        p[side+"_sha256"]=images[p[side]]["file_sha256"]
manual = dict(date="2026-10-20", reviewer="AI-assisted review; not human approval",
    text_review_scope="A/B 100문항 질문·보기·일반/ERA 본문 전체 열람. C는 동일 해시의 기존 50문항 검토 결과 재사용. D의 신규 전수 의미 검토는 수행하지 않음.",
    newly_viewed_images=manual_ids,calculation_checks=calculations,near_image_decisions=pair_decisions,
    diagnostic_false_positives_resolved="A 8행은 answer_value와 보기 문자열의 목록/퍼센트 병기 차이. B-050은 [Germany, South Africa]와 'Germany and South Africa'의 표기 차이. 잘못된 정답 변경으로 판정하지 않음.")
save("manual_review_full.json",manual)
save("issues_full.json",issues)
with (OUT/"issues_full.csv").open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(issues[0]));writer.writeheader();writer.writerows(issues)

totals={k:sum(s["tokens"][k]["total"] for s in audit["summaries"].values())
        for k in ("general","era","general_body","era_body")}
totals["response_increase_pct"]=(totals["era"]/totals["general"]-1)*100
totals["body_increase_pct"]=(totals["era_body"]/totals["general_body"]-1)*100
save("review_summary_full.json",dict(total_questions=200,total_responses=400,source_checks_pass=200,
    assignment_pass=200,answer_mapping_pass=200,format_pass=200,archive_status="all_draft",
    chartqa_answer_distribution=dict(Counter(r["answer_letter"] for o in "AB" for r in data[o])),
    tokens=totals,issues=len(issues),unresolved_AB_image_candidate_pairs=344,
    verdict="collection_complete_integration_and_human_review_pending",manual_scope=manual["text_review_scope"]))

head="""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ERA 200문항 구축 계획 검토 · 2026-10-20</title>
<style>body{font:16px/1.65 system-ui,'Malgun Gothic',sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172334;background:#f5f7fa}
h1{font-size:30px;line-height:1.3}h2{margin-top:36px;font-size:23px}.card,details{background:white;border:1px solid #d9e1ea;border-radius:10px;padding:20px;margin:14px 0}
table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px;border:1px solid #d9e1ea;text-align:left;vertical-align:top}th{background:#eaf0f7}
.muted{color:#516174}.verdict{border-left:5px solid #bf791b;background:#fff7e8;padding:18px}a{color:#145a94}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 system-ui}
summary{cursor:pointer;font-weight:650}.twocol{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{max-width:100%;height:auto}.label{font-size:13px;color:#52667c}.small{font-size:14px}
@media(max-width:700px){.twocol{display:block}body{padding:0 12px}table{font-size:12px}}
@media print{body{background:white;margin:0}details{break-inside:avoid}a{color:inherit}}</style><body>
<p class="label">MMDL · 파일럿 데이터 구축 검토 · 2026-10-20 KST</p><h1>ABCD 200문항: 취합 완료, 최종 확정 전 보완 필요</h1>
<div class="verdict">확정 계획 v1.3의 수량·원본 보존·배정·쌍 응답 제작 조건은 확인했습니다.
현재 자료는 모두 draft이며, C의 패키지·내용 보완, 시각 계열 판정, 독립 사람 교차 검수와 학습 전 검증이 남아 있습니다.</div>
<p>기준: <a href="../era_pilot_team_rules_v1_20261014.md">공통 규칙 v1.3</a>.
원본 ZIP과 제출 내용은 수정하지 않았습니다. 사용자 제공 로컬 ZIP 4개를 검사했으며 C/D는 이전에 읽은 원격 SHA256과 일치합니다.
A/B의 원격 SHA256은 별도로 확인하지 않았습니다.</p>"""
parts=[head,"<h2>수량과 자동 검사</h2><table><tr><th>담당</th><th>문항</th><th>원본·배정·정답 대응</th><th>이미지 경로</th><th>응답 형식</th><th>본문 길이 경고</th><th>상태</th></tr>"]
for o,s in audit["summaries"].items():
    parts.append(f'<tr><td>{o}</td><td>50</td><td>{s["source_pass"]}/50 · {s["assignment"]}/50 · {s["answer_maps_to_option"]}/50</td>'
                 f'<td>{s["image_path_valid"]}/50</td><td>{s["response_format"]}/50</td><td>{s["length_review_flag"]}</td><td>draft 50</td></tr>')
parts.append("""</table><p>ChartQA 100 + ScienceQA 100 = 고유 원본 ID 200개. 일반 응답 200개 + ERA 응답 200개.
ChartQA는 모두 4지선다이며 정답 위치는 A 26 / B 25 / C 24 / D 25입니다. ScienceQA는 원래 보기와 순서를 유지합니다.
200문항 모두 train 출처이며 응답은 영어, 최종 답 줄과 ERA 표제는 400개 응답 전체에서 정상입니다.</p>
<p>각 담당의 최신 image_log는 작성완료 50행과 제외 A 6 / B 10 / C 43 / D 48행을 포함합니다.
작성완료 ID가 제출과 일치하고 홀짝 순서의 그룹 이력·제외 사유가 보존되어 있습니다.</p>
<h2>우선 처리할 항목</h2>""")
for x in issues:
    parts.append(f'<div class="card"><b>{esc(x["scope"])} · {esc(x["status"])}</b><p>{esc(x["detail"])}</p>'
                 f'<p><b>조치:</b> {esc(x["action"])}</p><p class="label">근거: {esc(x["basis"])}</p></div>')
parts.append(f"""<h2>길이·노출량</h2><table><tr><th>범위</th><th>일반</th><th>ERA</th><th>증가</th></tr>
<tr><td>전체 응답 토큰</td><td>{totals['general']:,}</td><td>{totals['era']:,}</td><td>{totals['response_increase_pct']:.2f}%</td></tr>
<tr><td>표제·최종 답 줄 제외 본문</td><td>{totals['general_body']:,}</td><td>{totals['era_body']:,}</td><td>{totals['body_increase_pct']:.2f}%</td></tr></table>
<p>동일한 Qwen3-VL-4B-Instruct tokenizer JSON(sha256: {esc(audit['tokenizer_sha256'])}) 사용.
응답 최대 189토큰, 512 초과 0개입니다. 전체 응답에는 ERA 표제 비용이 포함됩니다.
본문은 각 표제와 마지막 Final answer 줄을 제거하되 나머지 개행을 유지했습니다. 이전 제출 보고서와 문자열 정규화 방식에 따른 차이가 있을 수 있습니다.
실제 chat template·이미지 토큰·EOS·loss mask 검사는 포함하지 않았습니다.</p>
<h2>중복 검사 범위와 해석</h2>
<p>팀 내부 200문항의 원본 ID 및 정확 파일/픽셀 중복은 0개입니다.
MMMU validation 900문항·982이미지를 모두 읽어 정확 질문, 정확 픽셀, dHash≤4 또는 pHash≤6을 비교했고 후보 0개였습니다.
단어 n-gram·OCR·crop·원본 계열 계보까지 포함한 완전한 누출 검증은 아닙니다.</p>
<p>C/D는 기존 이미지 계열 판정과 이번 원본 해시가 일치하므로 89계열·복수문항 9계열 기록을 이어 쓸 수 있습니다.
A/B 관련 346쌍은 주로 같은 OWID 레이아웃에 따른 pHash 후보입니다. 직접 본 두 쌍은 주제·국가·단위가 다른 차트였고 나머지는 미판정입니다.
후보 수를 중복 수로 집계하지 않았고, 전체 200문항의 최종 시각 계열 수는 확정하지 않았습니다.</p>
<h2>내용 검토의 실제 범위</h2><p>{esc(manual['text_review_scope'])}</p>
<p>이번에 원본 크기로 열람한 이미지는 12개입니다. 별도 정답을 숨긴 신규 전수 blind review는 수행하지 않았습니다.
표본 계산은 (0.9+3.5+8.8)/3=4.4, 15.4−14.22=1.18, 64−27=37, 네 국가 차이 [58,24,20,24]와 일치했습니다.
따라서 자동 형식 통과를 200문항 의미·시각 근거의 최종 승인으로 해석하면 안 됩니다.</p>
<h2>재검토 순서</h2><ol>
<li>C 이미지 경로·누락 메타데이터를 보완하고 두 조건의 공통 내보내기 스키마를 정한다.</li>
<li>C의 기존 9문항 지적과 B-034 정보 차이를 처리하고 B-006 길이 예외를 유지할지 검수한다.</li>
<li>원본 이미지 계열 등록부를 완성하고 중복 계열의 유지·대체 결정을 기록한다. 대체 시 같은 선별 규칙을 따른다.</li>
<li>A↔B·C↔D 사람 교차 검수와 별도 dev/eval 중복 검사를 완료한다.</li>
<li>실제 processor·chat template 검증 후 동일한 200 ID로 plain/ERA 파일을 고정하고 README를 갱신한다.</li></ol>
<h2>검증 파일</h2><ul>
<li><a href="audit_full.json">200문항 자동 검사 JSON</a> · <a href="item_checks_full.csv">문항별 검사 CSV</a></li>
<li><a href="issues_full.csv">수정·보완 목록 CSV</a> · <a href="manual_review_full.json">AI 내용 검토 범위·계산·판정 근거</a></li>
<li><a href="mmmu_overlap_full.json">MMMU 검사 범위·결과</a> · <a href="image_candidates_full.json">내부 이미지 후보</a></li>
<li><a href="../../scripts/audit_era_abcd_full_20261020.py">재현용 검증 코드</a></li></ul>
<h2>200문항 원문·두 응답 확인</h2><p class="muted">아래 상태는 제출 상태이며 이 보고서가 새로 승인한 상태가 아닙니다. 펼쳐서 실제 이미지와 응답을 비교할 수 있습니다.</p>""")
for o in "ABCD":
    parts.append(f"<h3>{o} · 50문항</h3>")
    for p in data[o]:
        sid=p["sample_id"]; c=checks[sid]
        parts.append(f'<details><summary>{esc(sid)} · {esc(p["question"])} · 답 {esc(p["answer_letter"])}</summary>'
            f'<p class="small">상태: draft · 일반 {c["tokens"]["general"]} / ERA {c["tokens"]["era"]} 토큰 · 본문 차이 {c["body_difference"]:.1%}</p>'
            f'<img loading="lazy" src="{esc(href(Path(images[sid]["image_path"])))}" alt="{esc(sid)} 원본 이미지">'
            f'<pre>{esc(p["input_prompt"])}</pre><div class="twocol"><div><b>일반</b><pre>{esc(p["response_general"])}</pre></div>'
            f'<div><b>ERA</b><pre>{esc(p["response_era"])}</pre></div></div></details>')
parts.append("</body></html>")
(OUT/"review_full.html").write_text("\n".join(parts),encoding="utf-8")
print(json.dumps({"summary":read(OUT/"review_summary_full.json"),"html":str(OUT/"review_full.html")},ensure_ascii=True,indent=2))
