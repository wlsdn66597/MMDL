"""Summarize frozen development scores after all four blind review files were sealed."""
from pathlib import Path
import json
import math

import score

HERE=Path(__file__).resolve().parent
REVIEW=HERE/"review"
NAMES={"base":"LoRA 전 원본","ordinary":"일반 ERA","preserved":"중간 계산 보존 ERA"}
DOMAINS={"inspection":"합격률","operations":"손익표","motion":"운동 그래프"}


def rate(value):
    if value["unknown"]:
        return f"{value['bounds'][0]}~{value['bounds'][1]}/{value['n']}"
    return f"{value['yes']}/{value['n']}"


def test_pair(a,b):
    if a+b==0:return 1
    return min(1,2*sum(math.comb(a+b,i) for i in range(min(a,b)+1))/(2**(a+b)))


def main():
    overall=score.load(REVIEW/"summary.json")
    rows=score.lines(REVIEW/"scored_items.jsonl")
    assert len(rows)==345 and overall["n_responses"]==345
    sensitivity_key="b3d2d45592e41994d256"
    alt=[dict(row,consistent=True,joint=row["answer"] is True and row["E"] is True and row["R"] is True and row["A"] is True) if row["key"]==sensitivity_key else row for row in rows]
    assert any(r["key"]==sensitivity_key for r in rows)
    sb=score.summary(alt)
    # Alternate replaces one disputed consistency label only; all recorded official grades stay fixed.
    study=dict(key=sensitivity_key,alternative_gate=sb["gate"],alternative_tests=sb["tests"],
               alternative_scene_counts={k:v["charts"]["scenes"] for k,v in sb["arms"].items()})
    score.jsonout(REVIEW/"tamar_sensitivity.json",study)
    md=["# 중간 계산 보존 ERA 개발 평가","",
        f"세 모델이 도표 96개와 ScienceQA 19개에 한 번씩 답했다. 총 345개 응답 중 문항·내용·종료 상태가 같은 사례를 합친 {overall['unique_reviews']}개를 모델명을 가리고 검수했다.",
        "", "| 모델 | ScienceQA 정답 | ScienceQA 정답+연결 | 도표 정답 | 도표 정답+연결 | 8개 모두 정답+연결인 장면 | 완전 ERA |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in score.ARMS:
        a=overall["arms"][arm]
        md.append(f"| {NAMES[arm]} | {rate(a['science']['answer'])} | {rate(a['science']['joint'])} | {rate(a['charts']['answer'])} | {rate(a['charts']['joint'])} | {rate(a['charts']['scenes'])} | {a['strict_era_total']}/115 |")
    md+=["","## 같은 문항에서 유지·회복·손실","",
         "| 문항·출발점 → 비교군 | 기존 정답 유지 | 오답 회복 | 정답 손실 | 양쪽 오답 | 미확정 |","|---|---:|---:|---:|---:|---:|"]
    for name,cmp in overall["paired"].items():
        left,right=name.split("_to_")
        for ds,dslabel in (("science","ScienceQA"),("charts","도표")):
            v=cmp[ds]["answer"]
            md.append(f"| {dslabel} · {NAMES[left]} → {NAMES[right]} | {v['retained']} | {v['gained']} | {v['lost']} | {v['both_wrong']} | {v['unknown']} |")
    md+=["","ScienceQA에서 원본과 일반 ERA가 모두 맞히고 보존 ERA가 잃은 두 사례를 원문과 그림으로 확인했다.",
         "",
         "- 화석층 문항 984에서 보존 모델은 E에 깃털이 맨 아래층이고 알이 중간층이라고 적었고, 낮은 층이 오래됐다는 R도 제시했다. A에서는 깃털층이 알보다 위라고 바꿔 깃털이 더 젊다고 골랐다.",
         "- 퍼넷 사각형 19041에서는 우성 표현형 3칸과 열성 표현형 1칸을 맞게 셌고, 정상 귀와 덤보 귀 순서로 3:1을 선택했다. 원질문의 순서는 덤보 귀와 정상 귀여서 1:3이 정답이었다."]
    md+=["","## 도표에서 어떤 연결이 달라졌나","",
         "| 항목 | LoRA 전 | 일반 ERA | 보존 ERA |","|---|---:|---:|---:|"]
    for field,title in (("E","이미지 판독 E"),("R","규칙 R"),("A","실제 적용 A"),("consistent","미수정 모순 없음")):
        md.append("| "+title+" | "+" | ".join(rate(overall["arms"][arm]["charts"][field]) for arm in score.ARMS)+" |")
    for k,t in (("variant","이미지 수치 변경 양쪽 연결 성공"),("rule_index","질문 규칙 변경 양쪽 연결 성공"),("order","보기 순서 변경 양쪽 연결 성공")):
        md.append("| "+t+" | "+" | ".join(rate(overall["arms"][arm]["charts"]["factor_pairs"][k]["joint"]) for arm in score.ARMS)+" |")
    for d,title in DOMAINS.items():
        md.append("| "+title+" · 정답이고 연결 성공 | "+" | ".join(rate(overall["arms"][arm]["charts"]["domain_joint"][d]) for arm in score.ARMS)+" |")
    md+=["",f"여덟 입력 모두에서 연결 성공한 장면을 일반 ERA와 비교해 보존 모델이 회복한 장면은 {', '.join(overall['scene_recovered']) or '0개'}이고, 잃은 장면은 {', '.join(overall['scene_lost']) or '0개'}다."]
    if not any(v is None for a in score.ARMS for v in overall["scene_labels"][a].values()):
        md.append(f"장면 단위 양측 정확 McNemar p={test_pair(len(overall['scene_recovered']),len(overall['scene_lost'])):.3f}. 독립 장면은 12개다.")
    md+=["","## 자발적인 이미지 근거 수정","",
         "| 모델 | 집합 | 첫 판독 오류 | 첫 판독 미확정 | 오류 중 이미지로 고치고 정답 |","|---|---|---:|---:|---:|"]
    for arm in score.ARMS:
        for ds,dsname in (("charts","도표"),("science","ScienceQA")):
            ini=overall["arms"][arm][ds]["initial_E"]
            rate_txt=rate(ini["corrected"]) if ini["wrong"] else "측정 기회 0건"
            md.append(f"| {NAMES[arm]} | {dsname} | {ini['wrong']} | {ini['unknown']} | {rate_txt} |")
    md+=["","## 사전 기준",""]
    tests={"natural_total":"ScienceQA에서 원본 이상의 정답 수","retained_original":"일반 ERA보다 더 많은 원본 정답 유지",
           "joint_8_scene_increase":"일반 ERA보다 더 많은 완전 연결 장면","multi_scene_and_domain":"두 과제 이상에서 두 장면 이상 회복",
           "preserved_90pct_strict_format":"115응답 중 104개 이상 완전 ERA"}
    md.extend(f"- {label} {'통과' if overall['tests'][key] else '미충족'}" for key,label in tests.items())
    md+=["",f"MMMU 진입 판정은 {'통과' if overall['gate'] else '미충족'}이다. Tamar의 540과 468을 함께 적은 응답을 일관된 수정으로 대체 처리한 민감도 점검에서도 진입 판정은 {'통과' if sb['gate'] else '미충족'}이다.",
         "","도메인별로 한 명의 Codex 검수자가 같은 원본 이미지와 참고값을 사용했다. 독립 전문가의 이중 판정은 수행하지 않았다. 작은 ScienceQA 19개·도표 12장면의 결과로 MMMU 정답률을 추정하지 않는다.",
         "","[익명 검수 규약](review_protocol.md) · [집계](review/summary.json) · [응답별 판정](review/scored_items.jsonl) · [민감도 기록](review/tamar_sensitivity.json)"]
    score.dump(HERE/"findings.md","\n".join(md)+"\n")


if __name__=="__main__":
    main()
