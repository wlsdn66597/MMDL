"""Publish the single-call comparisons and rule/image/option transitions on sealed results."""
from math import comb

from run import HERE, sha, read


def ratio(tally):
    lower,upper=tally["bounds"]
    return f"{lower}/{tally['n']}" if lower==upper else f"{lower}~{upper}/{tally['n']}"


def main():
    review=HERE/"review"
    summary=read(review/"summary.json")
    decision=read(HERE.parent/"decision.json")
    assert decision["score_sha256"]==sha(review/"summary.json")
    names={"base":"LoRA 전 Qwen","ordinary":"같은 뜻 대조군","preserved":"방향 반전 실험군"}
    md=["# 관계 방향 반전 학습의 단일 호출 평가","",
        f"LoRA 전 모델의 기존 115개 응답을 재사용했고, 두 새 모델에서 230개를 생성했다. 총 345개 응답의 고유 검수 키는 {summary['unique_reviews']}개였고, 그중 {summary['cached_unique']}개는 문항·그림·원문·종료까지 일치해 이미 고정된 판정을 재사용했다.",
        "", "| 모델 | 도표 정답 | 도표 정답+연결 | 8개 모두 연결 성공한 장면 | ScienceQA 정답 | ScienceQA 정답+연결 | 엄격한 ERA 형식 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm,n in names.items():
        s=summary["arms"][arm]
        cols=[ratio(s["charts"][k]) for k in ("answer","joint","scenes")]+[ratio(s["science"][k]) for k in ("answer","joint")]
        md.append("| "+" | ".join([n,*cols,f"{s['strict_era_total']}/115"])+" |")
    md+=["","## 유지·회복 비교","","| 출발 → 도착 | 집합·조건 | 유지 | 회복 | 손실 | 양쪽 실패 | 미판정 |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for src,dst in (("base","ordinary"),("base","preserved"),("ordinary","preserved")):
        for d,n in (("science","ScienceQA"),("charts","도표")):
            for field,m in (("answer","정답"),("joint","정답+연결")):
                c=summary["paired"][src+"_to_"+dst][d][field]
                md.append("| "+" | ".join([names[src]+" → "+names[dst],n+" "+m]+[str(c[k]) for k in ("retained","gained","lost","both_wrong","unknown")])+" |")
    md+=["","## 두 조건을 모두 맞힌 짝",
         "","| 바뀌는 조건 | LoRA 전 Qwen | 같은 뜻 | 방향 반전 |","|---|---:|---:|---:|"]
    for key,label in (("variant","그림의 값"),("rule_index","질문 규칙"),("order","보기 순서")):
        md.append("| "+" | ".join([label]+[ratio(summary["arms"][a]["charts"]["factor_pairs"][key]["joint"]) for a in names])+" |")
    md+=["","## 장면 및 오류","","| 유형 | LoRA 전 Qwen | 같은 뜻 | 방향 반전 |","|---|---:|---:|---:|"]
    for key,label in (("motion","운동"),("operations","손익"),("inspection","합격률")):
        md.append("| "+" | ".join([label]+[ratio(summary["arms"][a]["charts"]["domain_joint"][key]) for a in names])+" |")
    gained=summary["scene_recovered"];lost=summary["scene_lost"]
    md+=["",f"새로 완전히 맞은 장면은 {', '.join(gained) if gained else '0개'}, 잃은 장면은 {', '.join(lost) if lost else '0개'}였다."]
    if all(isinstance(v,bool) for a in ("ordinary","preserved") for v in summary["scene_labels"][a].values()):
        discord=len(gained)+len(lost)
        p=min(1.,2*sum(comb(discord,k) for k in range(min(len(gained),len(lost))+1))/(2**discord))
        md.append(f"12개 짝지은 장면의 양측 정확 McNemar p={p:.3f}이다.")
    md+=["","## 자발적 수정","","| 모델 및 집합 | 첫 E 오류 | 첫 E 미판정 | 오류를 이미지로 고치고 정답 |","|---|---:|---:|---:|"]
    for arm,n in names.items():
        for d,label in (("charts","도표"),("science","ScienceQA")):
            s=summary["arms"][arm][d]["initial_E"]
            md.append(f"| {n} · {label} | {s['wrong']} | {s['unknown']} | {ratio(s['corrected']) if s['wrong'] else '측정 기회 0건'} |")
    md+=["","## 사전 기준",""]
    criteria={"natural_total":"ScienceQA에서 원본 정답 수 이상","retained_original":"원본 정답을 대조군보다 더 많이 유지",
              "joint_8_scene_increase":"8개 변형 연결 성공 장면 수 증가","multi_scene_and_domain":"둘 이상의 유형에서 둘 이상의 장면 회복",
              "preserved_90pct_strict_format":"완전 ERA 104/115 이상"}
    md += [f"- {criteria[k]} – {'충족' if v else '미충족'}" for k,v in summary["tests"].items()]
    md+=["",f"MMMU 진입 기준은 {'충족' if decision['promote'] else '미충족'}이다. ScienceQA 19문항과 12개 도표 장면의 성적으로 900문항 정확도를 추정하지 않는다.",
         "", "[집계](review/summary.json) · [문항별 판단](review/scored_items.jsonl) · [고정 검수 규약](review_protocol.md)"]
    (HERE/"findings.md").write_text("\n".join(md)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
