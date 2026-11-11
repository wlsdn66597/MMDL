"""Build the round report from frozen grades and official comparison receipts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARMS=["base","ordinary","enhanced"]
LABEL={"base":"LoRA 전 Qwen","ordinary":"일반 ERA","enhanced":"E·R 보강 ERA"}


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def band(v,n=None):
    amount=str(v[0]) if v[0]==v[1] else f"{v[0]}~{v[1]}"
    return amount+(f"/{n}" if n is not None else "")


def main():
    scorefile=ROOT/"eval/review/summary.json"
    summ=read(scorefile);p=read(ROOT/"decision.json")
    assert p["summary_sha256"]==digest(scorefile)
    arms=summ["arms"]
    txt=["# 일반 ERA와 근거·규칙 보강 ERA의 단일 호출 비교",""]
    for arm in ("ordinary","enhanced"):
        assert arms[arm]["paired"]["n"]==96 and arms[arm]["general"]["n"]==96
    control,enh=arms["ordinary"],arms["enhanced"]
    txt.extend([f"짝의 두 질문 모두에서 정답과 E·R·A 연결이 맞은 독립 이미지 그룹은 일반형 {band(control['groups']['all'],48)}, 보강형 {band(enh['groups']['all'],48)}이었다. "
                f"독립 일반 문항의 정답은 LoRA 전 {band(arms['base']['general']['answer'],96)}, 일반형 {band(control['general']['answer'],96)}, 보강형 {band(enh['general']['answer'],96)}이었다.",
                "", "| 지표 | LoRA 전 | 일반 ERA | E·R 보강 ERA |","|---|---:|---:|---:|"])
    for ds,k,label in [("paired","answer","짝문항 정답"),("paired","joint","짝문항 정답과 연결"),("general","answer","일반 정답"),
                       ("general","joint","일반 정답과 연결"),("charts","answer","기존 도표 정답"),("charts","joint","기존 도표 정답과 연결"),
                       ("science","answer","기존 ScienceQA 정답"),("science","joint","기존 ScienceQA 정답과 연결")]:
        txt.append("| "+label+" | "+" | ".join(band(arms[a][ds][k],arms[a][ds]["n"]) for a in ARMS)+" |")
    for k,label,den in (("all","짝 모두 연결",48),("E","E 선택 짝 모두 연결",12),("R","R 변경 짝 모두 연결",12),
                        ("A","A 적용 짝 모두 연결",12),("choice","보기 변경 짝 모두 연결",12)):
        txt.append("| "+label+" | "+" | ".join(band(arms[a]["groups"][k],den) for a in ARMS)+" |")
    txt.extend(["","## 기존 정답 유지","", "| 평가 집합 | 비교 | 유지 | 회복 | 손실 | 양쪽 오답 |","|---|---|---:|---:|---:|---:|"])
    for ds,lbl in (("general","독립 일반"),("paired","독립 짝문항"),("charts","기존 도표"),("science","기존 ScienceQA")):
        for left,right in (("base","ordinary"),("base","enhanced"),("ordinary","enhanced")):
            c=summ["changes"][ds][f"{left}-{right}-answer"]
            txt.append(f"| {lbl} | {LABEL[left]} → {LABEL[right]} | {c['kept']} | {c['gained']} | {c['lost']} | {c['both_wrong']} |")
    txt.extend(["","## 처음 읽은 잘못된 근거를 고쳤는가","",
                "첫 E 오류 뒤에 같은 응답에서 실제 그림 근거를 다시 읽고 정답까지 도달한 수를 센다. 분모는 첫 E가 틀렸다고 확인한 수다.","",
                "| 평가 집합 | LoRA 전 | 일반 ERA | E·R 보강 ERA |","|---|---:|---:|---:|"])
    for ds,lbl in (("paired","독립 짝"),("general","독립 일반"),("charts","기존 도표"),("science","기존 ScienceQA")):
        txt.append("| "+lbl+" | "+" | ".join(band(arms[a][ds]["repaired"],arms[a][ds]["first_errors"]) if arms[a][ds]["first_errors"] else "기회 0건" for a in ARMS)+" |")
    txt.extend(["","## 학습 및 판정","",
        "VIC 원본 24장(C 15장, V 9장)마다 두 질문을 만들어 각 군을 296문항·38회 업데이트로 훈련했다. 같은 원본 Qwen과 같은 LoRA 초기값, 순서, seed 3407을 사용했다. "
        "36문항에서만 E·R이 바뀌었고 A와 최종 답은 모두 같았다. 변경 문항의 지도 토큰 합은 일반형 1,987개, 보강형 1,999개다. "
        "[학습 기록](train/training_report.md) · [훈련 이미지와 질문](data/source_review.md)","",
        "각 모델의 독립 192문항 단일 호출과 기존 115문항의 결과를 함께 비교했다. 새로 생성한 응답은 LoRA 전 192개와 학습군 각각 307개이고, LoRA 전 기존 115개는 이미지·프롬프트·해시를 대조해 재사용했다. "
        "별도 Qwen3-8B 평가기는 모델명을 가린 원문과 참조 증거를 읽었다. 감사 대상으로 정한 응답은 사람이 구축한 근거표와 원그림을 사용해 에이전트가 재검수했다. "
        "일반 문항은 ScienceQA 72개와 ChartQA 24개이며, 일부 ScienceQA는 힌트와 사전 지식으로도 풀 수 있다. "
        "[사전 규약](protocol.md) · [모델별 문항 판정](eval/review/scored.jsonl) · [익명 검수 보정](eval/review/manual.jsonl)","",
        "## MMMU 진입 판정",""])
    labels={"all":"독립 전체 짝 증가","E":"E 선택 짝 증가","R":"R 변경 짝 증가",
            "general_vs_base":"일반 정답이 LoRA 전 이상","base_kept_vs_control":"일반 원본 정답 유지가 대조군 이상"}
    txt.extend(f"- {labels[k]} {'충족' if ok else '미충족'}" for k,ok in summ["tests"].items())
    if p["promote"]:
        scope=ROOT/"mmmu/enhanced/full/summary.json"
        if scope.exists():
            raw=read(scope);assert raw["n"]==900 and raw["baseline_correct"]==575
            txt.extend(["",f"공식 MMMU는 보강형 {raw['correct']}/900, 원본 575/900이다. 원본 정답 손실 {raw['paired_lost']}개, 오답 회복 {raw['paired_gained']}개였다. [공식 결과](mmmu/enhanced/full/summary.json)"])
        else:txt.extend(["","개발 기준 통과. 공식 MMMU 전체 점수는 아직 미확정이다."])
    else:
        txt.extend(["","보강형의 개발 진입 기준이 충족되지 않아 MMMU를 실행하지 않았다. 이번 모델의 575/900 유지 여부는 미측정이다."])
    txt.extend(["","[변화가 난 문항의 그림과 풀이](cases.md) · [집계 원본](eval/review/summary.json) · [진입 결정](decision.json)"])
    supplement=ROOT/"eval/review/general_sensitivity.json"
    if supplement.is_file():
        qc=read(supplement)
        txt.extend(["","## 일반 문항의 내용 판정 민감도","",
            "일반 평가에서 양 군 연결 판정이 다르거나 정답인데 연결이 실패한 것으로 채점된 사례 등의 10문항을 선별하고, 앞선 수동 검수와 보기 전용 응답을 제외한 27개 익명 원문을 추가로 원그림과 대조했다. "
            "이 감사는 공식 정답 및 진입 조건에 쓰인 수를 바꾸지 않는다. 위의 표는 실험 종료 시 봉인된 집계이며, 아래는 추가 감사 뒤의 일반 내용 연결과 첫 E 수정 수다.","",
            "| 일반 문항 | LoRA 전 Qwen | 일반 ERA | 보강형 |","|---|---:|---:|---:|",
            "| 최종 정답 | "+" | ".join(f"{qc[a]['answer']}/96" for a in ARMS)+" |",
            "| 정답과 연결 | "+" | ".join(band([qc[a]['joint'],qc[a]['joint']+qc[a]['unknown']],96) for a in ARMS)+" |",
            "| 첫 E 오류에서 그림을 다시 읽어 정답 | "+" | ".join(band([qc[a]['repaired'],qc[a]['possible_repaired']],qc[a]['first_errors']) if qc[a]['first_errors'] else "기회 0건" for a in ARMS)+" |","",
            "[추가 판정 및 문항별 이유](eval/review/general_extra_labels.jsonl) · [재집계](eval/review/general_sensitivity.json)"])
    (ROOT/"report.md").write_text("\n".join(txt)+"\n",encoding="utf8")


if __name__=="__main__":main()
