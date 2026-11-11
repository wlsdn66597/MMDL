"""Report the observed comparison, including unresolved labels and the conditional MMMU result."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARMS=("base","ordinary","enhanced")
NAMES={"base":"LoRA 전 Qwen","ordinary":"일반 ERA","enhanced":"R 보강 ERA"}
def read(p):return json.loads(p.read_text(encoding="utf8"))
def band(values,n):
    a,b=values
    return (str(a) if a==b else f"{a}~{b}")+"/"+str(n)
def grouped(s,a,field,kind=None):
    vals=[g[field] for g in s["groups"][a].values() if kind is None or g["kind"]==kind]
    return band([sum(x is True for x in vals),sum(x is not False for x in vals)],len(vals))
def tab(name,values):return "| "+name+" | "+" | ".join(values)+" |"


def main():
    s=read(ROOT/"eval/review/summary.json");decision=read(ROOT/"decision.json")
    metrics=s["metrics"];control=metrics["ordinary"];enh=metrics["enhanced"]
    tx=["# 일반 ERA와 R 문장만 보강한 ERA 비교","",
        f"두 질문 모두에서 규칙을 맞힌 독립 그룹은 일반형 {grouped(s,'ordinary','r')}, 보강형 {grouped(s,'enhanced','r')}이다. "
        f"정답과 전체 ERA 연결까지 모두 맞힌 그룹은 일반형 {grouped(s,'ordinary','joint')}, 보강형 {grouped(s,'enhanced','joint')}이다.",
        "", "| 단일 호출 평가 | LoRA 전 | 일반 ERA | R 보강 ERA |","|---|---:|---:|---:|"]
    for lbl,f in (("짝 양쪽 R 선택","r"),("짝 양쪽 정답과 연결","joint"),("짝 양쪽 최종 정답","answer")):
        tx.append(tab(lbl,[grouped(s,a,f) for a in ARMS]))
    for ds,f,lbl in (("paired","r","규칙 문항의 R 선택"),("paired","intermediate","규칙 문항의 중간값·연산"),("paired","winner","규칙 문항에서 결론 내용 도출"),("paired","answer","규칙 문항의 최종 정답"),("paired","joint","규칙 문항 정답과 연결"),
                     ("general","answer","일반 문항의 최종 정답"),("general","joint","일반 문항 정답과 연결"),("sentinel","answer","보존 전용 문항 정답")):
        tx.append(tab(lbl,[band(metrics[a][ds][f],metrics[a][ds]["n"]) for a in ARMS]))
    tx+=["","## 유형별 변화","",
         "| 규칙 유형 | 일반형 두 R | 보강형 두 R | 일반형 두 연결 | 보강형 두 연결 | 연결 회복 / 손실 |",
         "|---|---:|---:|---:|---:|---:|"]
    labels={"rate":"건수·비율","change":"순변화·총변화량","route":"직행·조건부 환승","mixture":"동일 비중·가중 혼합"}
    diff=s["pair_changes"]["joint"]
    for kind in sorted(set(v["kind"] for v in s["groups"]["ordinary"].values())):
        n=lambda f:sum(s["groups"]["ordinary"][g]["kind"]==kind for g in diff[f])
        tx.append(tab(labels.get(kind,kind),[grouped(s,a,f,kind) for f in ("r","joint") for a in ("ordinary","enhanced")]+[f"{n('recovered')} / {n('lost')}"]))
    tx+=["","## 기존 정답 유지","",
          "| 집합 | 기준 → 후보 | 유지 | 회복 | 손실 | 양쪽 오답 |","|---|---|---:|---:|---:|---:|"]
    for ds,lbl in (("general","일반"),("sentinel","보존 전용"),("paired","규칙 짝문항")):
        for old,new in (("base","ordinary"),("base","enhanced"),("ordinary","enhanced")):
            d=s["changes"][ds][f"{old}-{new}-answer"]
            tx.append(tab(lbl,[NAMES[old]+" → "+NAMES[new],*[str(len(d[k])) for k in ("kept","recovered","lost","both_wrong")]]))
    t=read(ROOT/"train/data/receipt.json")["token_stats"]["modified24"]
    tx+=["","## 자료·학습·판정","",
        f"두 군은 공통 248건과 새 이미지 24장의 48건을 합쳐 296문항으로 같은 초기 Qwen에서 2 epoch, 38회 업데이트했다. "
        f"24문항의 R만 바뀌었으며 E·A·최종 답 및 보기까지 동일했다. 변경 문항의 epoch당 지도 토큰은 일반형 {t['ordinary']['total']:,}개, 보강형 {t['enhanced']['total']:,}개였다. "
        "[학습 검증](train/training_report.md) · [훈련 표적](data/rows.jsonl)","",
        "세 모델은 독립 입력 192개마다 한 번씩, 총 576회 답했다. 답 추출과 내용 평가는 별도이며 참조 근거를 받은 Qwen3-8B의 익명 판정을 사용했다. "
        "짝문항의 근거가 있는 모든 고유 응답, 일반 문항의 보존 사례·불확실한 추출 및 차이가 난 사례 등을 원그림과 대조했다. 보기만 낸 경우는 연결 성공에서 제외했다. "
        "짝 자료는 4개 규칙 유형과 유형별 두 시각적 형식의 신규 생성 자료로, 한 시드의 비교다. [사전 규약](protocol.md) · [원문 검수](eval/review/manual.jsonl) · [독립 원장](paired/sources.jsonl)","",
        "## MMMU 진입 조건",""]
    names=dict(rule_pairs="두 R 선택 그룹 증가",connected_pairs="두 연결 성공 그룹 증가",rule_and_joint_leave_one="한 그룹씩 제외해도 두 증가 유지",
               recovered_families="두 유형 이상에서 완전 연결 그룹 회복",general_at_base="일반 정답 원본 수준 이상",
               baseline_retained="원본 정답 유지가 일반형 이상",sentinel_no_new_loss="보존 전용 일반형 정답 손실 0")
    tx.extend(f"- {names[k]} {'충족' if v else '미충족'}" for k,v in s["tests"].items())
    if decision["promote"]:
        src=ROOT/"mmmu/enhanced/full/summary.json"
        if src.is_file():
            stat=read(src);assert stat["n"]==900 and stat["baseline_correct"]==575
            tx+=["",f"공식 MMMU는 {stat['correct']}/900이고 원본은 575/900이다. 원본 정답을 {575-stat['paired_lost']}개 유지하고 {stat['paired_lost']}개 잃었으며 {stat['paired_gained']}개를 회복했다. [공식 채점](mmmu/enhanced/full/summary.json)"]
        else:tx+=["","MMMU 진입 조건을 충족했다. 전체 점수는 공식 결과에 기록된다."]
    else:tx+=["","사전 기준 미충족으로 MMMU는 실행하지 않았다. 이번 모델의 575/900 유지 여부는 미측정이다."]
    if (ROOT/"cases.md").is_file():tx+=["","[변화한 사례와 실제 풀이](cases.md)"]
    tx+=["","[점수 집계](eval/review/summary.json) · [독립 재계산 및 결정](decision.json)"]
    (ROOT/"report.md").write_text("\n".join(tx)+"\n",encoding="utf8")
if __name__=="__main__":main()
