"""Recompute the released tables from per-item judgments and paired IDs."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("base", "all50", "last50")


def lines(path):
    return [json.loads(s) for s in path.read_bytes().splitlines()]


def display(lo,hi):
    return str(lo) if lo == hi else f"{lo}–{hi}"


def main():
    summary = json.loads((ROOT / "evaluation.json").read_bytes())
    scored = ROOT / "eval/scored.jsonl"
    assert hashlib.sha256(scored.read_bytes()).hexdigest() == summary["source"]
    rows = lines(scored)
    arms = {arm:{r["id"]:r for r in rows if r["arm"] == arm} for arm in ARMS}
    assert len(rows) == 576 and len(arms["base"]) == 192 and all(arms[a].keys()==arms["base"].keys() for a in ARMS)
    receipt = json.loads((ROOT / "manifest.json").read_bytes())
    weights = ROOT / "adapter/adapter_model.safetensors"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == receipt["adapter_file_sha256"]
    report = ["# 마지막 블록 LoRA와 전체 블록 LoRA 비교","",
              "aix408-40의 RTX 4090에서 일반 ERA 296문항을 2 epoch, 38회 업데이트했다. 마지막 블록의 q/k/v/o 어댑터 초기값은 전체 블록 대조군의 같은 블록과 일치한다. 감독 토큰 46,738개와 학습 순서가 같았다. 마지막 블록은 327,680개, 전체 블록은 11,796,480개 파라미터를 학습했다. 대조군은 기존 R9 일반형이다. 두 모델 모두 추론 강도 50%다.","",
              "독립 도표 48쌍과 일반 시각문항 96개에서 각 모델이 ERA 지시 없이 문항당 한 번 답했다. Qwen3-8B가 모델 이름 없이 원장에 대조한 내용 점수와 최종 정답을 결합했다. 연결은 올바른 근거·규칙·적용·중간 결과·결론·정답이 모두 일치한 경우다. 판정이 미정이면 범위를 표시한다.","",
              "| 평가 항목 | LoRA 전 | 전체 블록 | 마지막 블록 |","|---|---:|---:|---:|"]
    def add(name, values):
        report.append("| " + " | ".join([name, *map(str,values)]) + " |")
    for ds, label in (("paired","도표"),("general","일반")):
        measures = [summary["metrics"][a][ds] for a in ARMS]
        for arm,m in zip(ARMS,measures):
            vals = [r for r in arms[arm].values() if r["dataset"] == ds]
            keep = sum(r["answer"] and arms["base"][r["id"]]["answer"] for r in vals)
            assert len(vals) == 96 and keep == m["kept"] and sum(r["answer"] for r in vals) == m["correct"]
        add(f"{label} 정답 / 96",[m["correct"] for m in measures])
        add(f"{label} 기존 정답 유지 / {measures[0]['base_correct']}",[m["kept"] for m in measures])
        add(f"{label} 정답·연결 / 96",[display(*m["connected"]) for m in measures])
        retained_connections = []
        for a in ARMS:
            values = [r["joint"] for r in arms[a].values()
                      if r["dataset"] == ds and arms["base"][r["id"]]["answer"]]
            retained_connections.append(display(sum(v is True for v in values),sum(v is not False for v in values)))
        add(f"{label} 기존 정답을 유지하며 연결 / {measures[0]['base_correct']}",retained_connections)
        if ds == "paired":
            add("짝 양쪽 정답 / 48",[m["both_correct"] for m in measures])
            add("짝 양쪽 정답·연결 / 48",[display(*m["both_connected"]) for m in measures])
        add(f"{label} 보기만 답한 문항 / 96",[m["bare"] for m in measures])
        add(f"{label} 엄격한 ERA 형식 / 96",[sum(r["format"] for r in arms[a].values() if r["dataset"]==ds) for a in ARMS])
    report.extend(["","### 문항별 정답 변화","","| 비교 | 집합 | 유지 | 손실 | 회복 | 양쪽 오답 |","|---|---|---:|---:|---:|---:|"])
    for left,right,title in (("base","all50","LoRA 전 → 전체"),("base","last50","LoRA 전 → 마지막"),("all50","last50","전체 → 마지막")):
        for ds,label in (("paired","도표"),("general","일반")):
            buckets = defaultdict(int)
            for i,r in arms[left].items():
                if r["dataset"]==ds:
                    buckets[(r["answer"],arms[right][i]["answer"])] += 1
            add(title, [label,*[buckets[t] for t in ((True,True),(True,False),(False,True),(False,False))]])
    report.extend(["","### 일반 시각문항 출처별 결과","","| 출처 | 문항 | LoRA 전 정답·연결 | 전체 정답·연결 | 마지막 정답·연결 |","|---|---:|---:|---:|---:|"])
    general = ROOT.parent / "era_half_strength/confirm/general/items.jsonl"
    sources = defaultdict(list)
    for item in lines(general):
        sources[item["source"]].append(item["id"])
    for source, ids in sorted(sources.items()):
        vals=[]
        for a in ARMS:
            seq=[arms[a][i]["joint"] for i in ids]
            vals.append(display(sum(z is True for z in seq),sum(z is not False for z in seq)))
        add(source,[len(ids),*vals])
    report.extend(["","### 풀이 원문 대조",""])
    for ds,label in (("paired","도표"),("general","일반")):
        pairs=[(r,arms["last50"][i]) for i,r in arms["all50"].items() if r["dataset"]==ds]
        gained=[a for a,b in pairs if a["joint"] is False and b["joint"] is True]
        lost=sum(a["joint"] is True and b["joint"] is False for a,b in pairs)
        report.append(f"- {label} 96문항에서 마지막 블록이 전체 블록 대비 연결을 회복한 문항은 {len(gained)}개, 잃은 문항은 {lost}개다. 회복한 문항 중 {sum(a['bare'] for a in gained)}개는 전체 블록이 보기만 답한 문항이다.")
    report.extend(["","예를 들어 `hc-rate-05-1`의 밀봉 건수 최댓값 문제에서 전체 블록은 C만 제출했다. 마지막 블록은 Umber 101, Alder 32, Drift 105, Ridge 34를 열거하고 Drift가 가장 크다고 설명해 B를 골랐다. LoRA 전 모델도 이 문항을 맞혔다. [전체 블록 원문](eval/runs/all50/predictions.jsonl) · [마지막 블록 원문](eval/runs/last50/predictions.jsonl)","",
                   "LoRA 전과 마지막 블록은 도표 96개의 정오와 연결 판정이 문항별로 일치했다. 일반 문항에서 마지막 블록은 LoRA 전보다 연결 2개를 얻고 1개를 잃었다. 명시적인 Evidence·Rule·Application·최종 답 순서를 모두 만족한 문항은 세 모델 모두 0/192개였다."])
    report.extend(["", "### MMMU 진입 판정", ""])
    decision = {"pass":"통과","fail":"미통과","review":"미정 응답 검수 필요"}[summary["decision"]]
    report.append("사전 기준 판정은 **"+decision+"**이다. 두 집합 각각의 기존 정답 유지가 대조군 이상이고, 도표의 양쪽 연결과 일반 문항 연결 모두 대조군 이상이며 하나 이상에서 개선되어야 한다.")
    mmmu = ROOT / "mmmu/last50/full/summary.json"
    if mmmu.exists():
        official=json.loads(mmmu.read_bytes())
        assert official["n"]==900 and official["baseline_correct"]==575
        report.extend(["",f"공식 MMMU 단일 호출 정답은 {official['correct']}/900이다. 기존 정답 575개 중 {575-official['paired_lost']}개를 유지하고 {official['paired_lost']}개를 잃었다. 기존 오답에서 {official['paired_gained']}개를 회복했다. [공식 채점 집계](mmmu/last50/full/summary.json)"])
    elif summary["decision"] == "fail":
        report.extend(["","이 라운드의 마지막 블록 모델에 대한 MMMU 900 점수는 측정하지 않았다."])
    elif summary["decision"] == "pass":
        report.extend(["","공식 MMMU 900문항 최종 점수는 평가 및 채점 완료 후 반영한다."])
    report.extend(["","[사전 기준](protocol.md) · [학습 기록](manifest.json) · [채점 집계](evaluation.json) · [문항별 판정](eval/scored.jsonl) · [정오 전환 ID](transitions.json) · [평가셋 검수](../era_half_strength/confirm/review.md)"])
    (ROOT / "report.md").write_text("\n".join(report)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
