"""Render the Korean comparison from sealed training, item-level grades and gate receipts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARMS=("base","ordinary","enhanced")
NAMES={"base":"LoRA 전","ordinary":"일반 ERA","enhanced":"A 보강 ERA"}
KINDS={"rate":"건수·비율","change":"순변화·총변화량","route":"이동 경로","mixture":"동일 비중·가중 혼합"}
GATES={
    "connected_pairs":"양쪽 연결 그룹 순증",
    "joint_leave_one":"어느 한 그룹을 빼도 연결 순증",
    "recovered_families":"적어도 두 규칙 유형에서 연결 회복",
    "rule_pairs_noninferior":"양쪽 R 선택 그룹 수가 일반형 이상",
    "er_pairs_noninferior":"양쪽 E·R 성공 그룹 수가 일반형 이상",
    "common_er_connected":"공통 E·R 성공 그룹에서 연결 순증",
    "paired_baseline_retained":"규칙 문항에서 유지한 원본 정답 수가 일반형 이상",
    "general_baseline_retained":"일반 문항에서 유지한 원본 정답 수가 일반형 이상",
    "general_at_base":"일반 96문항의 총 정답 수가 LoRA 전 이상",
    "original_general_no_new_loss":"원본과 일반형이 모두 맞힌 일반 문항의 신규 손실 0",
    "sentinel_no_new_loss":"보존 전용 16문항에서 일반형 정답 신규 손실 0",
}
ERRORS={"candidate_missing":"필요한 후보 누락","other_evidence":"기타 근거 오류","no_grounded_answer":"근거 없는 답",
    "wrong_rule":"규칙 선택","operands":"피연산자 연결","arithmetic":"산술","loser_rounding":"비승자 근삿값",
    "winner_omitted":"결론 생략","wrong_winner":"결론 오류","wrong_letter":"보기 대응","other":"기타","uncertain":"미정","none":"모두 통과"}
TURN=("kept","recovered","lost","both_wrong","unknown")
DS={"paired":"규칙 문항","general":"일반 문항","sentinel":"보존 전용"}

def read(path):
    return json.loads(path.read_text(encoding="utf8"))

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def mdrow(*cols):
    return "| "+" | ".join(map(str,cols))+" |"

def band(span,total):
    lo,hi=span
    if total==0:
        return "해당 없음"
    return f"{lo}/{total}" if lo==hi else f"{lo}~{hi}/{total}"

def subset_span(flags):
    flags=list(flags)
    return [sum(v is True for v in flags),sum(v is not False for v in flags)]

def grouped(s,arm,metric):
    group=s["groups"][arm]
    return band(subset_span(record[metric] for record in group.values()),len(group))

def indices(ids):
    return ", ".join(f"`{i}`" for i in sorted(ids)) or "없음"

def grade_table(s):
    tx=["| 단일 호출 지표 | LoRA 전 | 일반 ERA | A 보강 ERA |","|---|---:|---:|---:|"]
    for label,f in (("양쪽 R 선택","r"),("양쪽 E·R 성공","er"),("양쪽 최종 정답·ERA 연결","joint"),
                    ("양쪽 최종 정답","answer"),("비승자 반올림 허용 시 양쪽 연결","rounding_joint")):
        tx.append(mdrow(label,*[grouped(s,a,f) for a in ARMS]))
    for ds,f,label in (("paired","intermediate","규칙 문항의 중간 연산"),("paired","winner","규칙 문항의 결론 도출"),
                       ("paired","a","규칙 문항의 A"),("paired","joint","규칙 문항의 정답·ERA 연결"),
                       ("paired","answer","규칙 문항 정답"),("general","answer","일반 문항 정답"),
                       ("general","joint","일반 문항 정답·ERA 연결"),("sentinel","answer","보존 전용 정답")):
        tx.append(mdrow(label,*[band(s["metrics"][a][ds][f],s["metrics"][a][ds]["n"]) for a in ARMS]))
    return tx

def retain_table(s):
    tx=["| 집합 | 기준 → 후보 | 유지 | 회복 | 손실 | 양쪽 오답 | 미정 |","|---|---|---:|---:|---:|---:|---:|"]
    for ds,label in DS.items():
        for a,b in (("base","ordinary"),("base","enhanced"),("ordinary","enhanced")):
            changes=s["changes"][ds][f"{a}-{b}-answer"]
            counts=[len(changes[c]) for c in TURN]
            assert sum(counts)==s["metrics"][a][ds]["n"]
            tx.append(mdrow(label,NAMES[a]+" → "+NAMES[b],*counts))
    return tx

def group_table(s):
    diff=s["pair_changes"]["joint"]
    kinds={g:x["kind"] for g,x in s["groups"]["ordinary"].items()}
    tx=["| 규칙 유형 | 유지 | 회복 | 손실 | 양쪽 실패 | 미정 |","|---|---:|---:|---:|---:|---:|"]
    for kind in sorted(set(kinds.values())):
        tx.append(mdrow(KINDS.get(kind,kind),*[sum(kinds[i]==kind for i in diff[k]) for k in TURN]))
    tx.append(mdrow("전체",*[len(diff[k]) for k in TURN]))
    return tx

def link_gate(d,root):
    folder=root/"mmmu/enhanced"
    if not d["promote"]:
        assert not (folder/"full/complete.json").exists()
        return "사전 진입 조건을 통과하지 못해 MMMU를 실행하지 않았다. 이번 후보의 575/900 유지 여부는 미측정이다."
    def official(scope):
        dest=folder/scope
        certificate=dest/"complete.json"
        if not certificate.is_file():return None
        check=read(certificate)
        assert check["scope"]==scope and sha(dest/"summary.json")==check["outputs"]["summary.json"]
        summary=read(dest/"summary.json")
        assert check["n"]==summary["n"] and scope==summary["scope"]
        return summary
    final=official("full")
    if final is None:
        screen=official("screen")
        if screen:
            return f"MMMU 진입 조건을 통과했다. 첫 {screen['n']}문항에서 {screen['correct']}개를 맞혔고 전체 900문항 결과는 대기 중이다. [첫 구간 공식 채점](mmmu/enhanced/screen/summary.json)"
        return "MMMU 진입 조건을 통과했다. 전체 900문항 공식 채점 완료 기록을 기다리고 있다."
    assert final["n"]==900 and final["baseline_correct"]==575
    lost,gained=final["paired_lost"],final["paired_gained"]
    assert 0<=lost<=575 and 0<=gained<=325 and final["correct"]==575-lost+gained
    status="유지했다" if final["correct"]>=575 else "밑돌았다"
    return (f"A 보강형은 공식 MMMU {final['correct']}/900으로 LoRA 전 기준 575/900을 {status}. "
            f"원본 정답 575개 중 {575-lost}개를 유지했고 {lost}개를 잃었다. 원본 오답 325개에서 {gained}개를 회복했다. "
            f"출력 한도 종료는 {final['length']}건이었다. [전체 공식 채점](mmmu/enhanced/full/summary.json)")

def id_report(scores,s):
    byid={}
    for r in scores:
        group=byid.setdefault(r["id"],{});assert r["arm"] not in group
        group[r["arm"]]=r
    assert len(byid)==192 and all(set(rs)==set(ARMS) for rs in byid.values())
    flag=lambda x:{True:"O",False:"X",None:"?"}[x]
    tx=["# 문항 및 그룹별 비교","","O는 통과, X는 실패, ?는 미정이다.","",
        "| 문항 ID | 평가 집합 | LoRA 전 정답 | 일반 정답 | A 보강 정답 | LoRA 전 연결 | 일반 연결 | A 보강 연결 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for item,rs in sorted(byid.items()):
        row=rs["base"];label="보존 전용(일반)" if row["sentinel"] else DS[row["dataset"]]
        tx.append(mdrow(f"`{item}`",label,*[flag(rs[a][f]) for f in ("answer","joint") for a in ARMS]))
    tx.extend(["","## 규칙 48그룹","","| 그룹 | 유형 | 일반 양쪽 E·R | A 보강 양쪽 E·R | 일반 양쪽 연결 | A 보강 양쪽 연결 |","|---|---|---:|---:|---:|---:|"])
    for g,c in sorted(s["groups"]["ordinary"].items()):
        h=s["groups"]["enhanced"][g]
        tx.append(mdrow(f"`{g}`",KINDS.get(c["kind"],c["kind"]),*[flag(v[f]) for f in ("er","joint") for v in (c,h)]))
    return tx

def build(root):
    review=root/"eval/review"
    decision=read(root/"decision.json")
    assert sha(review/"summary.json")==decision["summary_sha256"] and sha(review/"scored.jsonl")==decision["scored_sha256"]
    s=read(review/"summary.json")
    assert s["tests"]==decision["tests"] and s["promote"]==decision["promote"]==all(s["tests"].values())
    assert set(s["tests"])==set(GATES)
    scores=[json.loads(line) for line in (review/"scored.jsonl").read_text(encoding="utf8").splitlines() if line]
    assert len(scores)==decision["total_solver_calls"]==576 and decision["fresh_unique_per_model"]==192
    tr=read(root/"train/training_result.json");pre=root/"train/data/receipt.json"
    assert tr["verified"] and tr["epochs"]==2 and tr["n"]==296 and len(tr["changed_ids"])==24 and tr["preflight_sha256"]==sha(pre)
    preflight=read(pre)
    assert preflight["identical_evidence_rules_and_finals"] and preflight["identical_input_prefixes"]
    assert all(tr["arms"][a]["updates"]==38 for a in ("ordinary","enhanced"))
    c=s["common_er"]
    diff=s["pair_changes"]["joint"]
    control=s["groups"]["ordinary"];enh=s["groups"]["enhanced"]
    loo_min=min(sum(enh[k]["joint"] is True for k in enh if k!=g)-sum(control[k]["joint"] is not False for k in control if k!=g) for g in control)
    assert (loo_min>0)==s["tests"]["joint_leave_one"]
    tx=["# A 보강형의 ERA 연결과 기존 정답 유지","",
        f"독립 48쌍에서 양쪽 정답과 ERA 연결을 모두 맞힌 그룹은 일반형 {grouped(s,'ordinary','joint')}, A 보강형 {grouped(s,'enhanced','joint')}이었다. "
        f"일반형이 실패한 그룹 중 {len(diff['recovered'])}개를 회복하고 일반형이 맞힌 그룹 중 {len(diff['lost'])}개를 잃었다.","",
        f"두 군 모두 양쪽 E·R을 맞힌 같은 {c['n']}그룹에서 일반형 연결은 {band(c['ordinary'],c['n'])}, 보강형은 {band(c['enhanced'],c['n'])}이었다. "
        f"공통 집합에 포함될지 미정인 그룹은 {len(c['possible_members'])}개이며, 이를 반영한 순증의 하한은 {c['conservative_net']:+d}그룹이다.","",
        *grade_table(s),"","수치에 범위가 있으면 왼쪽은 확실히 통과한 수, 오른쪽은 미정까지 통과한 경우의 수다. 보존 전용 16문항은 일반 96문항에 포함된다.","",
        "## 그룹별 연결 변화","",*group_table(s),"",
        f"회복한 그룹은 {indices(diff['recovered'])}, 잃은 그룹은 {indices(diff['lost'])}이다. 한 그룹씩 빼고 계산한 연결 순증의 최솟값은 {loo_min:+d}그룹이다.","",
        "## 기존 정답 유지","",*retain_table(s),"",
        f"LoRA 전과 일반형이 모두 맞혔는데 A 보강형이 잃은 일반 문항은 {len(decision['original_general_losses'])}개다({indices(decision['original_general_losses'])}). "
        f"보존 전용에서 일반형 정답의 신규 손실은 {len(decision['sentinel_losses'])}개다({indices(decision['sentinel_losses'])}). [전체 문항 및 그룹별 비교](retention_ids.md)","",
        "## 최초 오류","",
        "| 규칙 문항당 최초 오류 또는 성공 | LoRA 전 | 일반 ERA | A 보강 ERA |","|---|---:|---:|---:|"]
    counts=s["first_error_counts"]
    assert set().union(*(x.keys() for x in counts.values()))<=set(ERRORS)
    assert all(sum(counts[a].values())==96 for a in ARMS)
    for error,label in ERRORS.items():
        if any(counts[a].get(error,0) for a in ARMS):tx.append(mdrow(label,*[counts[a].get(error,0) for a in ARMS]))
    tx.extend(["","비승자 반올림은 엄격 기준에서 손실이며, 승자·최종 답이 유지된 경우의 보조 지표를 첫 표에 함께 기록했다.","",
               "## 학습 및 평가 조건","",
               f"같은 LoRA 전 Qwen에서 각 {tr['n']}문항을 {tr['epochs']} epoch 학습했다. 일반형은 {tr['arms']['ordinary']['updates']}회, A 보강형은 {tr['arms']['enhanced']['updates']}회 업데이트했다. "
               f"표적 {len(tr['changed_ids'])}문항에서는 A만 달랐으며 E·R·최종 답과 입력은 같았다. "
               f"표적의 epoch당 지도 토큰은 일반형 {tr['length_stats']['modified24']['ordinary']['total']:,}, 보강형 {tr['length_stats']['modified24']['enhanced']['total']:,}개였다. "
               "[학습 검증](train/training_result.json) · [훈련 표적 원장](data/rows.jsonl)","",
               f"각 모델은 서로 다른 {decision['fresh_unique_per_model']}개 입력에 한 번씩, 총 {decision['total_solver_calls']}회 답했다. "
               "별도 평가기와 원그림 검수로 문항별 연결을 채점했다. 생성 시드는 3407 한 개였다. [채점 규약](eval/review_protocol.md) · [독립 검증](decision.json)","",
               "## MMMU 진입 및 공식 결과","","| 사전 조건 | 판정 |","|---|---|"])
    for k,name in GATES.items():tx.append(mdrow(name,"통과" if s["tests"][k] else "미통과"))
    tx.extend(["",link_gate(decision,root),"","[전체 채점 원장](eval/review/scored.jsonl) · [집계와 전이 ID](eval/review/summary.json)"])
    if (root/"cases.md").exists():
        tx.extend(["", "[회복·손실 문항의 실제 풀이와 그림](cases.md)"])
    return "\n".join(tx)+"\n","\n".join(id_report(scores,s))+"\n"

def main():
    report,ids=build(ROOT)
    (ROOT/"retention_ids.md").write_text(ids,encoding="utf8")
    (ROOT/"report.md").write_text(report,encoding="utf8")

if __name__=="__main__":main()
