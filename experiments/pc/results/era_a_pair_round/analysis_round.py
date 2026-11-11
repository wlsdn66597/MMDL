"""CPU-only report from committed predictions. Never issues a generation or judge request."""
from collections import Counter
import json
from pathlib import Path
import re
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def strict_era(row):
    response = row["model_response"]
    raw = response["raw"].strip()
    fields = list(re.finditer(r"(?im)^\s*(?:\*\*)?(Evidence|Rule|Application|Final answer)(?:\*\*)?\s*:", raw))
    return bool(response["finish_reason"] == "stop" and [m[1].lower() for m in fields] ==
                ["evidence","rule","application","final answer"] and
                all(raw[a.end():b.start()].strip(" \r\n*") for a,b in zip(fields, fields[1:])) and
                re.fullmatch(r"(?i)(?:\*\*)?Final answer(?:\*\*)?\s*:\s*\S[^\r\n]*", raw.splitlines()[-1].strip()))


def tally(rows, baseline):
    pairs = Counter((baseline[r["id"]], r["grading"]["final"]["hit"]) for r in rows)
    correct = pairs[True, True] + pairs[False, True]
    pending = pairs[True, None] + pairs[False, None]
    return dict(n=len(rows), correct=correct, pending=pending, correct_upper=correct+pending,
                baseline=sum(baseline[r["id"]] for r in rows), gained=pairs[False,True],
                lost=pairs[True,False], kept=pairs[True,True], both_wrong=pairs[False,False],
                era=sum(map(strict_era,rows)), length=sum(r["model_response"]["finish_reason"]=="length" for r in rows),
                possible_total900=[correct, correct+pending+900-len(rows)])


def arm_results(root, arm, archive, config, official):
    book = root / "mmmu" / arm
    run = book / arm
    ledger = run / "predictions.jsonl"
    if not ledger.is_file():
        return {}, {}
    roster = read(book / "roster.json")
    lines = ledger.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    rows = [json.loads(line) for line in lines]
    screen = set(roster["screen_ids"])
    order = roster["screen_ids"] + [rid for rid in roster["full_ids"] if rid not in screen]
    if [r["id"] for r in rows] != order[:len(rows)]:
        raise ValueError(f"Duplicate or out-of-order {arm} ledger")
    for row in rows:
        official.match_inputs(row, archive[row["id"]])
    attempts = run / "judge_attempts.jsonl"
    # A deadline can interrupt the last uncommitted line of either ledger.
    attempt_lines = attempts.read_bytes().splitlines(keepends=True) if attempts.exists() else []
    official.graded(rows, [json.loads(x) for x in attempt_lines if x.endswith(b"\n")], config)
    keyed = {r["id"]:r for r in rows}
    for scope in ("screen","full"):
        dest = run / scope
        if (dest / "complete.json").exists():
            proof = read(dest / "complete.json")
            if not all(official.sha(dest/name)==h for name,h in proof["outputs"].items()):
                raise ValueError(f"{arm}/{scope} score seal mismatch")
            for row in read(dest / "results.json")["records"]:
                if row != keyed[row["id"]]:
                    raise ValueError(f"Official and CPU replay disagree for {arm}/{row['id']}")
    baseline = {rid:r["grading"]["final"]["hit"] for rid,r in archive.items()}
    result = {key:tally(group,baseline) for key,group in (
        ("all_saved", rows), ("screen_saved",[r for r in rows if r["id"] in screen]),
        ("outside_screen_saved",[r for r in rows if r["id"] not in screen]))}
    result["official_full"] = read(run/"full/summary.json") if (run/"full/complete.json").exists() else None
    return result, keyed


def make_report(root, old, state):
    sys.path.insert(0, str(old))
    sys.path.insert(0, str(Path.home() / "era_decoder_lora_20261030/baseline/src"))
    import mmmu_eval as official
    archive_file = Path.home() / "mmmu_baseline_20261009/baseline_seed3407/results.json"
    if official.sha(archive_file) != official.BASELINE_RESULT:
        raise ValueError("Baseline archive identity mismatch")
    archive = {r["id"]:r for r in read(archive_file)["records"]}
    config = read(Path.home() / "era_decoder_lora_20261030/baseline/src/config.json")
    out, keyed = {}, {}
    for arm in ("control","candidate"):
        out[arm], keyed[arm] = arm_results(root,arm,archive,config,official)
    common = sorted(keyed["candidate"].keys() & keyed["control"].keys())
    paired = Counter()
    for rid in common:
        c,p = (keyed[a][rid]["grading"]["final"]["hit"] for a in ("control","candidate"))
        key = "unresolved" if None in (c,p) else ("both_correct" if c and p else "gained" if p else "lost" if c else "both_wrong")
        paired[key] += 1
    out.update(state=state, shared_n=len(common), candidate_vs_control_paired=dict(paired))
    official.save(root/"mmmu_status.json",out)
    md = ["# 같은 E·R에 대한 A 대조 학습", "", f"실행 상태 `{state}`. LoRA 전 정답률 기준은 seed 3407의 575/900이다.",
          "", "| 모델·집합 | 저장 | 확정 정답 | 저지 대기 | 같은 ID의 원본 정답 | 원본 대비 확정 회복 / 손실 | 완전 ERA |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for arm,label in (("control","정답 학습"),("candidate","A 대조 학습")):
        for scope,title in (("screen_saved","개발"),("outside_screen_saved","개발 외"),("all_saved","전체 저장분")):
            x = out[arm].get(scope)
            if x and x["n"]:
                md.append(f"| {label} · {title} | {x['n']} | {x['correct']} | {x['pending']} | {x['baseline']} | {x['gained']} / {x['lost']} | {x['era']} |")
    md.extend(["",f"두 군 공통 저장 {len(common)}문항. 대조 학습의 회복 {paired['gained']}, 손실 {paired['lost']}, 양쪽 정답 {paired['both_correct']}, 양쪽 오답 {paired['both_wrong']}, 미확정 {paired['unresolved']}.",
               "", "900문항 전체의 공식 결과는 full/complete.json이 있는 군만 확정한다. 새 도표에서의 내용 연결은 connection_work의 별도 결과로 판정한다."])
    dest = root / "server_report.md"
    temp = dest.with_suffix(".tmp")
    temp.write_text("\n".join(md)+"\n",encoding="utf8")
    temp.replace(dest)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    status = root / "status.json"
    make_report(root, Path.home()/"era_single_call_campaign", read(status)["state"] if status.exists() else "prepared")
