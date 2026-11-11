"""Read-only MMMU comparison. Only derived reports are written; a solver or judge is never loaded."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASE = "3791f54f30b9e127fa995f2f442ab0460c988addb069377596ea5830a7a8f0d8"
BASE_LEDGER = "14e96484cbd3c50e14ea9202316cf388e28be44ec34352f81d713f7dde7047ee"
OFFICIAL = "ce72d13a3ac569fa2e67aafab3d54d5d2665f6960d356ea7212be0dfb54e3c3e"
SOURCE = {
    "config.json": "b55e958623cd59e3da64aef4e228a98e83e586d924b80a8b3fd22dafd7c938e5",
    "evaluate.py": "2dbc17e8953cbdf1ea1091e2890489c7f0592efe21e5d56a13bf1b61b642e75d",
    "prepare.py": "581b4c2905fbd71f77c3e5f6f23a13c4444a125609a3c9b3ac4bfbe267f476ef",
    "parsers.py": "5023ed38d2e8af91304b258d5b3798ffc03df24324f49740953d943378a30b04",
}
JUDGE = "local_qwen3_8b_judge_parser"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return digest(path.read_bytes())


def journal(path):
    data = path.read_bytes() if path.exists() else b""
    committed = data[:data.rfind(b"\n") + 1].splitlines(keepends=True)
    return committed, [json.loads(line) for line in committed], len(data) - sum(map(len, committed))


def checked_source(args):
    for name, h in SOURCE.items():
        require(file_sha(args.source/name) == h, f"Different baseline source: {name}")
    require(file_sha(args.official) == OFFICIAL, "Different official MMMU runner")
    sys.path.insert(0, str(args.source.resolve()))
    spec = importlib.util.spec_from_file_location("official_mmmu", args.official)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hit(row):
    return row["grading"]["final"]["hit"]


def subset(ids, archive, rows):
    kept = lost = gained = both_wrong = missing = pending = 0
    for rid in ids:
        if rid not in rows:
            missing += 1
        elif hit(rows[rid]) is None:
            pending += 1
        else:
            b, c = hit(archive[rid]), hit(rows[rid])
            kept += bool(b and c)
            lost += bool(b and not c)
            gained += bool(not b and c)
            both_wrong += bool(not b and not c)
    return dict(n=len(ids), baseline_correct=sum(hit(archive[k]) for k in ids), kept=kept,
                lost=lost, gained=gained, both_wrong=both_wrong, net_on_graded=gained-lost,
                candidate_known_correct=kept+gained, pending_grade=pending, missing_prediction=missing,
                possible_correct=[kept+gained, kept+gained+pending+missing])


def routes(records):
    methods = Counter()
    length = extracted_by_judge = judge_failure = 0
    for row in records:
        grade = row["grading"]["final"]
        method = grade["method"]
        methods[method] += 1
        length += row["model_response"]["finish_reason"] == "length"
        extracted_by_judge += method == JUDGE
        judge_failure += bool(grade.get("extraction_failure_forced_incorrect", False))
        if row["model_response"]["finish_reason"] == "length":
            require(grade["hit"] == 0 and method == "output_limit_incorrect", f"Truncation policy: {row['id']}")
    return dict(methods=dict(methods), truncated=length, judge_parsed_or_exhausted=extracted_by_judge,
                judge_extraction_exhausted=judge_failure)


def comparison(args):
    official = checked_source(args)
    require(file_sha(args.baseline) == BASE, "Wrong pre-LoRA MMMU archive")
    original = read(args.baseline)
    config = read(args.source/"config.json")
    require(original["manifest"]["configuration"] == config, "Original configuration mismatch")
    archive = {r["id"]: r for r in original["records"]}
    require(len(archive) == len(original["records"]) == 900 and sum(map(hit, archive.values())) == 575,
            "Wrong pre-LoRA IDs or grades")
    baseline_file = args.baseline.parent/"predictions.jsonl"
    require(file_sha(baseline_file) == BASE_LEDGER, "Wrong pre-LoRA solver journal")
    _, basepred, tail = journal(baseline_file)
    require(not tail and [r["id"] for r in basepred] == list(archive)
            and all(r["model_response"] == archive[r["id"]]["model_response"] for r in basepred), "Baseline rows")
    subjects = sorted({r["subject"] for r in archive.values()})
    screen_set = set()
    for s in subjects:
        group = [rid for rid, row in archive.items() if row["subject"] == s]
        require(len(group) == 30, f"Subject size: {s}")
        screen_set.update(sorted(group, key=lambda rid: digest((official.SCREEN_KEY+rid).encode()))[:3])
    full = list(archive)
    screen = [rid for rid in full if rid in screen_set]
    held = [rid for rid in full if rid not in screen_set]
    run = args.book/args.name
    if (args.book/"roster.json").exists():
        plan = read(args.book/"roster.json")
        require(plan["screen_ids"] == screen and plan["full_ids"] == full and plan["seed"] == 3407
                and plan["roster_sha256"] == official.ROSTER_FILE and plan["rgb_sha256"] == official.PIXEL_FILE
                and plan["archive_sha256"] == BASE, "Roster drift")
    sealed = {}
    for scope in ("screen","full"):
        proof = run/scope/"complete.json"
        if proof.exists():
            sealed[scope] = read(proof)
    attempt_lines, attempts, judge_tail = journal(run/"judge_attempts.jsonl")
    lines, records, pred_tail = journal(run/"predictions.jsonl")
    ids = [r["id"] for r in records]
    require(ids == (screen+held)[:len(ids)] and len(set(ids)) == len(ids), "Duplicate, unexpected or out-of-order solver ID")
    manifest_file = run/"manifest.json"
    manifest = read(manifest_file) if manifest_file.exists() else None
    if records or sealed or manifest:
        require(manifest and manifest["solver_calls_per_item"] == 1 and manifest["source_sha256"] == OFFICIAL and
                manifest["pixels_sha256"] == official.PIXEL_FILE and manifest["baseline_sha256"] == BASE and
                manifest["configuration"] == config, "Invalid solver/scoring provenance")
        if args.name == "reverse":
            expected = HERE.parent/"train/models/r6/reverse"
            require(Path(manifest["model_identity"]["path"]) == expected/"merged", "Wrong R6 checkpoint path")
            weights = {k:v for k,v in manifest["model_identity"]["hashes"].items() if k.endswith(".safetensors")}
            require(weights and weights == read(expected/"merge.json")["weights"], "Wrong R6 checkpoint weights")
    for row in records:
        official.match_inputs(row, archive[row["id"]])
    pending = official.graded(records, attempts, config)
    rows = {r["id"]: r for r in records}
    require(set(pending) == {rid for rid, row in rows.items() if hit(row) is None}, "Pending replay mismatch")
    scopes = {}
    for scope, proof in sealed.items():
        dest = run/scope
        expected_ids = screen if scope == "screen" else full
        require(scope == "screen" or "screen" in sealed, "Full result unaccompanied by prior screen")
        require(proof["scope"] == scope and proof["n"] == len(expected_ids) and proof["model"] == manifest["fingerprint"]
                and set(proof["outputs"]) == {"results.json","summary.json"} and
                all(file_sha(dest/k) == h for k,h in proof["outputs"].items()), f"{scope}: result seal")
        require(proof["predictions_sha256"] == digest(b"".join(lines[:len(expected_ids)])), f"{scope}: solver journal seal")
        result, summary = read(dest/"results.json"), read(dest/"summary.json")
        require(result["manifest"] == manifest and result["summary"] == summary and result["records"] ==
                [rows[k] for k in expected_ids], f"{scope}: saved grades or screen reuse differ from CPU replay")
        tallies = subset(expected_ids, archive, rows)
        detail = routes(result["records"])
        require(summary["scope"] == scope and summary["n"] == len(expected_ids) and
                summary["baseline_correct"] == tallies["baseline_correct"] and summary["correct"] == tallies["candidate_known_correct"]
                and summary["judge_count"] == detail["judge_parsed_or_exhausted"] and summary["length"] == detail["truncated"]
                and summary["grading_methods"] == detail["methods"] and summary["paired_gained"] == tallies["gained"]
                and summary["paired_lost"] == tallies["lost"] and not tallies["pending_grade"], f"{scope}: official tallies")
        if scope == "full":
            rest = subset(held, archive, rows)
            require(summary["remaining810"] == {"n":810, "correct":rest["candidate_known_correct"],
                    "baseline_correct":rest["baseline_correct"]} and not pred_tail, "Held 810 or solver ledger tail mismatch")
        scopes[scope] = dict(correct=summary["correct"], n=summary["n"], proof_sha256=file_sha(dest/"complete.json"),
                             results_sha256=proof["outputs"]["results.json"], predictions_sha256=proof["predictions_sha256"])
    summary = subset(full, archive, rows)
    official_full = "full" in scopes
    stored = len(records)
    return dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), model=args.name, official_full=official_full,
                baseline_correct=575, matched_575=(summary["candidate_known_correct"] >= 575) if official_full else None,
                official_total_correct=scopes["full"]["correct"] if official_full else None, scopes=scopes, full=summary,
                screen=subset(screen, archive, rows), remaining810=subset(held, archive, rows),
                by_subject={s:subset([rid for rid in full if archive[rid]["subject"] == s], archive, rows) for s in subjects},
                routes_candidate=routes(records), routes_baseline=routes(archive.values()), audit=dict(
                    baseline_sha256=BASE, base_prediction_frames=900, official_runner_sha256=OFFICIAL,
                    baseline_image_fields=sum(len(r["image_fields"]) for r in archive.values()), candidate_image_fields=sum(len(r["image_fields"]) for r in records),
                    committed_prediction_frames=stored, unique_prediction_ids=len(set(ids)), prediction_tail_bytes=pred_tail,
                    judge_attempt_frames=len(attempt_lines), judge_tail_bytes=judge_tail, unique_judged_ids=len({a["id"] for a in attempts}),
                    current_inflight=(read(run/"solver_inflight.json") if (run/"solver_inflight.json").exists() else None),
                    reused_screen_rows_verified_in_full=len(screen) if official_full else None,
                    outside_screen_prediction_frames=len(set(ids)-screen_set), official_sealed_rows=900 if official_full else 90 if "screen" in sealed else 0))


def render(data):
    s, a = data["full"], data["audit"]
    if data["official_full"]:
        status = f"공식 결과는 {data['official_total_correct']}/900으로 LoRA 전 575/900 기준을 {'충족했다' if data['matched_575'] else '충족하지 못했다'}."
    else:
        status = (f"공식 전체 점수는 미확정이다. 저장된 응답의 공식 파서와 기존 저지 판정을 재생하면 확정 정답 "
                  f"{s['candidate_known_correct']}개, 미채점 {s['pending_grade']}개, 미저장 {s['missing_prediction']}개이며 "
                  f"최종 가능한 정답 수는 {s['possible_correct'][0]}~{s['possible_correct'][1]}/900이다.")
    lines = ["# MMMU 공식 채점 비교", "", status, "",
             f"저장된 solver 응답 {a['committed_prediction_frames']}개와 고유 문항 ID {a['unique_prediction_ids']}개를 대조했다. "
             f"전체 공식 봉인에 포함된 문항은 현재 {a['official_sealed_rows']}개다. "
             f"저지 시도 {a['judge_attempt_frames']}건은 답 추출에만 쓰였다.", "",
             "| 집합 | 원본 정답 | 후보 확정 정답 | 유지 | 손실 | 회복 | 양쪽 오답 | 채점 대기 | 미저장 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    fields = ("baseline_correct","candidate_known_correct","kept","lost","gained","both_wrong","pending_grade","missing_prediction")
    for label,key in (("전체 900","full"),("스크린 90","screen"),("나머지 810","remaining810")):
        lines.append("| "+" | ".join([label, *[str(data[key][k]) for k in fields]])+" |")
    if data["official_full"]:
        lines += ["", "스크린 90개 응답의 해시·원문·공식 판정이 전체 결과와 일치한다. 나머지 810개에서만 새 응답을 생성했다."]
    lines += ["", "| 채점 경로 | LoRA 전 900개 | 후보 저장분 |", "|---|---:|---:|"]
    bm = data["routes_baseline"]["methods"]
    cm = data["routes_candidate"]["methods"]
    for method in sorted(set(bm)|set(cm)):
        lines.append(f"| {method} | {bm.get(method, 0)} | {cm.get(method, 0)} |")
    lines += ["", "| 과목 | 원본 정답 | 후보 확정 정답 | 회복 | 손실 | 확정 순변화 | 미채점·미저장 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, q in sorted(data["by_subject"].items(), key=lambda x:(-x[1]["net_on_graded"], x[0])):
        lines.append(f"| {name} | {q['baseline_correct']} | {q['candidate_known_correct']} | {q['gained']} | {q['lost']} | {q['net_on_graded']:+d} | {q['pending_grade']+q['missing_prediction']} |")
    return "\n".join(lines)+"\n"


def save(path, text):
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf8")
    temp.replace(path)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--book", type=Path, default=HERE.parent/"mmmu")
    cli.add_argument("--name", default="reverse")
    cli.add_argument("--baseline", type=Path, default=Path.home()/"mmmu_baseline_20261009/baseline_seed3407/results.json")
    cli.add_argument("--source", type=Path, default=Path.home()/"era_decoder_lora_20261030/baseline/src")
    cli.add_argument("--official", type=Path, default=Path.home()/"era_single_call_campaign/mmmu_eval.py")
    cli.add_argument("--out", type=Path, default=HERE)
    args = cli.parse_args()
    data = comparison(args)
    args.out.mkdir(parents=True, exist_ok=True)
    save(args.out/"mmmu_comparison.json", json.dumps(data, ensure_ascii=False, indent=2)+"\n")
    report = render(data)
    save(args.out/"mmmu_findings.md", report)
    if args.out.resolve() == HERE:
        save(HERE.parent/"mmmu_findings.md", report)
    print(json.dumps(dict(official_full=data["official_full"], full=data["full"], scopes=data["scopes"])))


if __name__ == "__main__":
    main()
