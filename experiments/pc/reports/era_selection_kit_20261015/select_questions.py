#!/usr/bin/env python3
"""Deterministic ERA pilot selection. Python 3.9+, standard library only."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from urllib.request import urlopen

VERSION = "era-select-v1-20261015"
HEADERS = ["담당", "데이터셋", "이미지그룹번호", "이미지상대경로", "원본문항ID",
           "선별버전", "원본지문", "사용모델", "제작방법", "상태", "제외범위", "메모"]
STATES = {"예정", "작성중", "작성완료", "제외"}
LOCK_PATH = Path(__file__).with_name("source_lock.json")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(value):
    value = value.replace("\\", "/")
    if value.startswith("/") or ":" in value or ".." in value.split("/"):
        raise ValueError("Unsafe relative image path: " + value)
    return str(PurePosixPath(value))


def load_sources(data_dir, owner):
    lock = read_json(LOCK_PATH)
    names = (["chartqa_train_human.json"] if owner in "AB" else
             ["scienceqa_problems.json", "scienceqa_splits.json"])
    loaded, hashes = {}, {}
    for name in names:
        blob = (Path(data_dir) / name).read_bytes()
        hashes[name] = digest(blob)
        if hashes[name] != lock[name]["sha256"]:
            raise ValueError(f"Source version mismatch: {name}. Use the shared locked snapshot.")
        loaded[name] = json.loads(blob.decode("utf-8-sig"))
    fingerprint = digest(json.dumps(hashes, sort_keys=True).encode())
    if owner in "AB":
        rows = chart_rows(loaded[names[0]])
    else:
        rows = science_rows(loaded[names[0]], loaded[names[1]])
    return rows, fingerprint


def chart_rows(items):
    rows = []
    for i, item in enumerate(items):
        if not item.get("imgname"):
            continue
        rows.append({"dataset": "ChartQA", "source_id": f"chartqa_train_human_{i:06d}",
                     "image_path": relative_path("train/png/" + item["imgname"]),
                     "question": item["query"], "original_answer": item["label"],
                     "original_options": [], "hint": "", "lecture": "", "solution": ""})
    return rows


def science_rows(problems, splits):
    rows = []
    # Original split-array order is the question order; never filesystem order.
    for pid in splits["train"]:
        pid = str(pid)
        item = problems[pid]
        if not item.get("image"):
            continue
        rows.append({"dataset": "ScienceQA", "source_id": pid,
                     "image_path": relative_path(f"train/{pid}/{item['image']}"),
                     "question": item["question"], "original_answer": item["answer"],
                     "original_options": item["choices"], "hint": item.get("hint", ""),
                     "lecture": item.get("lecture", ""), "solution": item.get("solution", "")})
    return rows


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_path"]].append(row)
    # Case-sensitive Unicode code-point order. No locale/natural sorting.
    for number, path in enumerate(sorted(groups), 1):
        yield number, path, groups[path]


def assigned_owner(dataset, number):
    return ("AB" if dataset == "ChartQA" else "CD")[(number - 1) % 2]


def binary_chart(row):
    return row["dataset"] == "ChartQA" and str(row["original_answer"]).strip().lower() in {
        "yes", "no", "true", "false"}


def new_log(owner, number, row, fingerprint):
    return dict(zip(HEADERS, [owner, row["dataset"], str(number), row["image_path"],
                             row["source_id"], VERSION, fingerprint, "", "", "예정", "", ""]))


def read_log(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADERS:
            raise ValueError(f"CSV header mismatch: {path}; preserve the template headers/order.")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return [row for row in rows if any(row.values())]


def validate_log(log, owner, rows, fingerprint):
    index = {r["source_id"]: (n, r) for n, _, items in grouped(rows) for r in items}
    ids, active_paths, blocked_paths = set(), set(), set()
    for rec in log:
        sid = rec["원본문항ID"]
        if sid not in index or sid in ids:
            raise ValueError(f"Unknown or repeated question ID: {sid}")
        ids.add(sid)
        number, row = index[sid]
        expected = new_log(owner, number, row, fingerprint)
        for field in HEADERS[:7]:
            if rec[field] != expected[field]:
                raise ValueError(f"Changed source field {field}: {sid}")
        if assigned_owner(row["dataset"], number) != owner:
            raise ValueError(f"Wrong owner/parity: {sid}")
        if rec["상태"] not in STATES:
            raise ValueError(f"Unknown status: {sid}")
        if rec["상태"] == "제외":
            if rec["제외범위"] not in {"문항", "이미지"} or not rec["메모"].strip():
                raise ValueError(f"Excluded rows need scope and reason: {sid}")
            if rec["제외범위"] == "이미지":
                blocked_paths.add(row["image_path"])
        else:
            if rec["제외범위"]:
                raise ValueError(f"Non-excluded row has exclusion scope: {sid}")
            if binary_chart(row):
                raise ValueError(f"Binary ChartQA answer cannot enter this four-choice pilot: {sid}")
            if row["image_path"] in active_paths:
                raise ValueError(f"More than one active question per image: {sid}")
            active_paths.add(row["image_path"])
        if rec["상태"] == "작성완료" and (not rec["사용모델"].strip() or not rec["제작방법"].strip()):
            raise ValueError(f"Completed row needs model (or '미사용') and authoring method: {sid}")
    if active_paths & blocked_paths:
        raise ValueError("An image marked excluded still has an active question.")
    return index


def select(owner, rows, fingerprint, target, history):
    if target < 1:
        raise ValueError("Target must be positive.")
    index = validate_log(history, owner, rows, fingerprint)
    log = [dict(r) for r in history]
    active = [r for r in log if r["상태"] != "제외"]
    if len(active) > target:
        raise ValueError("Target is below the existing active count; no records were removed.")
    used = {r["이미지상대경로"] for r in active}
    rejected = {r["원본문항ID"] for r in log if r["상태"] == "제외"}
    blocked = {r["이미지상대경로"] for r in log if r["상태"] == "제외" and r["제외범위"] == "이미지"}
    skipped_binary = 0
    for number, path, items in grouped(rows):
        if len(active) >= target:
            break
        if assigned_owner(items[0]["dataset"], number) != owner or path in used | blocked:
            continue
        for row in items:
            if row["source_id"] in rejected:
                continue
            if binary_chart(row):
                skipped_binary += 1
                continue
            rec = new_log(owner, number, row, fingerprint)
            active.append(rec)
            log.append(rec)
            used.add(path)
            break
    if len(active) != target:
        raise ValueError(f"Only {len(active)} candidates available; requested {target}. No output written.")
    log.sort(key=lambda r: (int(r["이미지그룹번호"]), r["원본문항ID"]))
    selected = []
    for rec in sorted(active, key=lambda r: int(r["이미지그룹번호"])):
        row = dict(index[rec["원본문항ID"]][1])
        row.update(owner=owner, image_group_no=int(rec["이미지그룹번호"]),
                   selection_version=VERSION, source_fingerprint=fingerprint,
                   review_status="unreviewed", image_content_overlap="not_run", mmmu_overlap="not_run")
        selected.append(row)
    return log, selected, skipped_binary


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def fetch_sources(data_dir):
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, spec in read_json(LOCK_PATH).items():
        dest = root / name
        if dest.exists():
            if digest(dest.read_bytes()) != spec["sha256"]:
                raise ValueError(f"Existing file differs; refusing overwrite: {dest}")
        else:
            with urlopen(spec["url"], timeout=60) as response:
                blob = response.read()
            if digest(blob) != spec["sha256"]:
                raise ValueError(f"Downloaded hash differs: {name}")
            with dest.open("xb") as f:
                f.write(blob)
        print(f"Verified: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download locked original QA JSONs; images are separate")
    fetch.add_argument("--data", type=Path, default=Path("data"))
    pick = sub.add_parser("select", help="Select first candidates for one owner")
    pick.add_argument("--owner", required=True, choices=list("ABCD"))
    pick.add_argument("--data", type=Path, default=Path("data"))
    pick.add_argument("--count", type=int, default=50)
    pick.add_argument("--history", type=Path, help="Previous edited image_log.csv")
    pick.add_argument("--out", required=True, type=Path, help="New, empty output directory")
    check = sub.add_parser("check", help="Check selected logs against source and for path/ID overlap")
    check.add_argument("--data", type=Path, default=Path("data"))
    check.add_argument("--logs", nargs="+", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_sources(args.data)
    elif args.command == "select":
        if args.out.exists() and any(args.out.iterdir()):
            raise ValueError("Output directory must be empty. Use a new directory to preserve manual edits.")
        rows, fingerprint = load_sources(args.data, args.owner)
        history = read_log(args.history) if args.history else []
        log, selected, skipped = select(args.owner, rows, fingerprint, args.count, history)
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "image_log.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()
            writer.writerows(log)
        with (args.out / "selected_questions.jsonl").open("w", encoding="utf-8") as f:
            for row in selected:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = {"owner": args.owner, "selected": len(selected), "history_rows": len(log),
                  "source_fingerprint": fingerprint, "selection_version": VERSION,
                  "binary_candidates_skipped_this_run": skipped,
                  "image_file_existence": "not_checked", "image_content_and_mmmu_overlap": "not_run"}
        write_json(args.out / "run_info.json", report)
        print(json.dumps(report, ensure_ascii=False))
    else:
        by_owner = defaultdict(list)
        for path in args.logs:
            for row in read_log(path):
                if row["담당"] not in "ABCD" or len(row["담당"]) != 1:
                    raise ValueError("Unknown owner in " + str(path))
                by_owner[row["담당"]].append(row)
        if not by_owner:
            raise ValueError("No records to check.")
        paths, ids, counts = set(), set(), Counter()
        for owner, log in by_owner.items():
            rows, fingerprint = load_sources(args.data, owner)
            validate_log(log, owner, rows, fingerprint)
            for rec in log:
                if rec["상태"] == "제외":
                    continue
                key = (rec["데이터셋"], rec["이미지상대경로"])
                qid = (rec["데이터셋"], rec["원본문항ID"])
                if key in paths or qid in ids:
                    raise ValueError("Duplicate active image/question: " + str(key))
                paths.add(key)
                ids.add(qid)
                counts[owner] += 1
        print(json.dumps({"path_and_id_check": "passed", "active_counts": dict(counts),
                          "image_content_and_mmmu_overlap": "not_run"}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit("ERROR: " + str(exc))
