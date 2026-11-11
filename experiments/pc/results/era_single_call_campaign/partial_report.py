"""Offline audit of the terminal r3-control ledger. Never runs a model or changes a campaign artifact."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import campaign
import mmmu_eval as protocol


HERE = Path(__file__).resolve().parent
RUN = HERE / "benchmark/r3-control"
BASE = HERE.parent / "mmmu_first_baseline_20261009/baseline_seed3407/results.json"
SRC = HERE.parents[1] / "code/training/era_decoder_lora/baseline/src"
SOURCE_HASHES = {
    "config.json": "b55e958623cd59e3da64aef4e228a98e83e586d924b80a8b3fd22dafd7c938e5",
    "evaluate.py": "2dbc17e8953cbdf1ea1091e2890489c7f0592efe21e5d56a13bf1b61b642e75d",
    "parsers.py": "5023ed38d2e8af91304b258d5b3798ffc03df24324f49740953d943378a30b04",
    "prepare.py": "581b4c2905fbd71f77c3e5f6f23a13c4444a125609a3c9b3ac4bfbe267f476ef",
}
SCREEN_SEAL = "3dc21a3c2c3a9a074c3ba0ec9795f1c23a13a6d1bbff0620e3a8536c512fe53d"
ROSTER_SEAL = "730736e98559c7436ec09911806715ac9f4c599b3cb6ed062cc191260a42caa3"
sha, read, require = protocol.sha, protocol.read, protocol.require


def sealed_results(scope, ids, config):
    directory = RUN / scope
    seal = read(directory / "complete.json")
    require(seal["scope"] == scope and seal["n"] == len(ids), "Incomplete receipt")
    for name in ("results.json", "summary.json"):
        require(sha(directory / name) == seal["outputs"][name], f"{scope} has a changed {name}")
    pack = read(directory / "results.json")
    require(pack["summary"] == read(directory / "summary.json"), "Conflicting scoring summaries")
    require([r["id"] for r in pack["records"]] == ids, "Changed official result roster")
    manifest = pack["manifest"]
    require(manifest["fingerprint"] == seal["model"] and manifest["configuration"] == config
            and manifest["baseline_sha256"] == protocol.BASELINE_RESULT
            and manifest["solver_calls_per_item"] == 1
            and manifest["source_sha256"] == sha(HERE / "mmmu_eval.py"), "Changed model/protocol identity")
    require(manifest["model_identity"]["path"] == read(HERE / "nominee.json")["checkpoint"],
            "Result model is different from the fixed nominee")
    return pack, seal


def ungraded(row):
    return {k: v for k, v in row.items() if k != "grading"}


def strict_count(rows):
    # Reuse the frozen campaign's exact format criterion with in-memory rows.
    with patch.object(campaign, "read", return_value={"records": rows}):
        return campaign.strict_era(None)


def summarize(records, archive):
    scores = Counter()
    methods = Counter()
    for row in records:
        b = archive[row["id"]]["grading"]["final"]["hit"]
        grade = row["grading"]["final"]
        scores[b, grade["hit"]] += 1
        methods[grade["method"]] += 1
    yes = scores[1, 1] + scores[0, 1]
    unknown = scores[1, None] + scores[0, None]
    base_correct = sum(n for (b, _), n in scores.items() if b == 1)
    return {
        "saved_unique": len(records), "subjects_with_saved_rows": len({r["subject"] for r in records}),
        "baseline_same_ids_correct": base_correct,
        "known_correct": yes, "known_wrong": len(records) - yes - unknown,
        "pending_judge": unknown, "saved_correct_range": [yes, yes + unknown],
        "saved_correct_difference_vs_baseline_range": [yes - base_correct, yes + unknown - base_correct],
        "both_correct": scores[1, 1], "lost": scores[1, 0], "gained": scores[0, 1],
        "both_wrong": scores[0, 0], "unknown_baseline_correct": scores[1, None],
        "unknown_baseline_wrong": scores[0, None],
        "grading_methods": dict(sorted(methods.items())),
        "finish_reasons": dict(Counter(r["model_response"]["finish_reason"] for r in records)),
        "strict_era": strict_count(records),
        "strict_era_and_known_answer_correct": strict_count([r for r in records if r["grading"]["final"]["hit"] == 1]),
        "pending_judge_ids": [r["id"] for r in records if r["grading"]["final"]["hit"] is None],
    }


def audit(screen_test):
    for name, expected in SOURCE_HASHES.items():
        require(sha(SRC / name) == expected, f"Changed original scorer source: {name}")
    sys.path.insert(0, str(SRC))
    import evaluate as original

    require(sha(BASE) == protocol.BASELINE_RESULT, "Changed original 575/900 baseline")
    require(sha(HERE / "benchmark/roster.json") == ROSTER_SEAL, "Changed frozen development/full split")
    require(sha(RUN / "screen/complete.json") == SCREEN_SEAL, "Changed original screen receipt")
    config = read(SRC / "config.json")
    roster = read(HERE / "benchmark/roster.json")
    full_ids, screen_ids = roster["full_ids"], roster["screen_ids"]
    screen_set = set(screen_ids)
    archive = {r["id"]: r for r in read(BASE)["records"]}
    require(list(archive) == full_ids and len(archive) == 900 and
            sum(r["grading"]["final"]["hit"] for r in archive.values()) == 575, "Baseline mismatch")
    screen, seal = sealed_results("screen", screen_ids, config)
    official = {r["id"]: r for r in screen["records"]}
    final_sealed = False
    trailing_bytes = 0
    ledger = RUN / "predictions.jsonl"
    if screen_test:
        # Reconstruct only the original 90 prediction records; do not open the terminal ledger.
        rows = [ungraded(r) for r in screen["records"]]
        original_bytes = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in rows)
        require(hashlib.sha256(original_bytes).hexdigest() == seal["predictions_sha256"], "Screen raw hash mismatch")
        attempts = []
        source_hash = seal["predictions_sha256"]
    else:
        lines = ledger.read_bytes().splitlines(keepends=True)
        source_hash = hashlib.sha256(b"".join(lines)).hexdigest()
        require(hashlib.sha256(b"".join(lines[:len(screen_ids)])).hexdigest() == seal["predictions_sha256"],
                "Original screen ledger prefix changed")
        if lines and not lines[-1].endswith(b"\n"):
            # prepare.append commits newline-terminated JSON, so a final torn line is uncommitted.
            trailing_bytes = len(lines.pop())
        rows = [json.loads(line) for line in lines]
        attempt_file = RUN / "judge_attempts.jsonl"
        attempts = original.prepare.read_jsonl(attempt_file) if attempt_file.exists() else []
        if (RUN / "full/complete.json").exists():
            full, full_seal = sealed_results("full", full_ids, config)
            require(full_seal["model"] == seal["model"] and full_seal["predictions_sha256"] == source_hash,
                    "Sealed full results reference another model/ledger")
            require(all(full["manifest"][k] == screen["manifest"][k] for k in screen["manifest"]), "Full manifest changed")
            full_official = {r["id"]: r for r in full["records"]}
            require(all(full_official[k] == r for k, r in official.items()), "Sealed screen grades changed")
            official = full_official
            final_sealed = True

    order = screen_ids + [k for k in full_ids if k not in screen_set]
    require([r["id"] for r in rows] == order[:len(rows)] and len(rows) >= len(screen_ids),
            "Unexpected duplicate, missing, reordered or out-of-roster predictions")
    require(not final_sealed or len(rows) == 900, "Sealed final has incomplete ledger")
    original.check_records(rows, config, partial=len(rows) != 900)
    for row in rows:
        protocol.match_inputs(row, archive[row["id"]])
        if row["id"] in official:
            require(row == ungraded(official[row["id"]]), f"Sealed response differs for {row['id']}")
    # Uses the exact original final parser, length rule and fallback, and checks saved judge input hashes.
    protocol.graded(rows, attempts, config)
    reused_sealed_judgments = 0
    for row in rows:
        if row["id"] not in official:
            continue
        target = official[row["id"]]["grading"]["final"]
        actual = row["grading"]["final"]
        if actual["status"] == "pending_judge":
            require(target["status"] == "graded" and target["method"] == "local_qwen3_8b_judge_parser",
                    "Sealed grade has no matching official judge decision")
            row["grading"]["final"] = target
            reused_sealed_judgments += 1
        else:
            require(actual == target, f"Rescoring disagrees for {row['id']}")
    s = summarize(rows, archive)
    held = summarize([r for r in rows if r["id"] not in screen_set], archive)
    uncommitted = len(archive) - len(rows)
    lower, upper = s["saved_correct_range"]
    possible_total = [lower, upper + uncommitted]
    result = {
        "status": "screen_only_selftest" if screen_test else ("sealed_official_full" if final_sealed else "partial_unsealed"),
        "nominee": "r3-control", "original_baseline_total": 575, "total_items": 900,
        "ledger_sha256": source_hash, "uncommitted_trailing_bytes_excluded": trailing_bytes,
        "official_judge_attempts_read": len(attempts), "sealed_judgments_reused_without_raw_attempt": reused_sealed_judgments,
        "all_saved": s, "saved_from_810": held, "without_committed_output": uncommitted,
        "baseline_correct_on_ids_without_committed_output": 575 - s["baseline_same_ids_correct"],
        "total_correct_mathematical_range": possible_total,
        "minimum_correct_among_pending_and_unsaved_to_tie_575": max(0, 575 - lower),
        "can_still_tie_or_exceed_575": possible_total[1] >= 575,
        "semantic_joint_rate_all_saved": None,
        "semantic_evidence": "See nominee_cards_review.md for the separate five-card development review.",
    }
    for label, pack, stats in [("screen", screen, summarize(rows[:len(screen_ids)], archive))] + (
            [("full", full, s)] if final_sealed else []):
        observed = pack["summary"]
        check = {"n": stats["saved_unique"], "correct": stats["known_correct"],
                 "baseline_correct": stats["baseline_same_ids_correct"], "paired_gained": stats["gained"],
                 "paired_lost": stats["lost"], "grading_methods": stats["grading_methods"],
                 "length": stats["finish_reasons"].get("length", 0),
                 "judge_count": stats["grading_methods"].get("local_qwen3_8b_judge_parser", 0)}
        require(not stats["pending_judge"] and all(observed[k] == v for k, v in check.items()),
                f"Official {label} summary disagrees")
    if final_sealed:
        require(full["summary"]["remaining810"] == {"n": 810, "correct": held["known_correct"],
                "baseline_correct": held["baseline_same_ids_correct"]}, "Official held-out sum disagrees")
        result["official_full_summary"] = full["summary"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-test", action="store_true", help="Use only the sealed 90-row screen fixture")
    parser.add_argument("--output", type=Path, help="Optional fresh path for a JSON copy")
    args = parser.parse_args()
    report = json.dumps(audit(args.screen_test), indent=2, ensure_ascii=False)
    if args.output:
        with args.output.open("x", encoding="utf8") as out:
            out.write(report + "\n")
    print(report)
