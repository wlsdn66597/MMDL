"""Record an ETA check, then follow the existing reporter through verified completion."""
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
KST = ZoneInfo("Asia/Seoul")


def save(name, entry):
    path = ROOT/name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    tmp.replace(path)


def read(name):
    path = ROOT/name
    return json.loads(path.read_bytes()) if path.is_file() else {}


def inspect(root_state, ready, verified, progress):
    if root_state.get("state") == "failed":
        return {"state":"failed", "stage":root_state.get("stage"), "error":root_state.get("error")}
    if ready and verified.get("verified") is True and root_state.get("state") == "completed":
        summary = verified["summary"]
        assert summary["n"] == 900 and summary["baseline_correct"] == 575
        return {"state":"completed", "correct":summary["correct"], "baseline_correct":575,
                "maintained":summary["correct"] >= 575, "transitions":verified["transitions"],
                "archive_sha256":ready["sha256"]}
    return {"state":"waiting_for_verified_report", "stage":root_state.get("stage"),
            **{k:progress.get(k) for k in ("generated","resolved","pending_judge","resolved_correct","resolved_baseline","time_kst")}}


def main(due):
    when = datetime.fromisoformat(due)
    if when.utcoffset() is None:
        raise ValueError("An explicit timezone is required")
    target = when.astimezone(KST).isoformat(timespec="seconds")
    prior = read("eta_schedule.json")
    if prior and prior["due_kst"] != target:
        raise ValueError("A different ETA checkpoint is already registered")
    if not prior:
        save("eta_schedule.json",{"registered_kst":datetime.now(KST).isoformat(timespec="seconds"),
                                 "due_kst":target,"session":"mmmu_eta_check","recheck_interval_minutes":15})
    time.sleep(max(0,when.timestamp()-time.time()))
    while True:
        result = inspect(read("status.json"),read("final_ready.json"),read("verification.json"),read("progress.json"))
        checked = datetime.now(KST).isoformat(timespec="seconds")
        snapshot = {"due_kst":target,"checked_kst":checked,**result}
        save("eta_check.json", snapshot)
        with (ROOT/"eta_checks.jsonl").open("a",encoding="utf8") as f:
            f.write(json.dumps(snapshot,ensure_ascii=False) + "\n")
        if result["state"] in ("completed","failed"):
            return
        time.sleep(900)


if __name__ == "__main__":
    main(sys.argv[1])
