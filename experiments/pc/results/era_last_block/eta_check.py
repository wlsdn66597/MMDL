"""Record the requested 05:07 KST check and follow the same run to verification."""
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
KST = ZoneInfo("Asia/Seoul")
DUE = datetime(2026, 11, 11, 5, 7, tzinfo=KST)
SESSION = "era_last35_eta_0507"


def read(name):
    path = ROOT / name
    return json.loads(path.read_bytes()) if path.is_file() else {}


def save(name, data):
    path = ROOT / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
    tmp.replace(path)


def snapshot():
    state = read("status.json")
    result = read("official_verification.json")
    progress = read("official_progress.json")
    row = dict(checked_kst=datetime.now(KST).isoformat(timespec="seconds"),
               due_kst=DUE.isoformat(), state=state.get("state"), progress=progress)
    if state.get("state") == "failed":
        row.update(stage=state.get("stage"), error=state.get("error"))
    elif result.get("verified") and state.get("state") == "completed":
        summary = result["summary"]
        assert summary["n"] == 900 and summary["baseline_correct"] == 575
        row.update(correct=summary["correct"], baseline_correct=575,
                   maintained=result["maintained"], transitions=result["transitions"])
    else:
        row.update(state="awaiting_verification" if state.get("state") == "completed" else "running",
                   stage=state.get("stage"))
    return row


def main():
    if sys.argv[1:] == ["--status"]:
        print(json.dumps(snapshot()))
        return
    assert not sys.argv[1:]
    schedule = read("eta_schedule.json")
    if schedule:
        assert schedule["due_kst"] == DUE.isoformat() and schedule["session"] == SESSION
    else:
        save("eta_schedule.json", dict(due_kst=DUE.isoformat(), session=SESSION,
             registered_kst=datetime.now(KST).isoformat(timespec="seconds"), recheck_minutes=15))
    time.sleep(max(0, DUE.timestamp() - time.time()))
    while True:
        row = snapshot()
        if not (ROOT / "eta_0507.json").exists():
            save("eta_0507.json", row)
        save("eta_latest.json", row)
        print(json.dumps(row), flush=True)
        if row["state"] in {"completed", "failed"}:
            return
        time.sleep(15 * 60)


if __name__ == "__main__":
    main()
