"""Prepare a separate, outcome-blind review of the five prespecified MMMU source cards."""
import hashlib
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
old = Path.home() / "era_single_call_campaign"
sys.path.insert(0, str(old))
import connection
import mmmu_eval


def make_packet():
    paths = dict(base=root/"mmmu/control/screen_baseline.json",
                 control=root/"mmmu/control/control/screen/results.json",
                 paired=root/"mmmu/candidate/candidate/screen/results.json")
    reviews, key = [], []
    for label, path in paths.items():
        for r in mmmu_eval.read(path)["records"]:
            rid = r["id"]
            if rid not in connection.CARDS:
                continue
            raw = r["model_response"]["raw"]
            mask = hashlib.sha256(f"r4-audit/{rid}/{label}".encode()).hexdigest()[:12]
            reviews.append(dict(id=rid,blind_id=mask,question=r["question"],options=r["options"],
                                card=connection.CARDS[rid],raw=raw))
            key.append(dict(id=rid,blind_id=mask,arm=label,official=r["grading"]["final"]["hit"]))
    if len(reviews) != 15:
        raise ValueError("Expected five image-backed cards for each arm")
    for file, rows in (("blind_mmmu.json", reviews),("blind_key.json", key)):
        mmmu_eval.freeze(root/"reviews"/file, sorted(rows,key=lambda r:(r["id"],r["blind_id"])))


if __name__ == "__main__":
    make_packet()
