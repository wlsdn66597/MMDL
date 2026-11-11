"""Seal a stage's exact inputs and atomically wake its tmux worker."""
import argparse
from pathlib import Path

from campaign import ROOT, read, sha, stamp, write


def main(stage):
    if stage == "train_release":
        files = ["protocol.md", "campaign.py", "train/train_a.py", "train/verify_training.py",
                 "data/rows.jsonl", "data/receipt.json"]
        files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/"data/images").glob("*.png"))]
        assert len(files) == 30, "Expected 24 training images"
    elif stage == "evaluation_release":
        files = ["check_splits.py", "paired/sources.jsonl", "paired/items.jsonl", "paired/receipt.json",
                 "general/items.jsonl", "general/receipt.json", "eval/infer.py", "eval/prepare.py",
                 "eval/grade.py", "eval/score.py", "eval/answer_only.py", "eval/validate.py"]
        for corpus in ("paired", "general"):
            files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/corpus/"images").iterdir()) if p.is_file()]
        assert len(files) >= 12+48+96, "Missing evaluation image files"
    else:
        files = ["eval/review/manual.jsonl", "eval/review/audit_packet.jsonl"]
    hashes = {name:sha(ROOT/name) for name in files}
    dest = ROOT/"signals"/(stage+".json")
    if dest.exists():
        assert read(dest)["hashes"] == hashes
    else:
        write(dest,dict(hashes=hashes,time_kst=stamp()))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=["train_release", "evaluation_release", "manual_review"])
    main(cli.parse_args().stage)
