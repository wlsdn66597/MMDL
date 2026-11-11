"""Verify train/eval image identity, group integrity and disjointness before inference."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "train"))
import train_a


def load_rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s]


def pixels(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        h = hashlib.sha256(str(rgb.size).encode())
        h.update(rgb.tobytes())
    return h.hexdigest()


def main():
    cfg = train_a.previous.load_config(train_a.engine.CFG)
    train = train_a.shared(cfg) + train_a.new_rows()
    assert len(train) == 296
    pix = {}

    def seen(path):
        path = Path(path).resolve()
        if path not in pix:
            pix[path] = pixels(path)
        return pix[path]

    trained = {seen(r["image_path"]) for r in train}
    training_ids = {r["id"] for r in train}
    prior = ROOT.parent / "era_r_only_round"
    previous = {seen(prior / r["image"]) for r in load_rows(prior/"paired/items.jsonl")}
    groups = defaultdict(set)
    roster = {}
    for corpus in ("paired", "general"):
        rs = load_rows(ROOT / corpus / "items.jsonl")
        assert len(rs) == 96 and not training_ids.intersection(r["id"] for r in rs)
        hashes = set()
        for r in rs:
            h = seen(ROOT / r["image"])
            if corpus == "paired":
                groups[r["group"]].add(h)
            hashes.add(h)
        assert not hashes & trained, f"{corpus} shares a training image"
        assert not hashes & previous, f"{corpus} repeats an R8 independent chart"
        roster[corpus] = hashes
    assert len(groups) == len(roster["paired"]) == 48 and all(len(v)==1 for v in groups.values())
    assert len(roster["general"]) == 96 and not roster["paired"] & roster["general"]
    output = dict(training_items=len(train), train_unique_rgb=len(trained), paired_unique_rgb=48,
                  general_unique_rgb=96, r8_independent_charts_checked=len(previous),
                  train_eval_rgb_overlap=0, paired_general_rgb_overlap=0, r8_chart_overlap=0)
    train_a.previous.freeze(ROOT/"split_verification.json", output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
