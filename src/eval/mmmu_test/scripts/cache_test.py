"""Warm the exact test Arrow cache used by the unchanged offline image loader."""
import json
from pathlib import Path


def main():
    from datasets import load_dataset
    root = Path(__file__).resolve().parent.parent
    cfg = json.loads((root / "eval_settings.json").read_text(encoding="utf-8"))["settings"]["dataset"]
    ids = []
    for subject in cfg["configs"]:
        ds = load_dataset(cfg["repo"], subject, split="test", revision=cfg["revision"])
        ids.extend(ds["id"])
        print(f"Cached {subject}: {len(ds)}", flush=True)
    if len(ids) != 10500 or len(set(ids)) != 10500:
        raise ValueError(f"Expected 10500 unique Test IDs, got {len(ids)} rows / {len(set(ids))} unique")
    print("Cached 10500 unique Test images/records for offline generation.")


if __name__ == "__main__":
    main()
