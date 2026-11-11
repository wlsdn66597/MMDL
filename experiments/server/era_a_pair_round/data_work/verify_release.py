"""Recheck uploaded row/image bytes; add independent content checks on the source workstation."""
import argparse
from collections import Counter
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def load(file):
    return json.loads(file.read_text(encoding="utf8"))


def rows(file):
    return [json.loads(line) for line in file.read_text(encoding="utf8").splitlines()]


def verify_assets():
    receipt = load(HERE / "receipt.json")
    path = HERE / "train.jsonl"
    train = rows(path)
    assert digest(path) == receipt["train_sha256"]
    assert len(train) == len({r["id"] for r in train}) == receipt["n"]
    for r in train:
        picture = (HERE / r["image"]).resolve()
        assert picture.is_relative_to((HERE / "images").resolve())
        assert digest(picture) == r["image_sha256"]
        assert r["chosen_application"] != r["rejected_application"]
    return train, dict(items=len(train), populated_images=len({r["image"] for r in train}), train_sha256=digest(path))


def numeric(text):
    return D(re.findall(r"\d+(?:\.\d+)?", text)[-1])


def audit(rows_all):
    previous = HERE.parents[2] / "results/era_single_call_campaign/data/train.jsonl"
    original = {r["id"]: r for r in rows(previous)}
    holdout = load(HERE.parent / "connection_work/spec/gold.json")
    labels = sorted({s for i in holdout for s in i["marks"]})
    counts = Counter()
    for r in rows_all:
        text = " ".join(r[k] for k in ("prompt","evidence","rule","chosen_application","rejected_application"))
        assert not any(re.search(r"\b" + re.escape(s) + r"\b", text, re.I) for s in labels)
        options = dict(re.findall(r"^([A-E])\. ([^\n]+)$", r["prompt"], re.M))
        gold = r["final_answer"]
        if options:
            assert gold in options
            counts["multiple_choice"] += 1
        if r["id"] in original:
            source = original[r["id"]]
            for k in ("prompt","evidence","rule","image_sha256"):
                assert r[k] == source[k]
            assert gold == source["answer"]
            counts["original_question_ERA_final_preserved"] += 1
            continue
        table = {s: D(v) for s, v in r["table"].items()}
        ordered = sorted(table, key=table.get)
        mode = r["contrast"]
        positive, negative = r["chosen_application"], r["rejected_application"]
        if mode.endswith("_selection"):
            wanted = ordered[:3] if mode.startswith("smallest") else list(reversed(ordered))[:3]
            selected = re.findall(r"(\w+) \(([\d.]+)\)", positive)
            alternate = re.findall(r"(\w+) \(([\d.]+)\)", negative)
            assert [s for s, _ in selected] == wanted and {s for s,_ in alternate} != set(wanted)
            assert all(table[s] == D(v) for s,v in selected + alternate)
            right = sum(table[s] for s in wanted)
            assert sum(table[s] for s,_ in alternate) == numeric(negative) != right
        elif mode == "rank_difference":
            top, second = ordered[-1], ordered[-2]
            right = table[top] - table[second]
            assert positive.startswith(f"{top} is highest") and f"and {second} is next" in positive
            alt = re.search(r"and (\w+) is next", negative)[1]
            assert alt != second and table[top] - table[alt] == numeric(negative) != right
        else:
            operands = re.search(r"value for (\w+) and (\w+)\?", r["question"]).groups()
            left, right_name = operands
            right = table[left] + table[right_name]
            for app, expected in [(positive, [table[left], table[right_name]]), (negative, [table[right_name], table[left]])]:
                a, b = re.search(r"contributes ([\d.]+) and \w+ contributes ([\d.]+),", app).groups()
                assert [D(a), D(b)] == expected and numeric(app) == right
            assert table[left] != table[right_name]
            counts["same_total_wrong_row_binding"] += 1
        assert right == D(options.get(gold, gold)) == numeric(positive)
        counts["source_table_recomputed"] += 1
    return dict(counts=counts, diagnostic_label_count=len(labels), diagnostic_labels_seen_in_train_text=0,
                curated_review_count=sum(r["contrast"]=="item_specific_binding" for r in rows_all),
                curated_review="Each chosen and rejected binding inspected against archived E/R, Q and option text (or open answer). Source images were originally reviewed; no new human audit over all pixels.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--audit", action="store_true")
    cli.add_argument("--out", type=Path)
    args = cli.parse_args()
    data, outcome = verify_assets()
    if args.audit:
        outcome.update(audit(data))
    result = json.dumps(outcome, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_bytes(result.encode("utf8"))
    print(result, end="")
