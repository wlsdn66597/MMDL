"""Describe retention by the unadapted model's response length on its correct items."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent
FILES = {
    "base": RESULTS/"mmmu_first_baseline_20261009/baseline_seed3407/results.json",
    "control": RESULTS/"era_a_pair_round/control_recovery/score/results.json",
    "paired": RESULTS/"era_a_pair_round/mmmu/candidate/candidate/full/results.json",
}


def read_records(path):
    with path.open(encoding="utf8") as stream:
        rows = json.load(stream)["records"]
    keyed = {row["id"]: row for row in rows}
    assert len(rows) == len(keyed)
    return keyed


def tokens(row):
    return row["model_response"]["output_tokens"]


def hit(row):
    return bool(row["grading"]["final"]["hit"])


def describe(ids, model):
    chosen = [model[i] for i in ids]
    errors = [r for r in chosen if not hit(r)]
    return dict(n=len(chosen),lost=len(errors), lost_stopped=sum(r["model_response"]["finish_reason"]=="stop" for r in errors),
                median_tokens=median(tokens(r) for r in chosen))


def main():
    models = {name: read_records(path) for name,path in FILES.items()}
    base = models["base"]
    same = set(base).intersection(models["control"], models["paired"])
    ids = sorted((i for i in same if hit(base[i])),key=lambda i:(tokens(base[i]),i))
    assert (len(same),len(ids)) == (899,575)
    groups = [ids[j*len(ids)//4:(j+1)*len(ids)//4] for j in range(4)]
    report = []
    for group in groups:
        report.append(dict(base_output_token_range=[tokens(base[group[0]]),tokens(base[group[-1]])],
                           n=len(group), base_median_tokens=median(tokens(base[i]) for i in group),
                           control=describe(group,models["control"]),paired=describe(group,models["paired"])))
    stats = {}
    for name in ("control","paired"):
        m = models[name]
        stats[name] = Counter("shorter" if tokens(m[i])<tokens(base[i]) else "same_or_longer" for i in ids)
    stability_path = RESULTS/"mmmu_multiseed_20261011/item_stability.json"
    stability = {r["id"]:r for r in json.loads(stability_path.read_text(encoding="utf8"))}
    assert stability.keys() == base.keys()
    for i, item in stability.items():
        assert set(item["hits"]) == {"3407","42","123","2026","9876"}
        assert item["hits"]["3407"] == hit(base[i])
    stability_rows = []
    for count in range(1,6):
        group = [i for i in ids if stability[i]["correct_seed_count"]==count]
        if not group:
            continue
        stability_rows.append(dict(original_correct_across_five_seeds=count,n=len(group),
                                   control=describe(group,models["control"]),paired=describe(group,models["paired"])))
    stable_quartiles = []
    for group in groups:
        chosen = [i for i in group if stability[i]["correct_seed_count"]==5]
        stable_quartiles.append(dict(n=len(chosen),control=describe(chosen,models["control"]),
                                     paired=describe(chosen,models["paired"])))
    receipt = dict(cohort="Originally correct, common to all three models. Quartiles sorted by (base output tokens, ID); no split by adapted outcome.",
                   inputs={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in FILES.items()},
                   rows=report, lengths_on_base_correct=stats,
                   base_seed_stability_sha256=hashlib.sha256(stability_path.read_bytes()).hexdigest(),
                   baseline_correct_repeats=stability_rows,unanimous_base_within_same_four_bins=stable_quartiles,
                   limitation="Token count is observed after generation. Subject, item difficulty, and reasoning style can confound the relationship.")
    (HERE/"response_length.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
