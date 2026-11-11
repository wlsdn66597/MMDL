"""Replay official saved hit/length outcomes on identical MMMU inputs; zero model calls."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys


OUT = Path(__file__).resolve().parent
RESULTS = OUT.parents[1]
OLD = RESULTS / "mmmu900_epochs_20261031/evidence"
R4 = RESULTS / "era_a_pair_round"
sys.path.insert(0, str(R4))
from analysis_round import strict_era


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def right(row):
    return bool(row["grading"]["final"]["hit"])


def length(row):
    return row["model_response"]["finish_reason"] == "length"


def transition(a, b, ids):
    pairs = Counter((right(a[k]), right(b[k])) for k in ids)
    return dict(n=len(ids), both_correct=pairs[True,True], gained=pairs[False,True],
                lost=pairs[True,False], both_wrong=pairs[False,False])


def summarize(rows):
    return dict(n=len(rows), correct=sum(right(r) for r in rows), strict_era=sum(strict_era(r) for r in rows),
                length=sum(length(r) for r in rows))


def main():
    paths = dict(base=RESULTS/"mmmu_first_baseline_20261009/baseline_seed3407/results.json",
                 e2=OLD/"epoch_2_seed3407/results.json", e3=OLD/"epoch_3_seed3407/results.json",
                 e4=OLD/"epoch_4_seed3407/results.json", control=R4/"control_recovery/score/results.json",
                 pair=R4/"mmmu/candidate/candidate/full/results.json")
    packs = {name:load(p) for name,p in paths.items()}
    scores = {}
    for name, pack in packs.items():
        rows = pack["records"]
        scores[name] = {r["id"]:r for r in rows}
        assert len(rows) == len(scores[name]) == (899 if name=="control" else 900)
        for row in rows:
            assert row["grading"]["final"]["status"]=="graded" and row["grading"]["final"]["hit"] in (0,1)
    base = scores["base"]
    assert set(base) - scores["control"].keys() == {"validation_Mechanical_Engineering_11"}
    config = packs["base"]["manifest"]["configuration"]
    fields = ("subject","dataset_index","dataset_config","question_type","question","options","reference_answer",
              "image_fields","original_image_sizes_wh","input_prompt","rendered_chat")
    for name,pack in packs.items():
        assert pack["manifest"]["configuration"] == config
        for rid,row in scores[name].items():
            assert all(row[k]==base[rid][k] for k in fields)
            assert row["model_response"]["input_tokens"] == base[rid]["model_response"]["input_tokens"]
    coverage = load(R4/"control_recovery/score/coverage_seal.json")
    cand_seal = load(R4/"mmmu/candidate/candidate/full/complete.json")
    assert sha(paths["control"]) == coverage["outputs"]["results.json"]
    assert sha(paths["pair"]) == cand_seal["outputs"]["results.json"]
    roster = load(R4/"mmmu/candidate/roster.json")
    screen = set(roster["screen_ids"])
    common = [k for k in base if k in scores["control"]]
    subsets = {"all_available":set(base),"common899":set(common),"screen90":screen,"outside_screen":set(base)-screen}
    summary = {"source_sha256":{name:sha(p) for name,p in paths.items()},
               "metrics":{sub:{name:summarize([r for k,r in s.items() if k in ids]) for name,s in scores.items()}
                          for sub,ids in subsets.items()}, "transitions":{}, "donor":{}}
    for a,b in [("base",k) for k in scores if k!="base"] + [
            ("e2","e3"),("e3","e4"),("e3","pair"),("e3","control"),("control","pair")]:
        ids = sorted(scores[a].keys() & scores[b].keys())
        summary["transitions"][a+"/"+b] = transition(scores[a],scores[b],ids)
    for arm in ("pair","control"):
        rows = scores[arm]
        breakdown = {}
        for key,b_hit,d_hit in (("base+_donor+",True,True),("base+_donor-",True,False),
                                ("base-_donor+",False,True),("base-_donor-",False,False)):
            ids=[k for k in rows if right(base[k])==b_hit and right(scores["e3"][k])==d_hit]
            breakdown[key]=dict(n=len(ids),arm_right=sum(right(rows[k]) for k in ids))
        summary["donor"][arm] = breakdown
        for before in ("base","e3"):
            groups = {}
            for a,b in ((False,False),(False,True),(True,False),(True,True)):
                ids=[k for k in rows if length(scores[before][k])==a and length(rows[k])==b]
                groups[f"{'length' if a else 'stop'}/{'length' if b else 'stop'}"]=transition(scores[before],rows,ids)
            summary.setdefault("length_transitions",{})[before+"/"+arm] = groups
    token_bins = ((1,128),(129,512),(513,2048),(2049,32768))
    summary["original_correct_token_bins"] = {}
    for low,high in token_bins:
        ids = [k for k in common if right(base[k]) and low<=base[k]["model_response"]["output_tokens"]<=high]
        summary["original_correct_token_bins"][f"{low}-{high}"] = dict(n=len(ids),lost={
            name:sum(not right(scores[name][k]) for k in ids) for name in ("e2","e3","e4","control","pair")})
    summary["original_correct_within_subject"] = {}
    for subject in sorted({r["subject"] for r in base.values()}):
        eligible = [k for k in common if base[k]["subject"]==subject and right(base[k])]
        groups = {"short": [k for k in eligible if base[k]["model_response"]["output_tokens"]<=512],
                  "long": [k for k in eligible if base[k]["model_response"]["output_tokens"]>512]}
        if all(groups.values()):
            summary["original_correct_within_subject"][subject] = {group:dict(n=len(ids),
                paired_lost=sum(not right(scores["pair"][k]) for k in ids)) for group,ids in groups.items()}
    print(json.dumps(summary,indent=2))
    (OUT/"stages.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    main()
