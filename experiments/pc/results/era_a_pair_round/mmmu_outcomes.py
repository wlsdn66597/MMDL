"""CPU replay of saved grades, paired contrasts, and blinded answer-extraction audit."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

from analysis_round import arm_results, read, strict_era

ROOT = Path(__file__).resolve().parent
OLD = Path.home() / "era_single_call_campaign"
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(Path.home() / "era_decoder_lora_20261030/baseline/src"))
import mmmu_eval as official

JUDGE = "local_qwen3_8b_judge_parser"


def hit(row):
    return row["grading"]["final"]["hit"]


def reason(row):
    return row["model_response"]["finish_reason"]


def method(row):
    return row["grading"]["final"].get("method", "unresolved")


def mcnemar(gain, loss):
    total = gain + loss
    return min(1., 2 * sum(math.comb(total, i) for i in range(min(gain, loss) + 1)) / (2 ** total))


def paired(pairs):
    out = Counter()
    for x, y in pairs:
        a, b = hit(x), hit(y)
        if a is None or b is None:
            out["pending"] += 1
        else:
            out[("both_right" if b else "lost") if a else ("gained" if b else "both_wrong")] += 1
        for label, v in (("left", a), ("right", b)):
            out[label] += int(v == 1)
            out[label + "_pending"] += int(v is None)
        out["n"] += 1
    result = {k:out[k] for k in ("n","left","right","left_pending","right_pending","pending","both_right","gained","lost","both_wrong")}
    if not out["pending"]:
        result["mcnemar_exact_twosided"] = mcnemar(out["gained"],out["lost"])
    return result


def details(pairs):
    parts = {}
    for key, func in (("types",lambda x,y:x["question_type"]), ("subjects",lambda x,y:x["subject"]),
                      ("length",lambda x,y:reason(x)+"/"+reason(y)),
                      ("scoring_routes",lambda x,y:method(x)+"/"+method(y)),
                      ("judge_involved",lambda x,y:"yes" if JUDGE in (method(x),method(y)) else "no")):
        parts[key] = {label:paired([(x,y) for x,y in pairs if func(x,y)==label])
                      for label in sorted({func(x,y) for x,y in pairs})}
    return parts


def held_groups(root, arm, rows, baseline):
    screen = set(read(root / "mmmu" / arm / "roster.json")["screen_ids"])
    parts = {}
    for key, ids in (("screen", screen), ("outside_screen",set(rows)-screen), ("all",set(rows))):
        pairs = [(baseline[k],rows[k]) for k in sorted(set(rows)&ids)]
        summary = paired(pairs)
        summary["strict_baseline"] = sum(strict_era(x) for x,_ in pairs)
        summary["strict_adapted"] = sum(strict_era(y) for _,y in pairs)
        summary["strict_and_correct"] = sum(strict_era(y) and hit(y)==1 for _,y in pairs)
        parts[key] = summary
    return parts


def audit_queue(pairs):
    # Prepare a source-ID-ranked sample independently inside each scoring route/direction.
    def order(rid):
        return hashlib.sha256(("r4/extraction-audit/20261107/" + rid).encode()).hexdigest()
    queue, key, universe = [], {}, {}
    for gain in (True, False):
        for judge in (True, False):
            group = []
            for x,y in pairs:
                if not (hit(x)==int(not gain) and hit(y)==int(gain)):
                    continue
                if any(reason(r)!="stop" or len(r["model_response"]["raw"])>9000 for r in (x,y)):
                    continue
                if (JUDGE in (method(x),method(y)))==judge:
                    group.append((x,y))
            universe[str((gain,judge))] = len(group)
            for x,y in sorted(group,key=lambda p:order(p[0]["id"]))[:2]:
                rid = x["id"]
                card = dict(id=rid,question=x["question"],options=x["options"],image_fields=x["image_fields"],answers={})
                for label, record in (("original",x),("candidate",y)):
                    blind = hashlib.sha256((rid+"/v4/anonymous/"+label).encode()).hexdigest()[:8]
                    card["answers"][blind]=record["model_response"]["raw"]
                    key[blind]=dict(id=rid,model=label,hit=hit(record),gold=x["reference_answer"],
                                    official_extracted=record["grading"]["final"].get("extracted_answer"),method=method(record))
                card["answers"]=dict(sorted(card["answers"].items()))
                queue.append(card)
    queue.sort(key=lambda c:order(c["id"]))
    return queue, key, universe


def gather(final, root=ROOT):
    path = Path.home() / "mmmu_baseline_20261009/baseline_seed3407/results.json"
    assert official.sha(path)==official.BASELINE_RESULT
    original = {r["id"]:r for r in read(path)["records"]}
    assert len(original)==900 and sum(map(hit,original.values()))==575
    config = read(Path.home() / "era_decoder_lora_20261030/baseline/src/config.json")
    summary, records, seals = {}, {}, {}
    for arm in ("candidate","control"):
        if not final and arm=="control":
            continue
        official_stats, rows = arm_results(root,arm,original,config,official)
        if not rows:
            continue
        book = root / "mmmu" / arm / arm
        path = book / "full/complete.json"
        seals[arm] = official.sha(path) if path.is_file() else None
        records[arm]=rows
        pairs = [(original[k],rows[k]) for k in rows]
        summary[arm]=dict(groups=held_groups(root,arm,rows,original),details=details(pairs),official_full=bool(seals[arm]),
                          observed_potential_total=official_stats["all_saved"]["possible_total900"])
    if "control" in records:
        c, p = records["control"],records["candidate"]
        common = sorted(c.keys() & p.keys())
        screen = set(read(root/"mmmu/candidate/roster.json")["screen_ids"])
        summary["candidate_vs_control"] = {}
        for name, ids in (("shared",common),("screen",[k for k in common if k in screen]),
                          ("outside_screen",[k for k in common if k not in screen])):
            pairs = [(c[k],p[k]) for k in ids]
            summary["candidate_vs_control"][name]=dict(paired=paired(pairs),strict_control=sum(strict_era(x) for x,_ in pairs),
                                                     strict_candidate=sum(strict_era(y) for _,y in pairs))
    if "candidate" in records:
        pairs=[(original[k],r) for k,r in records["candidate"].items()]
        queue,key,universe = audit_queue(pairs)
        summary["audit_universe"] = universe
        official.save(root/"reviews/change_blind.json",queue)
        official.save(root/"reviews/change_key.json",key)
    summary["seals"]=seals
    official.save(root/("mmmu_findings.json" if final else "candidate_findings.json"),summary)
    print(json.dumps({arm:data["groups"]["all"] for arm,data in summary.items() if arm in ("candidate","control")}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--terminal",action="store_true")
    gather(parser.parse_args().terminal)
