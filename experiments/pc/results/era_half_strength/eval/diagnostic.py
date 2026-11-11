"""Reuse sealed R9 grades and judge previously unseen single-call responses."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R9 = ROOT.parent / "era_a_only_round"
REVIEW = HERE / "review"
OLD = R9 / "eval"
ARMS = ("ordinary", "enhanced")
SCORED = "4a7cb52f661642a66b9b527c69901ad20f04350277da19edf77ed755fb396b23"
SUMMARY = "683bdc873df9b67a106d4e7d0dc7fe6ca71454bd54a6fe937ad7482a82d9096f"
sys.path.insert(0, str(OLD))
import grade as judge
import score as rubric
import audit_files as shards

read, lines, sha = judge.data, judge.lines, judge.sha


def immutable(name, obj, *, rows=False):
    judge.freeze(REVIEW / name, judge.json_lines(obj) if rows else judge.json_bytes(obj))


def archive():
    prev = OLD / "review"
    decision = read(R9 / "decision.json")
    assert sha(prev / "scored.jsonl") == decision["scored_sha256"] == SCORED
    assert sha(prev / "summary.json") == decision["summary_sha256"] == SUMMARY
    packet = read(prev / "packet.json")
    spec = OLD / "spec"
    assert sha(spec / "inputs.jsonl") == packet["inputs_sha256"]
    assert sha(spec / "gold.jsonl") == packet["gold_sha256"]
    assert sha(prev / "packet.jsonl") == packet["packet_sha256"]
    old_scores = lines(prev / "scored.jsonl")
    cached = {}
    for r in old_scores:
        content = {k:v for k,v in r.items() if k != "arm"}
        assert cached.setdefault(r["key"],content) == content
    cards = {r["key"]:r for r in lines(prev / "packet.jsonl")}
    assert cards.keys() == cached.keys()
    return old_scores, cached, cards, lines(spec / "inputs.jsonl"), lines(spec / "gold.jsonl")


def predictions(arm, inputs):
    run = HERE / "runs" / arm
    journal = HERE / "fresh/runs" / ("half-"+arm)
    pred = run / "predictions.jsonl"
    result = lines(pred)
    receipt = read(run / "receipt.json")
    done = read(journal / "done.json")
    manifest = read(journal / "manifest.json")
    assert done["output_sha256"] == sha(journal / "predictions.jsonl")
    assert done["manifest_sha256"] == sha(journal / "manifest.json")
    assert not (journal / "submitted.json").exists()
    assert result == lines(journal / "predictions.jsonl")
    assert len(result) == len(inputs) == len({r["id"] for r in result}) == receipt["fresh"] == done["n"] == 192
    assert receipt["sha256"] == sha(pred) and receipt["arm"] == arm
    assert manifest["request_seed"] == 3407 and manifest["max_output"] == 2048
    prior = read(OLD / "fresh/runs" / ("r9-"+arm) / "manifest.json")
    for key in ("inputs_sha256","config_sha256","engine_sha256","runner_sha256","batch","request_seed","max_output"):
        assert manifest[key] == prior[key], (arm,key)
    for out, expected in zip(result,inputs):
        assert all(out[k] == expected[k] for k in ("id","prompt","image_sha256"))
        assert out["seed"] == 3407 and out["max_output"] == 2048
    return result


def packet():
    _, cache, old_cards, inputs, gold = archive()
    new = {}
    cached = set()
    proofs = {}
    for arm in ARMS:
        pred = predictions(arm,inputs)
        proofs[arm] = dict(n=len(pred), sha256=sha(HERE/"runs"/arm/"predictions.jsonl"))
        for out, x, g in zip(pred,inputs,gold):
            assert x["id"] == g["id"] and x["dataset"] == g["dataset"]
            key = judge.review_key(x["id"],out["raw"],out["finish_reason"])
            card = dict(key=key,id=x["id"],dataset=x["dataset"],group=g["group"],kind=g["kind"],
                    question=g["question"],prompt=x["prompt"],image=x["image"],image_sha256=x["image_sha256"],
                    options=g["options"],gold=g["gold"],reference=g["reference"],raw=out["raw"],finish=out["finish_reason"])
            if key in cache:
                assert card == old_cards[key]
                cached.add(key)
            else:
                assert new.setdefault(key,card) == card
    immutable("cached.json",dict(r9_scored_sha256=SCORED,keys=sorted(cached)))
    immutable("packet.jsonl",[new[k] for k in sorted(new)],rows=True)
    meta = dict(n_responses=384,unique=len(new),per_arm=proofs,unique_including_cache=len(new)+len(cached),
                cached_keys=len(cached),cached_sha256=sha(REVIEW/"cached.json"),packet_sha256=sha(REVIEW/"packet.jsonl"),
                inputs_sha256=sha(OLD/"spec/inputs.jsonl"),gold_sha256=sha(OLD/"spec/gold.jsonl"))
    immutable("packet.json",meta)
    print(meta,flush=True)


def run_judge():
    original = read(OLD/"review/judge_identity.json")
    assert hashlib.sha256(judge.RUBRIC.encode()).hexdigest() == original["rubric_sha256"]
    assert sha(Path(judge.__file__)) == original["program"]
    judge.REVIEW = REVIEW
    if read(REVIEW/"packet.json")["unique"]:
        judge.grade()
        current = read(REVIEW/"judge_identity.json")
        assert current["judge_files"] == original["judge_files"] and current["config"] == original["config"]
    else:
        immutable("judge_identity.json",dict(all_cached=True,original_rubric=original["rubric_sha256"]))
        immutable("judgments.jsonl",[],rows=True)
        immutable("judge_complete.json",dict(n=0,packet_sha256=sha(REVIEW/"packet.jsonl"),
                 judgments_sha256=sha(REVIEW/"judgments.jsonl"),identity_sha256=sha(REVIEW/"judge_identity.json")))


def parser():
    assert sha(rubric.LIB) == "43ed42de0b9e5097838d0d93043afa053e8df3aa7a4f39d4dcfa2fde2ce2ecc2"
    spec = importlib.util.spec_from_file_location("archived_era_parser",rubric.LIB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def answer_snapshot():
    _, cache, _, inputs, gold = archive()
    sc = parser()
    result = {}
    for arm in ARMS:
        counts = defaultdict(Counter)
        for p,g in zip(predictions(arm,inputs),gold):
            letter = rubric.as_choice(p["raw"],g["options"],sc)
            if not letter and rubric.key(p) in cache:
                letter = cache[rubric.key(p)]["choice"]
            bucket = "wrong" if p["finish_reason"] != "stop" else "unknown" if not letter else "correct" if letter==g["gold"] else "wrong"
            counts[g["dataset"]][bucket] += 1
        result[arm] = counts
    immutable("provisional_answers.json",result)
    print(result)


def records(final=False):
    old, cache, _, inputs, gold = archive()
    seal = read(REVIEW/"judge_complete.json")
    proof = read(REVIEW/"packet.json")
    assert sha(REVIEW/"cached.json")==proof["cached_sha256"]
    cached = set(read(REVIEW/"cached.json")["keys"])
    assert read(REVIEW/"cached.json")["r9_scored_sha256"]==SCORED
    for f,k in (("packet.jsonl","packet_sha256"),("judgments.jsonl","judgments_sha256"),("judge_identity.json","identity_sha256")):
        assert sha(REVIEW/f)==seal[k]
    cards = {r["key"]:r for r in lines(REVIEW/"packet.jsonl")}
    grades = {r["key"]:r for r in lines(REVIEW/"judgments.jsonl")}
    assert cards.keys()==grades.keys() and not cards.keys() & cached
    assert len(cards)==proof["unique"]==seal["n"]
    for r in grades.values():
        assert r["identity"] == seal["identity_sha256"]
    patches = {}
    if final:
        required = {r["key"] for r in lines(REVIEW/"audit_packet.jsonl")}
        patches = {r["key"]:r for r in lines(REVIEW/"manual.jsonl")}
        assert required == patches.keys() and len(required)==len(lines(REVIEW/"manual.jsonl"))
        for k,n in patches.items():
            assert n["reason"].strip() and set(n["fields"])<=set((*rubric.FIELDS,"choice"))
            for f,v in n["fields"].items():
                assert v in ["?",*"ABCDEFGH"] if f=="choice" else v is None or type(v) is bool
            grades[k].update(n["fields"])
    bare = {k for k,c in cards.items() if rubric.only_options(c["raw"],c["options"])}
    sc = parser()
    rows = []
    used = set()
    for arm in ARMS:
        assert sha(HERE/"runs"/arm/"predictions.jsonl")==proof["per_arm"][arm]["sha256"]
        for p,g in zip(predictions(arm,inputs),gold):
            k = rubric.key(p)
            used.add(k)
            if k in cached:
                rows.append(dict(cache[k],arm=arm+"50"))
                continue
            z = grades[k]
            if k in bare:
                z.update(e=False,r=False,a=False,intermediate=False,winner=False,first=None,corrected=None)
            extracted = rubric.as_choice(p["raw"],g["options"],sc)
            choice = extracted or z["choice"]
            ans = p["finish_reason"]=="stop" and choice==g["gold"]
            application = rubric.every([z["a"],z["intermediate"],z["winner"]])
            joint = rubric.every([ans,z["e"],z["r"],application,z["consistent"]])
            first_error = "no_grounded_answer" if k in bare else patches.get(k,{}).get("first_error")
            if final and g["dataset"]=="paired" and k not in bare:
                assert first_error in rubric.ERRORS and (first_error=="none")==(joint is True),(g["id"],first_error)
            if first_error=="loser_rounding":
                assert ans and all(z[f] is True for f in ("e","r","winner","consistent")) and z["intermediate"] is False
            rows.append(dict(id=p["id"],key=k,arm=arm+"50",group=g["group"],kind=g["kind"],dataset=g["dataset"],
                     sentinel=g["sentinel"],choice=choice,fallback=extracted is None,answer=ans,joint=joint,bare=k in bare,
                     format=sc.format_ok(p["raw"],p["finish_reason"]),finish=p["finish_reason"],
                     **{f:application if f=="a" else z[f] for f in rubric.FIELDS},a_judgment=z["a"],
                     er=rubric.every([z["e"],z["r"]]),first_error=first_error,rounding_joint=True if first_error=="loser_rounding" else joint))
    assert len(rows)==len({(r["id"],r["arm"]) for r in rows})==384 and used==cards.keys() | cached
    return old+rows,cards,grades,bare


def audit():
    rows,cards,grades,bare = records()
    samples = sorted({r["id"] for r in rows if r["dataset"]=="general"},
                     key=lambda i:hashlib.sha256(("r9-general-sample/"+i).encode()).hexdigest())[:8]
    by_id=defaultdict(list)
    for r in rows: by_id[r["id"]].append(r)
    selected=set()
    for i,rs in by_id.items():
        assert len(rs)==5
        require = (rs[0]["dataset"]=="paired" or rs[0]["sentinel"] or i in samples or
                   any(r["fallback"] or r["corrected"] is True for r in rs) or len({(r["answer"],r["joint"]) for r in rs})>1)
        if require: selected.update(r["key"] for r in rs if r["key"] in cards and r["key"] not in bare)
    immutable("audit_packet.jsonl",[dict(cards[k],judge={f:grades[k][f] for f in (*rubric.FIELDS,"choice")})
                                    for k in sorted(selected)],rows=True)
    immutable("bare.jsonl",[dict(key=k,id=cards[k]["id"],raw=cards[k]["raw"]) for k in sorted(bare)],rows=True)
    immutable("audit_receipt.json",dict(n=len(selected),novel=len(cards),sha256=sha(REVIEW/"audit_packet.jsonl")))
    shards.REVIEW,shards.ROOT = REVIEW,ROOT
    shards.split()


def main():
    commands={"packet":packet,"judge":run_judge,"answers":answer_snapshot,"audit":audit}
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=(*commands,"merge","score"))
    stage=cli.parse_args().stage
    if stage=="merge":
        shards.REVIEW=REVIEW
        shards.merge()
    elif stage=="score":
        from half_report import summarize
        rows,_,_,_=records(final=True)
        summarize(rows)
    else:
        commands[stage]()


if __name__=="__main__":
    main()
