"""Blind scoring for the one-sentence cue. All solver generations are read-only."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HALF = ROOT.parent / "era_half_strength"
REVIEW = HERE / "review"
sys.path.insert(0, str(HALF / "eval"))
import diagnostic as half

judge, rubric, shards = half.judge, half.rubric, half.shards
read, lines, sha = half.read, half.lines, half.sha
ARMS = ("base", "ordinary", "enhanced")
ORIGINAL = {"base":"base", "ordinary":"ordinary50", "enhanced":"enhanced50"}
HALF_SCORED = "85db0dc94b7ac345988a6e69c82230f288a9df21bd1a8b09dc12d8be816a44ca"
HALF_SUMMARY = "58f6fb9bb8c3d5855089b1e807504d9a39672ec49302e5e7e81fdcfc30859fad"
CUE = "State the relevant evidence from the image, the rule you use, and how it leads to your choice. Keep each part brief."
TERMINAL = 'On the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'


def immutable(name, obj, *, rows=False):
    judge.freeze(REVIEW/name, judge.json_lines(obj) if rows else judge.json_bytes(obj))


def sources():
    _, _, r9_cards, old_inputs, gold = half.archive()
    old_review = HALF/"eval/review"
    assert sha(old_review/"scored.jsonl") == HALF_SCORED and sha(old_review/"summary.json") == HALF_SUMMARY
    scores = lines(old_review/"scored.jsonl")
    assert len(scores) == 960
    assert sha(old_review/"packet.jsonl") == read(old_review/"packet.json")["packet_sha256"]
    cards = dict(r9_cards)
    for card in lines(old_review/"packet.jsonl"):
        assert cards.setdefault(card["key"],card) == card
    by_key = {}
    for row in scores:
        content = {k:v for k,v in row.items() if k!="arm"}
        assert by_key.setdefault(row["key"],content) == content
    assert cards.keys() == by_key.keys()
    spec = HERE/"spec"
    receipt = read(spec/"receipt.json")
    assert sha(spec/"inputs.jsonl") == receipt["inputs"]
    assert sha(spec/"gold.jsonl") == receipt["gold"] == sha(half.OLD/"spec/gold.jsonl")
    inputs = lines(spec/"inputs.jsonl")
    assert len(inputs) == len({r["id"] for r in inputs}) == 192
    for old,new,g in zip(old_inputs,inputs,gold,strict=True):
        assert old["id"] == g["id"] and old["prompt"].endswith("\n"+TERMINAL)
        assert old["prompt"].count(CUE)==0 and new==dict(old,prompt=old["prompt"][:-len(TERMINAL)]+CUE+"\n"+TERMINAL)
    return scores,by_key,cards,inputs,gold


def predictions(arm, inputs):
    folder = HERE/"runs"/arm
    journal = HERE/"fresh/runs"/("cue-"+arm)
    manifest,done,receipt = (read(journal/"manifest.json"),read(journal/"done.json"),read(folder/"receipt.json"))
    assert done["manifest_sha256"] == receipt["journal_manifest_sha256"] == sha(journal/"manifest.json")
    assert done["output_sha256"] == sha(journal/"predictions.jsonl")
    preds = lines(folder/"predictions.jsonl")
    assert sha(folder/"predictions.jsonl") == receipt["sha256"]
    assert preds == lines(journal/"predictions.jsonl") and not (journal/"submitted.json").exists()
    assert receipt["fresh"] == done["n"] == len(preds) == 192 and receipt["arm"] == arm
    assert manifest["inputs_sha256"] == receipt["inputs_sha256"] == sha(HERE/"spec/inputs.jsonl")
    if arm=="base":
        prior = half.OLD/"fresh/runs/r9-base"
    else:
        prior = HALF/"eval/fresh/runs"/("half-"+arm)
    archive = read(prior/"manifest.json")
    for field in ("weights","config_sha256","engine_sha256","runner_sha256","batch","request_seed","max_output","eval_budget"):
        assert manifest[field]==archive[field], (arm,field)
    assert manifest["request_seed"]==3407 and manifest["max_output"]==2048
    for row,item in zip(preds,inputs,strict=True):
        assert all(row[k]==item[k] for k in ("id","prompt","image_sha256"))
        assert row["seed"]==3407 and row["max_output"]==2048 and row["finish_reason"] in ("stop","length")
    return preds


def packet():
    _,by_key,old_cards,inputs,gold=sources()
    r9_keys={r["key"] for r in lines(half.OLD/"review/packet.jsonl")}
    new,cached,proofs={},{},{}
    for arm in ARMS:
        preds=predictions(arm,inputs)
        proofs[arm]=dict(n=len(preds),sha256=sha(HERE/"runs"/arm/"predictions.jsonl"))
        for out,inp,item in zip(preds,inputs,gold,strict=True):
            key=judge.review_key(out["id"],out["raw"],out["finish_reason"])
            card=dict(key=key,id=item["id"],dataset=item["dataset"],group=item["group"],kind=item["kind"],
                      question=item["question"],prompt=inp["prompt"],image=inp["image"],image_sha256=inp["image_sha256"],
                      options=item["options"],gold=item["gold"],reference=item["reference"],raw=out["raw"],finish=out["finish_reason"])
            if key in by_key and card["finish"]=="stop":
                # Only the solver instruction changed. The content rubric, question,
                # gold, image identity, exact response and finish must still agree.
                previous=old_cards[key]
                assert {**previous,"prompt":card["prompt"]}==card
                origin="r9" if key in r9_keys else "half"
                value=dict(id=item["id"],source=origin,image_sha256=inp["image_sha256"],raw_sha256=hashlib.sha256(out["raw"].encode()).hexdigest())
                assert cached.setdefault(key,value)==value
            else:
                assert new.setdefault(key,card)==card
    immutable("cached.json",dict(r9_scored_sha256=half.SCORED,half_scored_sha256=HALF_SCORED,keys=cached))
    immutable("packet.jsonl",[new[k] for k in sorted(new)],rows=True)
    meta=dict(n_responses=576,unique=len(new),unique_including_cache=len(new)+len(cached),cached_keys=len(cached),per_arm=proofs,
              cached_sha256=sha(REVIEW/"cached.json"),packet_sha256=sha(REVIEW/"packet.jsonl"),inputs_sha256=sha(HERE/"spec/inputs.jsonl"),
              gold_sha256=sha(HERE/"spec/gold.jsonl"))
    immutable("packet.json",meta)
    print(meta,flush=True)


def run_judge():
    # The sealed helper establishes the identical rubric, model, decode settings
    # and reviewer context, and handles an entirely cached packet.
    from unittest.mock import patch
    with patch.object(half,"REVIEW",REVIEW),patch.object(half,"immutable",immutable):
        half.run_judge()


def adjudications(final):
    proof,seal = read(REVIEW/"packet.json"),read(REVIEW/"judge_complete.json")
    assert sha(REVIEW/"cached.json")==proof["cached_sha256"]
    assert sha(HERE/"spec/inputs.jsonl")==proof["inputs_sha256"] and sha(HERE/"spec/gold.jsonl")==proof["gold_sha256"]
    for file,field in (("packet.jsonl","packet_sha256"),("judgments.jsonl","judgments_sha256"),("judge_identity.json","identity_sha256")):
        assert sha(REVIEW/file)==seal[field]
    cards={r["key"]:r for r in lines(REVIEW/"packet.jsonl")}
    grades={r["key"]:r for r in lines(REVIEW/"judgments.jsonl")}
    assert cards.keys()==grades.keys() and len(cards)==proof["unique"]==seal["n"]
    assert all(x["identity"]==seal["identity_sha256"] for x in grades.values())
    patches={}
    if final:
        required={r["key"] for r in lines(REVIEW/"audit_packet.jsonl")}
        notes=lines(REVIEW/"manual.jsonl");patches={n["key"]:n for n in notes}
        assert required==patches.keys() and len(required)==len(notes)
        for k,n in patches.items():
            assert n["reason"].strip() and set(n["fields"])<=set((*rubric.FIELDS,"choice"))
            for f,v in n["fields"].items():
                assert v in ["?",*"ABCDEFGH"] if f=="choice" else v is None or type(v) is bool
            grades[k].update(n["fields"])
    return cards,grades,patches


def records(final=False):
    old,by_key,_,inputs,gold=sources()
    prior = {v:k for k,v in ORIGINAL.items()}
    rows=[dict(r,arm=prior[r["arm"]]+"_before") for r in old if r["arm"] in prior]
    cards,grades,patches=adjudications(final)
    cached=set(read(REVIEW/"cached.json")["keys"])
    assert not cached & cards.keys()
    bare={k for k,c in cards.items() if rubric.only_options(c["raw"],c["options"])}
    parser=half.parser()
    used=set()
    proof=read(REVIEW/"packet.json")
    for arm in ARMS:
        assert sha(HERE/"runs"/arm/"predictions.jsonl")==proof["per_arm"][arm]["sha256"]
        for p,g in zip(predictions(arm,inputs),gold,strict=True):
            key=rubric.key(p);used.add(key)
            if key in cached:
                assert p["finish_reason"]=="stop"
                rows.append(dict(by_key[key],arm=arm+"_cue"))
                continue
            z=dict(grades[key])
            if key in bare:
                z.update(e=False,r=False,intermediate=False,winner=False,a=False,first=None,corrected=None)
            extracted=rubric.as_choice(p["raw"],g["options"],parser)
            choice=extracted or z["choice"]
            answer=p["finish_reason"]=="stop" and choice==g["gold"]
            application=rubric.every([z["a"],z["intermediate"],z["winner"]])
            joint=rubric.every([answer,z["e"],z["r"],application,z["consistent"]])
            first_error="no_grounded_answer" if key in bare else patches.get(key,{}).get("first_error")
            if g["dataset"]=="paired" and final and key not in bare:
                assert first_error in rubric.ERRORS and (first_error=="none")==bool(joint is True),(g["id"],first_error)
            if first_error=="loser_rounding":
                assert answer and all(z[k] is True for k in ("e","r","winner","consistent")) and z["intermediate"] is False
            rows.append(dict(id=p["id"],key=key,arm=arm+"_cue",group=g["group"],kind=g["kind"],dataset=g["dataset"],
                             sentinel=g["sentinel"],choice=choice,fallback=extracted is None,answer=answer,joint=joint,bare=key in bare,
                             format=parser.format_ok(p["raw"],p["finish_reason"]),finish=p["finish_reason"],
                             **{f:application if f=="a" else z[f] for f in rubric.FIELDS},a_judgment=z["a"],
                             er=rubric.every([z["e"],z["r"]]),first_error=first_error,rounding_joint=True if first_error=="loser_rounding" else joint))
    assert len(rows)==len({(r["arm"],r["id"]) for r in rows})==1152 and used==(cached|cards.keys())
    return rows,cards,grades,bare


def audit():
    rows,cards,grades,bare=records()
    by_id=defaultdict(dict)
    for r in rows:by_id[r["id"]][r["arm"]]=r
    samples=sorted([i for i,b in by_id.items() if b["base_before"]["dataset"]=="general"],
                   key=lambda i:hashlib.sha256(("r9-general-sample/"+i).encode()).hexdigest())[:8]
    selected=set()
    for i,b in by_id.items():
        assert len(b)==6
        cohort=any(b[a+"_before"]["bare"] and b[a+"_before"]["answer"] for a in ARMS)
        rs=list(b.values())
        compare=len({(r["answer"],r["joint"]) for r in rs})>1
        needed=cohort or rs[0]["dataset"]=="paired" or rs[0]["sentinel"] or i in samples or compare or any(
            r["fallback"] or r["corrected"] is True for r in rs)
        if needed:
            selected.update(r["key"] for r in rs if r["key"] in cards and r["key"] not in bare)
    immutable("audit_packet.jsonl",[dict(cards[k],judge={f:grades[k][f] for f in (*rubric.FIELDS,"choice")})
                                    for k in sorted(selected)],rows=True)
    immutable("bare.jsonl",[dict(key=k,id=cards[k]["id"],raw=cards[k]["raw"]) for k in sorted(bare)],rows=True)
    immutable("audit_receipt.json",dict(n=len(selected),novel=len(cards),sha256=sha(REVIEW/"audit_packet.jsonl")))
    shards.REVIEW,shards.ROOT=REVIEW,ROOT
    shards.split()


def main():
    cli=argparse.ArgumentParser();cli.add_argument("mode",choices=("packet","judge","audit","merge","score"))
    mode=cli.parse_args().mode
    if mode=="merge":
        shards.REVIEW=REVIEW
        shards.merge()
    elif mode=="score":
        from cue_report import summarize
        summarize(records(final=True)[0])
    else:
        {"packet":packet,"judge":run_judge,"audit":audit}[mode]()


if __name__=="__main__":
    main()
