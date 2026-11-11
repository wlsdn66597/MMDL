"""Independently recompute the cue pilot gates from the sealed prompts, outputs and grades."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R9 = ROOT.parent / "era_a_only_round"
HALF = ROOT.parent / "era_half_strength"
ARMS = ("base", "ordinary", "enhanced")
CUE = "State the relevant evidence from the image, the rule you use, and how it leads to your choice. Keep each part brief."
FINAL_LINE = 'On the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'
R9_INPUTS = "ff9f3846e0820e3939dd07488dcdb5c3bc2d800b8d96a81fbea82afe3f8934a5"
R9_GOLD = "5396d9c80697c0577f6657e249f390d12e321575beae4964168e1da8d3829525"
HALF_SCORED = "85db0dc94b7ac345988a6e69c82230f288a9df21bd1a8b09dc12d8be816a44ca"
HALF_VERIFICATION = "24843962db095a699fede223e423471f7c221b2f98077b1f8a93ca5e06197de2"
R9_JUDGE = "89ab822705ecb8f6468d94efbf9183d564a011be2b80ff0438b3bf8641cd05be"
THRESHOLDS = dict(general_joint=92, general_answer=95, paired_base_kept=90, pair_joint=32)


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def by_id(items, key="id"):
    index = {item[key]:item for item in items}
    assert len(index)==len(items), f"Duplicate {key}"
    return index


def check_inputs():
    spec = ROOT / "eval/spec"
    original = R9 / "eval/spec"
    proof = read(spec/"receipt.json")
    new_file, old_file = spec/"inputs.jsonl", original/"inputs.jsonl"
    assert digest(old_file)==R9_INPUTS and digest(original/"gold.jsonl")==R9_GOLD
    assert proof["inputs"]==digest(new_file) and proof["gold"]==digest(spec/"gold.jsonl")==R9_GOLD
    old, new, gold = [lines(f) for f in (old_file,new_file,spec/"gold.jsonl")]
    assert len(old)==len(new)==len(gold)==len({r["id"] for r in new})==192
    paths = {}
    for a,b,g in zip(old,new,gold,strict=True):
        assert set(a)==set(b)=={"id","dataset","image","image_sha256","prompt"}
        assert b["id"]==g["id"] and b["dataset"]==g["dataset"]
        assert b["prompt"].splitlines()[-2:]==[CUE,FINAL_LINE]
        assert b["prompt"].count(CUE)==1 and CUE not in a["prompt"]
        assert b|{"prompt":b["prompt"].replace(CUE+"\n","",1)}==a
        path=(R9/b["image"]).resolve()
        assert path.is_relative_to(R9.resolve()) and path.is_file()
        if path not in paths:
            paths[path]=digest(path)
        assert b["image_sha256"]==paths[path]
    assert len(paths)==144 and sum(r["dataset"]=="paired" for r in new)==96
    return new, by_id(gold), proof


def response_key(p):
    value=json.dumps([p["id"],p["raw"],p["finish_reason"]], ensure_ascii=False, separators=(",",":"))
    return hashlib.sha256(value.encode()).hexdigest()


def check_journal(arm, scores, spec, spec_digest):
    journal=ROOT/"eval/fresh/runs"/("cue-"+arm)
    export=ROOT/"eval/runs"/arm
    manifest=read(journal/"manifest.json")
    done=read(journal/"done.json")
    receipt=read(export/"receipt.json")
    prev=R9 if arm=="base" else HALF
    name=("r9-" if arm=="base" else "half-")+arm
    baseline_certificate=read(HALF/"verification.json")["solver_journals"][arm if arm=="base" else arm+"50"]
    archived=prev/"eval/fresh/runs"/name
    assert digest(archived/"manifest.json")==baseline_certificate["manifest_sha256"]
    assert digest(archived/"predictions.jsonl")==baseline_certificate["predictions_sha256"]
    before=read(archived/"manifest.json")
    assert digest(journal/"predictions.jsonl")==done["output_sha256"]==digest(export/"predictions.jsonl")==receipt["sha256"]
    assert digest(journal/"manifest.json")==done["manifest_sha256"]==receipt["journal_manifest_sha256"]
    assert manifest["inputs_sha256"]==receipt["inputs_sha256"]==spec_digest
    assert not (journal/"submitted.json").exists() and done["n"]==receipt["n"]==receipt["fresh"]==192
    for k in ("weights","batch","request_seed","max_output","config_sha256","engine_sha256","runner_sha256","eval_budget"):
        assert manifest[k]==before[k],(arm,k)
    assert receipt["solver"]==before["runner_sha256"] and receipt["arm"]==manifest["arm"]==arm
    if arm!="base":
        original=HALF/"scale_receipts"/(arm+".json")
        if (ROOT/"scale_receipts"/(arm+".json")).is_file():
            assert digest(ROOT/"scale_receipts"/(arm+".json"))==digest(original)
        assert receipt["scale_receipt_sha256"]==digest(original)
        scale=read(original)
        assert scale["weights"]==manifest["weights"] and scale["factor"]==0.5
        assert scale["pre_bf16_exact_half_targets"]==scale["bf16_export_checked_targets"]==144
        assert scale["unchanged_non_target_tensors"]==570
    else:
        assert receipt["scale_receipt_sha256"] is None
    preds=lines(journal/"predictions.jsonl")
    assert manifest["request_seed"]==3407 and manifest["max_output"]==2048
    for p,x in zip(preds,spec,strict=True):
        rid=x["id"]
        assert all(p[k]==x[k] for k in ("id","prompt","image_sha256"))
        assert p["seed"]==3407 and p["max_output"]==2048 and p["finish_reason"] in {"stop","length"}
        assert scores[rid]["key"]==response_key(p) and scores[rid]["finish"]==p["finish_reason"]
    return {"completed_unique":len(by_id(preds)), "predictions_sha256":receipt["sha256"],
            "manifest_sha256":done["manifest_sha256"], "scale_receipt_sha256":receipt["scale_receipt_sha256"]}


def and3(values):
    values=tuple(values)
    assert values and set(values)<={True,False,None}
    return False if False in values else None if None in values else True


def check_scores(records, gold):
    out=defaultdict(dict)
    for row in records:
        rid,arm=row["id"],row["arm"]
        assert rid not in out[arm] and rid in gold
        g=gold[rid]
        assert all(row[k]==g[k] for k in ("dataset","group","kind","sentinel"))
        assert row["answer"] is (row["choice"]==g["gold"] and row["finish"]=="stop")
        assert row["a"] is and3([row["intermediate"],row["winner"],row["a_judgment"]])
        assert row["er"] is and3([row["e"],row["r"]])
        assert row["joint"] is and3([row["answer"],row["er"],row["a"],row["consistent"]])
        if row["bare"]:
            assert row["e"] is False and row["r"] is False and row["a"] is False
        out[arm][rid]=row
    required={arm+"_"+condition for arm in ARMS for condition in ("before","cue")}
    assert set(out)==required and all(set(rs)==set(gold) for rs in out.values())
    return out


def band(values):
    values=tuple(values)
    return [sum(v is True for v in values),sum(v is not False for v in values)]


def belongs(row, subset):
    if subset=="all":
        return True
    if subset=="sentinel":
        return row["sentinel"]
    if subset=="ChartQA":
        return row["dataset"]=="general" and row["kind"]=="ChartQA"
    if subset=="ScienceQA":
        return row["dataset"]=="general" and not row["sentinel"] and row["kind"]!="ChartQA"
    return row["dataset"]==subset


def derive(rows, summary):
    basecorrect={rid for rid,r in rows["base_before"].items() if r["dataset"]=="paired" and r["answer"]}
    assert len(basecorrect)==90
    facts={}
    for arm in ARMS:
        old,new=rows[arm+"_before"],rows[arm+"_cue"]
        sections={}
        for subset in ("all","general","paired","sentinel","ChartQA","ScienceQA"):
            ids={i for i,r in old.items() if belongs(r,subset)}
            eligible={i for i in ids if old[i]["bare"] and old[i]["answer"]}
            recover={i for i in eligible if new[i]["answer"] and new[i]["joint"] is True}
            kept={i for i in eligible if new[i]["answer"]}
            lost={i for i in ids if old[i]["answer"] and not new[i]["answer"]}
            cohort=summary["cohorts"][arm][subset]
            for field,expected in (("eligible",eligible),("linked",recover),("answer_kept",kept),("answer_lost",eligible-kept)):
                assert set(cohort[field])==expected,(arm,subset,field)
            if subset in ("general","paired","sentinel"):
                assert set(summary["changes"][arm][subset]["answer"]["lost"])==lost
            sections[subset]=dict(eligible_ids=sorted(eligible),linked_ids=sorted(recover),
                                  still_correct_ids=sorted(kept),newly_wrong_ids=sorted(lost))
        group_rows=defaultdict(list)
        for r in new.values():
            if r["dataset"]=="paired":
                group_rows[r["group"]].append(r)
        assert len(group_rows)==48 and all(len(rs)==2 for rs in group_rows.values())
        group_joint={g:and3(r["joint"] for r in items) for g,items in group_rows.items()}
        general=[r for r in new.values() if r["dataset"]=="general"]
        retained=sorted(i for i in basecorrect if new[i]["answer"])
        assert retained==sorted(summary["base_retained"][arm+"_cue"]["kept"])
        assert len(general)==96
        counts=dict(general_joint=sum(r["joint"] is True for r in general),
                    general_answer=sum(r["answer"] for r in general),paired_base_kept=len(retained),
                    pair_joint=sum(v is True for v in group_joint.values()))
        m=summary["metrics"][arm+"_cue"]
        assert m["groups"]["joint"]==band(group_joint.values())
        for g, value in group_joint.items():
            assert summary["groups"][arm+"_cue"][g]["joint"] is value
        for k in ("answer","joint","bare","e","r","a","format"):
            assert m["general"][k]==band(r[k] for r in general)
        facts[arm]=dict(counts=counts, old_bare_cohorts=sections, base_correct_paired_kept_ids=retained,
                        connected_group_ids=sorted(g for g,v in group_joint.items() if v is True))
    counts=facts["ordinary"]["counts"]
    tests={k:counts[k]>=v for k,v in THRESHOLDS.items()}
    assert summary["primary"]=="ordinary_cue" and summary["counts"]==counts and summary["thresholds"]==THRESHOLDS
    assert tests==summary["tests"] and all(tests.values())==summary["advance"]
    return dict(counts=counts,thresholds=THRESHOLDS,tests=tests,advance=all(tests.values()),arms=facts)


def main():
    review=ROOT/"eval/review"
    spec,gold,spec_proof=check_inputs()
    summary=read(review/"summary.json")
    assert digest(HALF/"verification.json")==HALF_VERIFICATION
    assert digest(HALF/"eval/review/scored.jsonl")==HALF_SCORED
    assert all(digest(review/k)==h for k,h in summary["proof"].items())
    rows=check_scores(lines(review/"scored.jsonl"),gold)
    old=lines(HALF/"eval/review/scored.jsonl")
    armlabel={"base":"base","ordinary50":"ordinary","enhanced50":"enhanced"}
    for r in old:
        if r["arm"] in armlabel:
            target=armlabel[r["arm"]]+"_before"
            assert rows[target][r["id"]]==r|{"arm":target}
    packet=read(review/"packet.json")
    complete=read(review/"judge_complete.json")
    assert packet["n_responses"]==576 and complete["packet_sha256"]==digest(review/"packet.jsonl")
    cache_file=review/"cached.json"
    assert digest(cache_file)==packet["cached_sha256"]
    cache=set(read(cache_file)["keys"])
    old_keys={r["key"]:{k:v for k,v in r.items() if k!="arm"} for r in old}
    fresh_keys=by_id(lines(review/"packet.jsonl"),"key")
    assert len(cache)==packet["cached_keys"] and len(fresh_keys)==packet["unique"]==complete["n"]
    assert cache<=old_keys.keys() and not cache&fresh_keys.keys()
    current={r["key"] for arm in ARMS for r in rows[arm+"_cue"].values()}
    assert current==cache|fresh_keys.keys() and len(current)==packet["unique_including_cache"]
    for arm in ARMS:
        for row in rows[arm+"_cue"].values():
            if row["key"] in cache:
                assert row["finish"]=="stop" and old_keys[row["key"]]|{"arm":arm+"_cue"}==row
    for name,k in (("judgments.jsonl","judgments_sha256"),("judge_identity.json","identity_sha256")):
        assert complete[k]==digest(review/name)
    if complete["n"]:
        identity=read(review/"judge_identity.json")
        assert digest(R9/"eval/review/judge_identity.json")==R9_JUDGE
        prior=read(R9/"eval/review/judge_identity.json")
        for k in ("judge_files","rubric_sha256","program","config"):
            assert identity[k]==prior[k],k
    journals={arm:check_journal(arm,rows[arm+"_cue"],spec,spec_proof["inputs"]) for arm in ARMS}
    result=derive(rows,summary)
    protocol_hash=digest(ROOT/"protocol.md")
    assert read(ROOT/"started.json")["protocol"]==protocol_hash
    result.update(protocol_sha256=protocol_hash,inputs_sha256=spec_proof["inputs"],
                  scored_sha256=digest(review/"scored.jsonl"),summary_sha256=digest(review/"summary.json"),
                  old_half_scored_sha256=HALF_SCORED,fresh_calls=sum(x["completed_unique"] for x in journals.values()),
                  journals=journals)
    assert result["fresh_calls"]==576
    output=ROOT/"verification.json"
    if output.exists():
        assert read(output)==result
    else:
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:result[k] for k in ("counts","thresholds","tests","advance","fresh_calls")},ensure_ascii=False))


if __name__=="__main__":
    main()
