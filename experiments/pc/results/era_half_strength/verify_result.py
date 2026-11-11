"""Reconstruct the prespecified 50% diagnostic decision from scored rows and solver journals."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R9 = ROOT.parent/"era_a_only_round"
ARMS = ("base","ordinary","enhanced","ordinary50","enhanced50")
PROTOCOL = "046ab484422f667588a38c9de98d1e19f1129a1a1e3e93cf1d46ce1de207cc36"
SCORED = "4a7cb52f661642a66b9b527c69901ad20f04350277da19edf77ed755fb396b23"
SUMMARY = "683bdc873df9b67a106d4e7d0dc7fe6ca71454bd54a6fe937ad7482a82d9096f"


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def all3(values):
    flags=set(values)
    assert flags <= {True,False,None} and flags
    return False if False in flags else None if None in flags else True


def band(values):
    values=list(values)
    return [values.count(True),len(values)-values.count(False)]


def transitions(a,b):
    assert a.keys()==b.keys()
    tags={(True,True):"kept",(True,False):"lost",(False,True):"recovered",(False,False):"both_wrong"}
    result={name:[] for name in (*tags.values(),"unknown")}
    for key in sorted(a):
        result[tags.get((a[key],b[key]),"unknown")].append(key)
    return result


def verify_journal(arm,rows,spec):
    source=arm.removesuffix("50")
    stage=ROOT if arm.endswith("50") else R9
    journal=stage/"eval/fresh/runs"/(("half-" if stage==ROOT else "r9-")+source)
    export=stage/"eval/runs"/source
    manifest=read(journal/"manifest.json")
    done=read(journal/"done.json")
    receipt=read(export/"receipt.json")
    h=digest(journal/"predictions.jsonl")
    assert h==done["output_sha256"]==receipt["sha256"]==digest(export/"predictions.jsonl")
    assert done["manifest_sha256"]==digest(journal/"manifest.json")
    assert done["n"]==receipt["fresh"]==192 and not (journal/"submitted.json").exists()
    prior=read(R9/"eval/fresh/runs/r9-base/manifest.json")
    for field in ("config_sha256","engine_sha256","runner_sha256","inputs_sha256","batch","request_seed","max_output","eval_budget"):
        assert manifest[field]==prior[field],(arm,field)
    assert manifest["inputs_sha256"]==digest(R9/"eval/spec/inputs.jsonl")
    weight_receipt=None
    if stage==ROOT:
        assert receipt["journal_manifest_sha256"]==done["manifest_sha256"]
        weight_receipt=ROOT/"scale_receipts"/(source+".json")
        if weight_receipt.exists():
            weights=read(weight_receipt)
            assert digest(weight_receipt)==receipt["scale_receipt_sha256"] and weights["factor"]==0.5
            assert weights["arm"]==source and weights["weights"]==manifest["weights"]
            assert weights["pre_bf16_exact_half_targets"]==weights["bf16_export_checked_targets"]>0
            assert weights["unchanged_non_target_tensors"]+weights["bf16_export_checked_targets"]==weights["model_tensor_count"]
    predictions=lines(journal/"predictions.jsonl")
    assert len(predictions)==len(spec)==192
    for pred,item in zip(predictions,spec,strict=True):
        assert all(pred[k]==item[k] for k in ("id","prompt","image_sha256"))
        assert pred["seed"]==3407 and pred["max_output"]==2048 and pred["finish_reason"] in ("stop","length")
        scored=rows[arm][pred["id"]]
        key=sha256(json.dumps([pred["id"],pred["raw"],pred["finish_reason"]],ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        assert scored["key"]==key and scored["finish"]==pred["finish_reason"]
    return dict(predictions_sha256=h,manifest_sha256=done["manifest_sha256"],
                scale_receipt_checked=weight_receipt.is_file() if weight_receipt else None)


def summarize(records,summary):
    rows={a:{} for a in ARMS}
    for r in records:
        arm, rid=r["arm"],r["id"]
        assert rid not in rows[arm] and r["answer"] in (True,False)
        assert r["a"] is all3([r["a_judgment"],r["winner"],r["intermediate"]])
        assert r["er"] is all3([r["e"],r["r"]]) and r["joint"] is all3([r["er"],r["a"],r["consistent"],r["answer"]])
        rows[arm][rid]=r
    assert all(len(v)==192 and v.keys()==rows["base"].keys() for v in rows.values())
    groups={}
    for arm,rs in rows.items():
        buckets=defaultdict(list)
        for r in rs.values():
            if r["dataset"]=="paired": buckets[r["group"]].append(r)
        assert len(buckets)==48 and all(len(v)==2 for v in buckets.values())
        groups[arm]={g:{k:all3(row[k] for row in members) for k in ("answer","er","r","joint","rounding_joint")}
                     for g,members in buckets.items()}
        for g,flags in groups[arm].items():
            assert all(summary["groups"][arm][g][k] is correct for k,correct in flags.items())
    def subset(arm,ds,field):
        return {rid:r[field] for rid,r in rows[arm].items() if (r["sentinel"] if ds=="sentinel" else r["dataset"]==ds)}
    for name,cells in summary["transitions"].items():
        a,b=name.split("-")
        for ds,stats in cells.items():
            for field,out in stats.items():
                if ds=="groups":
                    maps=[{g:members[field] for g,members in groups[x].items()} for x in (a,b)]
                else:
                    maps=[subset(x,ds,field) for x in (a,b)]
                assert transitions(*maps)==out,(name,ds,field)
    for tag,a,b in (("original_fixed29","ordinary","enhanced"),("half_common_er","ordinary50","enhanced50")):
        section=summary["common"][tag]
        confirmed=sorted(g for g in groups[a] if all(groups[x][g]["er"] is True for x in (a,b)))
        assert section["ids"]==confirmed and section["n"]==len(confirmed)
        if tag=="original_fixed29":
            assert len(confirmed)==29 and confirmed==sorted(read(R9/"eval/review/summary.json")["common_er"]["ids"])
        for x in ARMS:
            for field in ("er","joint"):
                assert section[field][x]==band(groups[x][g][field] for g in confirmed)
        if tag=="half_common_er":
            possible=sorted(g for g in groups[a] if g not in confirmed and all(groups[x][g]["er"] is not False for x in (a,b)))
            assert section["possibly_additional_groups"]==possible
        assert section["half_comparison"]==transitions(
            *[{g:groups[x][g]["joint"] for g in confirmed} for x in ("ordinary50","enhanced50")])
    a="enhanced50"
    keep=transitions(subset("base","paired","answer"),subset(a,"paired","answer"))["kept"]
    kept_sentinel=transitions(subset("base","sentinel","answer"),subset(a,"sentinel","answer"))["kept"]
    counts=dict(paired_kept=len(keep),general_answer=sum(subset(a,"general","answer").values()),
                connection=sum(v["joint"] is True for v in groups[a].values()),
                general_connection=sum(x is True for x in subset(a,"general","joint").values()),sentinel=len(kept_sentinel))
    thresholds=dict(paired_kept=75,general_answer=95,connection=22,general_connection=92,sentinel=16)
    tests={k:count>=thresholds[k] for k,count in counts.items()}
    assert tests==summary["tests"] and all(tests.values())==summary["advance"]
    return rows,dict(counts=counts,thresholds=thresholds,tests=tests,advance=all(tests.values()))


def main():
    review=ROOT/"eval/review"
    summary=read(review/"summary.json")
    assert digest(ROOT/"protocol.md")==PROTOCOL
    if (ROOT/"started.json").exists():
        assert read(ROOT/"started.json")["protocol"]==PROTOCOL
    assert digest(R9/"eval/review/scored.jsonl")==SCORED==summary["r9_scored_sha256"]
    assert digest(R9/"eval/review/summary.json")==SUMMARY==summary["r9_summary_sha256"]
    assert all(digest(review/k)==h for k,h in summary["proof"].items())
    data=lines(review/"scored.jsonl")
    rows,result=summarize(data,summary)
    for r in lines(R9/"eval/review/scored.jsonl"):
        assert rows[r["arm"]][r["id"]]==r
    spec=lines(R9/"eval/spec/inputs.jsonl")
    assert len(spec)==len({x["id"] for x in spec})==192 and summary["fresh_calls"]==384
    receipt=dict(**result,protocol_sha256=PROTOCOL,scored_sha256=digest(review/"scored.jsonl"),
                 summary_sha256=digest(review/"summary.json"),fresh_calls=384,
                 solver_journals={a:verify_journal(a,rows,spec) for a in ARMS})
    path=ROOT/"verification.json"
    if path.exists():assert read(path)==receipt
    else:path.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=="__main__":
    main()
