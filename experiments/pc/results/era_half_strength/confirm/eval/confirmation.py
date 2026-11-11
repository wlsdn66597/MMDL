"""Blind four-arm review of independent inputs when the diagnostic gate passes."""
import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HALF = ROOT.parent
REVIEW = HERE / "review"
sys.path.insert(0,str(HALF/"eval"))
import diagnostic as diag

rubric,judge,shards = diag.rubric,diag.judge,diag.shards
read,lines,sha = diag.read,diag.lines,diag.sha
ARMS=("base","ordinary","enhanced","enhanced100")


def immutable(name,value,*,rows=False):
    judge.freeze(REVIEW/name,judge.json_lines(value) if rows else judge.json_bytes(value))


def diagnostic_gate():
    source=HALF/"eval/review/summary.json"
    record=read(source)
    assert record["advance"] is True and len(record["tests"])==5 and all(record["tests"].values())
    for name,digest in record["proof"].items():
        assert digest==sha(source.parent/name)
    return sha(source)


def sources():
    spec=HERE/"spec"
    receipt=read(spec/"receipt.json")
    assert receipt["inputs"]==sha(spec/"inputs.jsonl") and receipt["gold"]==sha(spec/"gold.jsonl")
    inputs,gold=lines(spec/"inputs.jsonl"),lines(spec/"gold.jsonl")
    assert len(inputs)==len(gold)==len({x["id"] for x in inputs})==192
    assert Counter(r["dataset"] for r in inputs)=={"paired":96,"general":96} and sum(r["sentinel"] for r in gold)==16
    assert [r["id"] for r in inputs]==[r["id"] for r in gold]
    old=lines(diag.OLD/"spec/inputs.jsonl")
    assert not {x["id"] for x in inputs}&{x["id"] for x in old}
    assert not {x["image_sha256"] for x in inputs}&{x["image_sha256"] for x in old}
    return receipt,inputs,gold


def predictions(arm,inputs,receipt):
    assert arm in ARMS
    journal=HERE/"fresh/runs"/("half-"+arm)
    run=HERE/"runs"/arm
    manifest=read(journal/"manifest.json")
    done=read(journal/"done.json")
    proof=read(run/"receipt.json")
    pred=run/"predictions.jsonl"
    assert sha(pred)==sha(journal/"predictions.jsonl")==done["output_sha256"]==proof["sha256"]
    assert sha(journal/"manifest.json")==done["manifest_sha256"]==proof["journal_manifest_sha256"]
    assert not (journal/"submitted.json").exists()
    assert proof["arm"]==manifest["arm"]==arm and proof["stage"]=="confirm" and proof["inputs_sha256"]==receipt["inputs"]
    if arm in ("ordinary","enhanced"):
        assert proof["scale_receipt_sha256"]==sha(HALF/"scale_receipts"/(arm+".json"))
        assert read(HALF/"scale_receipts"/(arm+".json"))["weights"]==manifest["weights"]
    else:
        old_arm="base" if arm=="base" else "enhanced"
        prior=read(diag.OLD/"fresh/runs"/("r9-"+old_arm)/"manifest.json")
        assert prior["weights"]==manifest["weights"] and proof["scale_receipt_sha256"] is None
    baseline=read(diag.OLD/"fresh/runs/r9-base/manifest.json")
    for field in ("config_sha256","engine_sha256","runner_sha256","batch","request_seed","max_output","eval_budget"):
        assert manifest[field]==baseline[field]
    data=lines(pred)
    assert len(data)==len(inputs)==len({r["id"] for r in data})==proof["fresh"]==done["n"]==192
    for x,r in zip(inputs,data):
        assert all(x[f]==r[f] for f in ("id","prompt","image_sha256"))
        assert r["seed"]==3407 and r["max_output"]==2048
    return data


def packet():
    gate=diagnostic_gate()
    receipt,inputs,gold=sources()
    cards,arms={},{}
    for arm in ARMS:
        data=predictions(arm,inputs,receipt)
        arms[arm]=dict(n=len(data),sha256=sha(HERE/"runs"/arm/"predictions.jsonl"))
        for p,x,g in zip(data,inputs,gold):
            k=judge.review_key(x["id"],p["raw"],p["finish_reason"])
            card=dict(key=k,id=x["id"],dataset=x["dataset"],group=g["group"],kind=g["kind"],question=g["question"],
                      prompt=x["prompt"],image=x["image"],image_sha256=x["image_sha256"],options=g["options"],gold=g["gold"],
                      reference=g["reference"],raw=p["raw"],finish=p["finish_reason"])
            assert cards.setdefault(k,card)==card
    immutable("packet.jsonl",[cards[k] for k in sorted(cards)],rows=True)
    immutable("packet.json",dict(n_responses=768,unique=len(cards),per_arm=arms,packet_sha256=sha(REVIEW/"packet.jsonl"),
              inputs_sha256=receipt["inputs"],gold_sha256=receipt["gold"],diagnostic_gate_sha256=gate))


def run_judge():
    prior=read(diag.OLD/"review/judge_identity.json")
    assert hashlib.sha256(judge.RUBRIC.encode()).hexdigest()==prior["rubric_sha256"] and sha(Path(judge.__file__))==prior["program"]
    judge.REVIEW=REVIEW
    judge.grade()
    current=read(REVIEW/"judge_identity.json")
    assert current["judge_files"]==prior["judge_files"] and current["config"]==prior["config"]


def records(final=False):
    receipt,inputs,gold=sources()
    proof=read(REVIEW/"packet.json")
    seal=read(REVIEW/"judge_complete.json")
    assert proof["diagnostic_gate_sha256"]==diagnostic_gate() and proof["inputs_sha256"]==receipt["inputs"]
    assert proof["gold_sha256"]==receipt["gold"] and proof["packet_sha256"]==seal["packet_sha256"]
    for f,k in (("packet.jsonl","packet_sha256"),("judgments.jsonl","judgments_sha256"),("judge_identity.json","identity_sha256")):
        assert sha(REVIEW/f)==seal[k]
    cards={r["key"]:r for r in lines(REVIEW/"packet.jsonl")}
    grades={r["key"]:r for r in lines(REVIEW/"judgments.jsonl")}
    assert len(cards)==proof["unique"]==seal["n"] and cards.keys()==grades.keys()
    assert all(z["identity"]==seal["identity_sha256"] for z in grades.values())
    patches={}
    if final:
        required={r["key"] for r in lines(REVIEW/"audit_packet.jsonl")}
        notes=lines(REVIEW/"manual.jsonl")
        patches={r["key"]:r for r in notes}
        assert required==patches.keys() and len(required)==len(notes)
        for k,n in patches.items():
            assert n["reason"].strip() and set(n["fields"])<=set((*rubric.FIELDS,"choice"))
            for f,v in n["fields"].items():
                assert v in ["?",*"ABCDEFGH"] if f=="choice" else v is None or type(v) is bool
            grades[k].update(n["fields"])
    bare={k for k,r in cards.items() if rubric.only_options(r["raw"],r["options"])}
    for k in bare:
        grades[k].update(e=False,r=False,a=False,intermediate=False,winner=False,first=None,corrected=None)
    parser=diag.parser()
    rows=[]
    for arm in ARMS:
        data=predictions(arm,inputs,receipt)
        assert proof["per_arm"][arm]["sha256"]==sha(HERE/"runs"/arm/"predictions.jsonl")
        for p,g in zip(data,gold):
            k=rubric.key(p);z=grades[k]
            letter=rubric.as_choice(p["raw"],g["options"],parser)
            choice=letter or z["choice"]
            correct=p["finish_reason"]=="stop" and choice==g["gold"]
            application=rubric.every([z["a"],z["intermediate"],z["winner"]])
            joint=rubric.every([correct,z["e"],z["r"],application,z["consistent"]])
            error="no_grounded_answer" if k in bare else patches.get(k,{}).get("first_error")
            if final and g["dataset"]=="paired" and k not in bare:
                assert error in rubric.ERRORS and (error=="none")==(joint is True),(g["id"],error)
            if error=="loser_rounding":
                assert correct and all(z[f] is True for f in ("e","r","winner","consistent")) and z["intermediate"] is False
            rows.append(dict(id=p["id"],key=k,arm=arm,group=g["group"],kind=g["kind"],dataset=g["dataset"],sentinel=g["sentinel"],
                        choice=choice,fallback=letter is None,answer=correct,joint=joint,bare=k in bare,
                        format=parser.format_ok(p["raw"],p["finish_reason"]),finish=p["finish_reason"],
                        **{f:application if f=="a" else z[f] for f in rubric.FIELDS},a_judgment=z["a"],
                        er=rubric.every([z["e"],z["r"]]),first_error=error,rounding_joint=True if error=="loser_rounding" else joint))
    assert len(rows)==len({(r["arm"],r["id"]) for r in rows})==768
    return rows,cards,grades


def confirmation_tests(s,rows):
    by={arm:[r for r in rows if r["arm"]==arm] for arm in ARMS}
    answers={arm:{ds:{r["id"] for r in rs if r["dataset"]==ds and r["answer"]} for ds in ("paired","general")}
             for arm,rs in by.items()}
    sent={r["id"] for r in by["base"] if r["sentinel"]}
    assert len(sent)==16
    groups=s["groups"]
    def net(field,base,skip=None):
        high=rubric.bounds(v[field] for i,v in groups["enhanced"].items() if i!=skip)[0]
        low=rubric.bounds(v[field] for i,v in groups[base].items() if i!=skip)[1]
        return high-low
    ga=set(groups["ordinary"])
    gained={groups["enhanced"][g]["kind"] for g in ga if groups["enhanced"][g]["joint"] is True and groups["ordinary"][g]["joint"] is False}
    common=sorted(g for g in ga if groups["ordinary"][g]["er"] is True and groups["enhanced"][g]["er"] is True)
    possible=[g for g in ga if g not in common and groups["ordinary"][g]["er"] is not False and groups["enhanced"][g]["er"] is not False]
    delta=lambda g:int(groups["enhanced"][g]["joint"] is True)-int(groups["ordinary"][g]["joint"] is not False)
    common_delta=sum(delta(g) for g in common)+sum(min(0,delta(g)) for g in possible)
    assert s["common_er"]["ids"]==common and s["common_er"]["conservative_net"]==common_delta
    def keep(ds,arm):return len(answers["base"][ds]&answers[arm][ds])
    general=lambda arm:answers[arm]["general"]
    ordinary_tests=dict(rule_pairs_noninferior=net("r","ordinary")>=0,er_pairs_noninferior=net("er","ordinary")>=0,
                connected_pairs=net("joint","ordinary")>0,joint_leave_one=all(net("joint","ordinary",g)>0 for g in ga),
                recovered_families=len(gained)>=2,common_er_connected=common_delta>0,general_at_base=len(general("enhanced"))>=len(general("base")),
                general_baseline_retained=keep("general","enhanced")>=keep("general","ordinary"),
                paired_baseline_retained=keep("paired","enhanced")>=keep("paired","ordinary"),
                original_general_no_new_loss=not (general("base")&general("ordinary")-general("enhanced")),
                sentinel_no_new_loss=not ((general("ordinary")-general("enhanced"))&sent))
    assert ordinary_tests==s["tests"]
    strong=s["metrics"]["enhanced"]["general"]["joint"][0]
    full=s["metrics"]["enhanced100"]["general"]["joint"][1]
    against_full=dict(paired_kept_up=keep("paired","enhanced")>keep("paired","enhanced100"),
            general_answer_up=len(general("enhanced"))>len(general("enhanced100")),
            paired_joint_up=net("joint","enhanced100")>0,general_joint_preserved=strong>=full,
            sentinel_no_new_loss=not ((general("enhanced100")-general("enhanced"))&sent))
    return ordinary_tests,against_full


def score():
    def save_summary(path,summary):
        assert path==REVIEW/"summary.json"
        rows=lines(REVIEW/"scored.jsonl")
        tests,versus_full=confirmation_tests(summary,rows)
        comparisons={}
        for a,b in (("base","enhanced100"),("enhanced100","enhanced")):
            one=comparisons[a+"-"+b]={}
            for ds in ("paired","general","sentinel"):
                sub=[r for r in rows if (r["sentinel"] if ds=="sentinel" else r["dataset"]==ds)]
                one[ds]={k:rubric.transitions({r["id"]:r[k] for r in sub if r["arm"]==a},
                                               {r["id"]:r[k] for r in sub if r["arm"]==b}) for k in ("answer","joint")}
            one["groups"]={k:rubric.transitions({g:v[k] for g,v in summary["groups"][a].items()},
                                               {g:v[k] for g,v in summary["groups"][b].items()}) for k in ("er","joint","answer")}
        summary.update(against_ordinary50=tests,against_enhanced100=versus_full,enhanced100_transitions=comparisons,
                       tests={**tests,**{"vs100_"+k:v for k,v in versus_full.items()}},
                       promote=all(tests.values()) and all(versus_full.values()))
        immutable("summary.json",summary)

    with patch.object(rubric,"obj",save_summary):
        rubric.run()
    s=read(REVIEW/"summary.json")
    decision=dict(promote=s["promote"],against_ordinary50=s["against_ordinary50"],against_enhanced100=s["against_enhanced100"],
             n_solver_calls=768,scored_sha256=sha(REVIEW/"scored.jsonl"),summary_sha256=sha(REVIEW/"summary.json"),
             diagnostic_summary_sha256=diagnostic_gate())
    judge.freeze(ROOT/"decision.json",judge.json_bytes(decision))
    print(decision)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=["packet","judge","audit","merge","score"])
    stage=cli.parse_args().stage
    diagnostic_gate()
    rubric.HERE,rubric.ROOT,rubric.REVIEW,rubric.ARMS=HERE,ROOT,REVIEW,ARMS
    rubric.records=records
    shards.ROOT,shards.REVIEW=ROOT,REVIEW
    if stage=="packet":packet()
    elif stage=="judge":run_judge()
    elif stage=="audit":
        rubric.audit()
        shards.split()
    elif stage=="merge":shards.merge()
    else:score()


if __name__=="__main__":main()
