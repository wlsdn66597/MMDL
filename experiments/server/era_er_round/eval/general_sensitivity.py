"""Image-review secondary general-score disagreements; leave frozen gate decisions intact."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;D=HERE/"review"
def rows(n):
    return list(map(json.loads,(D/(n+".jsonl")).read_text(encoding="utf8").splitlines()))
def save(n,v):
    buf="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in v).encode()
    path=D/(n+".jsonl")
    if path.exists():assert path.read_bytes()==buf
    else:path.write_bytes(buf)
def packet():
    scores=defaultdict(dict)
    for r in rows("scored"):
        if r["dataset"]=="general":scores[r["id"]][r["arm"]]=r
    done={r["key"] for f in ("manual","answer_only") for r in rows(f)}
    marked={i for i,v in scores.items() if v["ordinary"]["joint"]!=v["enhanced"]["joint"] or any(
        r["key"] not in done and (r["answer"] and not r["joint"] or r["first"] is False and r["corrected"] is None) for r in v.values())}
    packet={r["key"]:r for r in rows("packet")}
    keys={r["key"] for i in marked for r in scores[i].values()}-done
    save("general_extra_cards",[packet[k] for k in sorted(keys)])
    print(len(marked),len(keys))

def finish():
    cards={r["key"] for r in rows("general_extra_cards")}
    patches={r["key"]:r for r in rows("general_extra_labels")}
    assert patches.keys()==cards
    scores=rows("scored")
    by=defaultdict(list)
    for r in scores:
        if r["dataset"]!="general":continue
        if r["key"] in patches:
            fix=patches[r["key"]]
            assert set(fix["fields"])<=set(("e","r","a","consistent","first","corrected")) and fix["reason"].strip()
            r.update(fix["fields"])
        flags=[r[k] for k in ("answer","e","r","a","consistent")]
        r["joint"]=False if False in flags else None if None in flags else True
        by[r["arm"]].append(r)
    audit={}
    for arm,rr in by.items():
        assert len(rr)==96
        wrong=[r for r in rr if r["first"] is False]
        audit[arm]=dict(answer=sum(x["answer"] for x in rr),joint=sum(x["joint"] is True for x in rr),unknown=sum(x["joint"] is None for x in rr),
                        first_errors=len(wrong),repaired=sum(r["corrected"] is True and r["answer"] for r in wrong),
                        possible_repaired=sum(r["corrected"] is not False and r["answer"] for r in wrong))
    (D/"general_sensitivity.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf8")
    print(audit)
if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("mode",choices=["packet","finish"])
    if cli.parse_args().mode=="packet":packet()
    else:finish()
