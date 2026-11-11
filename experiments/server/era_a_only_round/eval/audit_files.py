"""Split masked images across reviewers; merge reasons without solver identifiers."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
REVIEW=HERE/"review"


def lines(p):
    return [json.loads(s) for s in p.read_text(encoding="utf8").splitlines() if s]


def write(p,rows):
    raw="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode()
    if p.exists():assert p.read_bytes()==raw,p
    else:p.write_bytes(raw)


def split():
    groups=defaultdict(list)
    for c in lines(REVIEW/"audit_packet.jsonl"):
        groups[c["dataset"],c["group"]].append(c)
    bins=[[] for _ in range(4)];cost=[0]*4
    def weight(rs):return sum(320+len(r["raw"]) for r in rs)
    for _,rs in sorted(groups.items(),key=lambda kv:(-weight(kv[1]),kv[0])):
        i=min(range(4),key=lambda i:cost[i]);cost[i]+=weight(rs);bins[i].extend(rs)
    for i,rs in enumerate(bins):
        rs.sort(key=lambda r:(r["dataset"],r["group"],r["id"],r["key"]))
        write(REVIEW/f"audit_{i}.jsonl",rs)
    names=["packet.json","packet.jsonl","judge_complete.json","judge_identity.json","judgments.jsonl","bare.jsonl","audit_packet.jsonl",
           "audit_receipt.json",*[f"audit_{i}.jsonl" for i in range(4)]]
    out=ROOT/"audit_ready.tgz";tmp=out.with_suffix(".building")
    with tarfile.open(tmp,"w:gz",compresslevel=2) as tar:
        for n in names:tar.add(REVIEW/n,arcname="eval/review/"+n)
    tmp.replace(out)
    result=dict(sha256=hashlib.sha256(out.read_bytes()).hexdigest(),counts=[len(rs) for rs in bins])
    (ROOT/"audit_ready.json").write_text(json.dumps(result)+"\n",encoding="utf8");print(result)


def merge():
    rows=[]
    for i in range(4):
        cards={c["key"] for c in lines(REVIEW/f"audit_{i}.jsonl")}
        group=lines(REVIEW/f"manual_{i}.jsonl")
        assert len(cards)==len(group) and cards=={r["key"] for r in group}
        rows.extend(group)
    assert len(rows)==len({r["key"] for r in rows})==len(lines(REVIEW/"audit_packet.jsonl"))
    write(REVIEW/"manual.jsonl",sorted(rows,key=lambda r:r["key"]))


if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("mode",choices=["split","merge"])
    if cli.parse_args().mode=="split":split()
    else:merge()
