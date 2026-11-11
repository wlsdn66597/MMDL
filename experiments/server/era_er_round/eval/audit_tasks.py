"""Split selected anonymous keys by original image group without disclosing solver identities."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
REVIEW=HERE/"review"
def lines(p):return list(map(json.loads,p.read_text(encoding="utf8").splitlines()))


def main():
    c=argparse.ArgumentParser();c.add_argument("mode",choices=["split","merge"]);a=c.parse_args()
    cards=lines(REVIEW/"audit_packet.jsonl")
    if a.mode=="merge":
        patches=[]
        for i in range(4):patches.extend(lines(REVIEW/f"manual_{i}.jsonl"))
        assert len(patches)==len(cards)==len({r["key"] for r in patches})
        lookup={r["key"]:r["id"] for r in cards}
        assert all(r["key"] in lookup and r["reason"].strip() for r in patches)
        assert set(lookup)=={r["key"] for r in patches}
        p=REVIEW/"manual.jsonl"
        data="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in sorted(patches,key=lambda r:r["key"])).encode()
        if p.exists():assert p.read_bytes()==data
        else:p.write_bytes(data)
        return
    judges={r["key"]:r for r in lines(REVIEW/"judgments.jsonl")}
    groups=defaultdict(list)
    for r in cards:
        j=judges[r["key"]]
        current={k:j[k] for k in ("e","r","a","consistent","first","corrected","choice")}
        groups[(r["dataset"],r["group"])].append(r|dict(judge=current))
    shards=[[] for i in range(4)]
    for group,rs in sorted(groups.items(),key=lambda kv:(-len(kv[1]),kv[0])):
        min(shards,key=len).extend(sorted(rs,key=lambda r:(r["id"],r["key"])))
    for i,items in enumerate(shards):
        path=REVIEW/f"audit_{i}.jsonl"
        path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in items),encoding="utf8")
    print([len(x) for x in shards])


if __name__=="__main__":main()
