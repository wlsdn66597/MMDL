"""Read one anonymized image group with shared context shown once."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def main():
    cli=argparse.ArgumentParser();cli.add_argument("shard",type=int);cli.add_argument("groups",nargs="*")
    a=cli.parse_args()
    rows=list(map(json.loads,(HERE/"review"/f"audit_{a.shard}.jsonl").read_text(encoding="utf8").splitlines()))
    if not a.groups:
        unique={}
        for ix,r in enumerate(rows):unique.setdefault(r["group"],[]).append(ix)
        for g,ii in unique.items():print(g,ii)
        return
    prev=None
    for i,r in enumerate(rows):
        if r["group"] not in a.groups:continue
        if r["id"]!=prev:
            print(r["id"],r["image"],r["question"],r["options"],"GOLD",r["gold"])
            print("REFERENCE",json.dumps({k:v for k,v in r["reference"].items() if k!="chart"},ensure_ascii=False)[:3000])
            prev=r["id"]
        print(i,r["key"][:12],r["finish"],r["raw"])
        print("JUDGE",r["judge"])
if __name__=="__main__":main()
