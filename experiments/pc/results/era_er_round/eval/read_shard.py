"""Inspect an anonymous group without loading any model-to-key mapping."""
import argparse
import json
from pathlib import Path


if __name__=="__main__":
    c=argparse.ArgumentParser();c.add_argument("shard",type=int);c.add_argument("--start",type=int,default=0);c.add_argument("--n",type=int,default=5)
    a=c.parse_args()
    p=Path(__file__).resolve().parent/"review"/f"audit_{a.shard}.jsonl"
    rows=list(map(json.loads,p.read_text(encoding="utf8").splitlines()))
    for i,r in enumerate(rows[a.start:a.start+a.n],start=a.start):
        print(i,r["key"][:12],r["id"],r["image"],r["gold"],r["options"])
        print(r["question"])
        print("REF",json.dumps(r["reference"],ensure_ascii=False)[:2500])
        print("RAW",r["raw"])
        print("JUDGE",r["judge"],"\n")
