"""Print a subset of one anonymized audit shard; inputs contain no solver identities."""
import argparse
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("shard",type=int)
    parser.add_argument("--start",type=int,default=0);parser.add_argument("--n",type=int,default=6)
    args=parser.parse_args()
    rows=list(map(json.loads,(HERE/"review"/f"audit_{args.shard}.jsonl").read_text(encoding="utf8").splitlines()))
    for n,r in enumerate(rows[args.start:args.start+args.n],args.start):
        print(n,r["key"][:12],r["id"],r["image"],r["question"],r["options"],r["gold"])
        ref=r["reference"]
        print("REF",json.dumps({k:v for k,v in ref.items() if k!="chart"},ensure_ascii=False)[:1800])
        print("RAW",r["raw"])
        print("JUDGE",r["judge"],"\n")
