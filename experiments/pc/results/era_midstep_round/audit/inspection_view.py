"""Print only anonymous inspection cards; never open arm keys or prediction runs."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/"eval/review/inspection.jsonl"


def cards():
    return sorted([json.loads(s) for s in FILE.read_text(encoding="utf8").splitlines() if s], key=lambda r:r["key"])


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("start",type=int,default=0,nargs="?");p.add_argument("end",type=int,default=96,nargs="?")
    p.add_argument("--max",type=int,default=0);p.add_argument("--suspicious",action="store_true");args=p.parse_args()
    for n,r in enumerate(cards()):
        if not args.start<=n<args.end:continue
        ref=r["reference"]
        raw=r["raw"]
        print("\n",n,r["key"],r["finish"],ref["scene"],ref["variant"],ref["rule_index"],ref["source"],ref["value"],ref["gold"],ref["options"])
        if args.max and len(raw)>args.max:
            print(raw[:args.max//2],"...",raw[-args.max//2:])
        else:print(raw)
