"""Display and encode blinded, per-response motion judgments; never read the run-to-arm mapping."""
import argparse
import hashlib
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent
OUT=HERE.parent/"eval/review"
FIELDS=("E","R","A","consistent","answer","first_E_correct","image_corrected")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("read","seal"))
    p.add_argument("--scene")
    p.add_argument("--rule",type=int)
    a=p.parse_args()
    src=OUT/"motion.jsonl"
    rows=[json.loads(line) for line in src.read_text(encoding="utf8").splitlines()]
    if a.mode=="read":
        print(len(rows),"new unique text cards")
        for r in sorted(rows,key=lambda r:(r["reference"]["scene"],r["reference"]["rule_index"],r["id"],r["key"])):
            ref=r["reference"]
            if (a.scene is None or a.scene==ref["scene"]) and (a.rule is None or a.rule==ref["rule_index"]):
                print(r["key"][:10],ref["scene"],ref["evidence"],ref["rule_index"],ref["value"],ref["gold"],r["finish"],r["raw"])
        return
    keys={r["key"][:10]:r for r in rows}; assert len(keys)==len(rows)
    labels={}
    for line in (HERE/"motion_decisions.txt").read_text(encoding="utf8").splitlines():
        if line.strip() and not line.startswith("#"):
            prefix,bits,reason=line.split("|")
            assert prefix in keys and prefix not in labels and len(bits)==len(FIELDS) and reason.strip()
            r=keys[prefix]
            labels[prefix]=dict(key=r["key"],id=r["id"],**{f:{"1":True,"0":False,"?":None}[bit]
                                    for f,bit in zip(FIELDS,bits)},reason=reason)
    assert labels.keys()==keys.keys(),keys.keys()-labels.keys()
    dst=OUT/"motion_labels.jsonl"
    raw="".join(json.dumps(labels[k],ensure_ascii=False)+"\n" for k in sorted(labels)).encode()
    if dst.exists():assert dst.read_bytes()==raw
    else:dst.write_bytes(raw)
    print(len(labels),hashlib.sha256(raw).hexdigest())


if __name__=="__main__":
    main()
