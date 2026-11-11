"""Validate immutable stage inputs and signal the tmux runner."""
import argparse
from campaign import ROOT,read,sha,stamp,write


def main():
    cli=argparse.ArgumentParser();cli.add_argument("stage",choices=("build","review","decision"));s=cli.parse_args().stage
    if s=="build":
        files=["protocol.md","campaign.py","data/rows.jsonl","data/receipt.json","train/data/receipt.json","train/train_r.py","train/verify_training.py",
               "paired/items.jsonl","paired/sources.jsonl","paired/receipt.json","paired/overlap_final.json","general/items.jsonl","general/receipt.json",
               "eval/prepare.py","eval/infer.py","eval/spec/inputs.jsonl","eval/spec/gold.jsonl","eval/spec/receipt.json"]
        proof=read(ROOT/"train/data/receipt.json")
        assert proof["pair_sha256"]==sha(ROOT/"data/rows.jsonl") and proof["n"]==296 and proof["updates"]==38
        refs=read(ROOT/"paired/overlap_final.json")["reference_hashes"]
        for label,rel in (("round8_train","data/rows.jsonl"),("round8_general","general/items.jsonl")):
            assert sha(ROOT/rel) in refs[label].values()
        for name,h in read(ROOT/"eval/spec/receipt.json").items():assert sha(ROOT/"eval/spec"/(name+".jsonl"))==h
    elif s=="review":
        files=["eval/grade.py","eval/answer_only.py","eval/score.py","eval/spec/receipt.json"]
    else:
        files=["decision.json","eval/review/summary.json","eval/review/scored.jsonl","eval/review/manual.jsonl","eval/review/audit_packet.jsonl"]
    info=dict(stage=s,hashes={f:sha(ROOT/f) for f in files})
    if s=="decision":
        d=read(ROOT/"decision.json");res=read(ROOT/"eval/review/summary.json")
        assert d["summary_sha256"]==sha(ROOT/"eval/review/summary.json") and d["promote"]==res["promote"] and type(d["promote"])==bool
        info["promote"]=d["promote"]
    dst=ROOT/"signals"/(s+".json")
    if dst.exists():assert all(read(dst)[k]==v for k,v in info.items())
    else:write(dst,info|dict(time_kst=stamp()))
    print(s)


if __name__=="__main__":main()
