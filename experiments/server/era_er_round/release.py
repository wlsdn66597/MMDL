"""Issue a stage signal only after verifying its referenced records."""
import argparse
from pathlib import Path
from campaign import ROOT,read,sha,write,stamp


def main():
    cli=argparse.ArgumentParser();cli.add_argument("stage",choices=["build","review","decision"]);a=cli.parse_args()
    if a.stage=="build":
        files=["protocol.md","campaign.py","train/train_er.py","train/verify_training.py","train/data/receipt.json",
               "data/rows.jsonl","data/receipt.json","paired/receipt.json","paired/overlap_final.json","general/items.jsonl",
               "eval/infer.py","eval/spec/inputs.jsonl","eval/spec/gold.jsonl","eval/spec/cached_base.jsonl","eval/spec/receipt.json"]
        receipt=read(ROOT/"train/data/receipt.json")
        assert receipt["pair_sha256"]==sha(ROOT/"data/rows.jsonl")
        assert sha(ROOT/"general/items.jsonl")==read(ROOT/"general/receipt.json")["items_sha256"]
        refs=read(ROOT/"paired/overlap_final.json")["reference_hashes"]
        for tag,file in (("round7_general","general/items.jsonl"),("round7_train","data/rows.jsonl")):
            assert sha(ROOT/file) in refs[tag].values()
    elif a.stage=="review":
        files=["eval/grade.py","eval/score.py","eval/spec/inputs.jsonl","eval/spec/gold.jsonl"]
    else:
        files=["decision.json","eval/review/summary.json","eval/review/scored.jsonl","eval/review/manual.jsonl"]
    payload=dict(stage=a.stage,hashes={s:sha(ROOT/s) for s in files})
    if a.stage=="decision":
        d=read(ROOT/"decision.json");payload["promote"]=d["promote"]
        assert d["summary_sha256"]==sha(ROOT/"eval/review/summary.json") and type(d["promote"])==bool
        assert d["promote"]==read(ROOT/"eval/review/summary.json")["promote"]
    path=ROOT/"signals"/(a.stage+".json")
    if path.exists():assert all(read(path)[k]==v for k,v in payload.items())
    else:write(path,payload|dict(time_kst=stamp()))
    print(path)


if __name__=="__main__":main()
