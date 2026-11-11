"""Show human-reviewable proof lines and flag candidate source or numerical issues."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent

EXCLUSIONS={
    "chart-11925":"Source table contradicts printed chart values: the panel says 50.7 in 2010, 22.6 in 1990 and 35.7 in 2000.",
    "chart-5802":"The graphic gives both not-interested (30) and interested (69) for China; question fails to specify the series.",
    "chart-OECD_GROSS_NATIONAL_INCOME_CZE_DEW_ESP_HKG_MAR_000021":"The printed bars are 18.9k etc.; the unfactored options omit the multiplier/unit.",
    "chart-OECD_TOTAL_OFFICIAL_AND_PRIVATE_FLOWS_CZE_ITA_LTU_NLD_USA_000058":"The printed bars carry a k suffix but the answer scale is unqualified.",
    "chart-two_col_102574":"Requested seafood label and its value are truncated; all three printed readings cannot be verified.",
    "chart-two_col_104864":"The desired 50 and 45 are not printed next to the cut-off category labels; question specifically asks for printed readings.",
    "chart-two_col_3163":"The percent for 'It will not affect' is cropped after the first digit; exact value cannot be read directly.",
    "chart-two_col_62824":"The tiny slice for revenue sharing is unlabeled in this image.",
}


def numset(text):
    clean=text.replace("−","-").replace("–","-")
    clean=re.sub(r'(?<=\d)[, ](?=\d{3}(?:\D|$))','',clean)
    return {Decimal(v) for v in re.findall(r"(?<![\d.])[-+]?\d+(?:\.\d+)?",clean)}


def decision(raw):
    m=re.search(r"(?im)^\s*[* ]*final\s+answer[* ]*:\s*[* (]*([A-E])\b",raw)
    return m[1] if m else None


def latest():
    pred=HERE/"teacher_predictions.jsonl"
    if not pred.is_file():
        pred=HERE/"teacher_pred_snapshot.jsonl"
    return {r['id']:r for r in (json.loads(s) for s in pred.read_text(encoding="utf8").splitlines())}


def show(args):
    g={r["id"]:r for r in json.loads((HERE/"teacher_gold.json").read_text(encoding="utf8"))}
    trainer=set(json.loads((HERE.parent/"train/second_review/selection.json").read_text(encoding="utf8"))["ids"])
    data=latest(); seen=0
    for sid,r in sorted(data.items()):
        gold=g[sid]
        if args.schema and gold["schema"] not in args.schema:continue
        if args.manual and sid in trainer:continue
        if args.id and sid not in args.id:continue
        a=decision(r['raw'])
        if args.bad and (sid not in EXCLUSIONS and a==gold["answer"] and r["finish_reason"]=="stop" and all(Decimal(v) in numset(r["raw"]) for v in gold.get("intermediate",[]))):continue
        if seen<args.start:seen+=1;continue
        if seen >= args.start+args.count: break
        seen+=1
        print(sid,'gold',gold['answer'],gold['answer_text'],'actual',a,'finish',r['finish_reason'], 'inter',gold.get('intermediate',[]),'EXC',EXCLUSIONS.get(sid,''))
        if args.full:
            print(r['raw'])
        else:
            texts=[s.strip() for s in r['raw'].splitlines() if s.strip()]
            important=[t for t in texts if any(Decimal(v) in numset(t) for v in gold.get("intermediate",[])) or (gold['stratum']=='direct' and gold['answer_text'].replace('−','-') in t)]
            print((' | '.join(important) if important else ' | '.join(texts[:3]))[:args.chars])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--schema",nargs="+");p.add_argument("--id",nargs="+")
    p.add_argument("--manual",action="store_true");p.add_argument("--bad",action="store_true")
    p.add_argument("--full",action="store_true");p.add_argument("--start",type=int,default=0);p.add_argument("--count",type=int,default=999);p.add_argument("--chars",type=int,default=390)
    show(p.parse_args())
