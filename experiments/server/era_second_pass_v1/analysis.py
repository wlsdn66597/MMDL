"""Masked response review against ground truth, matched counts, and a local results viewer."""
import argparse
import csv
import hashlib
import html
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"d226_echeck_20261105"))
from analyze import answer

ARMS=("start","control","paired")
SUFFIXES=("rkeep","rreplace","rmix","read","dkeep","dreplace")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_records():
    assert all(digest(ROOT/name)==value for name,value in read(ROOT/"freeze.json").items())
    gold={r["id"]:r for r in read(ROOT/"holdout/gold.json")}
    inputs=read(ROOT/"holdout/inputs.json")
    data={}
    prompts={}
    for arm in ARMS:
        file=ROOT/"eval"/arm/"predictions.jsonl"
        if not file.exists():continue
        predictions=[json.loads(l) for l in file.read_text(encoding="utf8").splitlines()]
        assert [r["id"] for r in predictions]==[r["id"] for r in inputs[:len(predictions)]]
        for p in predictions:
            rid=p["id"];it=gold[rid]
            assert digest(ROOT/"holdout"/it["image"])==p["image_sha256"]==it["image_sha256"]
            assert prompts.setdefault(rid,p["prompt"])==p["prompt"]
            perm=list(ARMS);random.Random("audit-1518/"+rid).shuffle(perm)
            slot="ABC"[perm.index(arm)]
            result=answer(p,it["options"],it["letter"])
            data[rid,slot]=dict(arm=arm,id=rid,slot=slot,answer=result["correct"],chosen=result["letter"],
                              finish=p["finish_reason"],question=result["E_question_mark"],output=p["raw"],item=it)
    return gold,data


def preview(first,last,slots,modes,clip):
    gold,data=raw_records()
    for j in range(first,last+1):
        family=f"U{j:02}"
        for suffix in SUFFIXES:
            rid=family+"_"+suffix; it=gold[rid]
            if it["mode"] not in modes:continue
            vals=" ".join(f"{k}={v}" for k,v in sorted(it["rows"].items(),key=lambda p:float(p[1])))
            print(rid, it.get("direct_name",""), "rows",vals, "draft",it.get("draft",""),"gold",it["truth"],it["letter"])
            for slot in slots:
                row=data.get((rid,slot))
                if not row:continue
                raw=row["output"]; excerpt=raw if len(raw)<=clip else raw[:clip-165]+" [... clipped ...] "+raw[-165:]
                print(f" {slot} {'OK' if row['answer'] else 'ERR'} | "+excerpt.replace("\n"," | "))


def score():
    gold,data=raw_records()
    assert len(data)==216
    labels=read(ROOT/"labels.json")
    assert set(labels)==set(gold)
    reviews=[]
    for rid,marks in labels.items():
        assert len(marks)==3
        for slot,mark in zip("ABC",marks):
            row=data[rid,slot];it=row["item"];natural=it["mode"]!="draft"
            if natural:
                assert mark[0] in "rwu" and mark[1:] in ("","+","!","+!")
                initial={"r":"right","w":"wrong","u":"unclear"}[mark[0]]
                joint="+" in mark;repair="!" in mark
                assert not (joint or repair) or row["answer"],(rid,slot,mark)
                assert not repair or initial=="wrong"
            else:
                initial=None;joint=None;repair=None
                assert mark in ("x","-","k","g","g*") and (mark=="-")==(not row["answer"]),(rid,slot,mark)
                assert mark!="k" or rid.endswith("dkeep")
            reviews.append(dict(id=rid,arm=row["arm"],slot=slot,mode=it["mode"],mark=mark,answer=row["answer"],
                initial=initial,joint=joint,repair=repair,retain=not natural and rid.endswith("dkeep") and mark in ("g","k"),
                supported=not natural and mark in ("g","g*"),full_check=not natural and mark=="g",
                asked=row["question"],finish=row["finish"],chosen=row["chosen"]))
    stats={};records={(r["arm"],r["id"]):r for r in reviews}
    for arm in ARMS:
        own=[r for r in reviews if r["arm"]==arm]
        nat=[r for r in own if r["mode"]!="draft"]
        keep=[r for r in own if r["id"].endswith("_dkeep")]
        change=[r for r in own if r["id"].endswith("_dreplace")]
        def pairs(s,ref):
            return sum(bool(records[arm,f"U{j:02}_"+s+"keep"][ref] and records[arm,f"U{j:02}_"+s+"replace"][ref]) for j in range(1,13))
        stats[arm]=dict(n=48,answer=sum(r["answer"] for r in nat),joint=sum(r["joint"] for r in nat),
            initial_wrong=sum(r["initial"]=="wrong" for r in nat),unclear=sum(r["initial"]=="unclear" for r in nat),repair=sum(r["repair"] for r in nat),
            keep_answer=sum(r["answer"] for r in keep),retain=sum(r["retain"] for r in keep),keep_supported=sum(r["supported"] for r in keep),
            replace_answer=sum(r["answer"] for r in change),replace_supported=sum(r["supported"] for r in change),
            replace_full_check=sum(r["full_check"] for r in change),draft_both_full_check=pairs("d","full_check"),
            draft_both_answer=pairs("d","answer"),draft_both_supported=pairs("d","supported"),natural_pair_joint=pairs("r","joint"),
            natural_rank_questions=sum(r["mode"]=="rank" and r["asked"] for r in nat),
            natural_direct_questions=sum(r["mode"]=="direct" and r["asked"] for r in nat),
            by_type={mode:dict(n=sum(r["mode"]==mode for r in nat),joint=sum(r["joint"] for r in nat if r["mode"]==mode)) for mode in ("rank","direct")},
            truncated=sum(r["finish"]=="length" for r in own))
    comp={}
    for arm in ARMS[1:]:
        for other in ARMS[:ARMS.index(arm)]:
            counts={}
            for key in ("answer","joint"):
                tally=Counter()
                for (name,rid),row in records.items():
                    if name!=arm or row["mode"]=="draft":continue
                    before=records[other,rid][key];now=row[key]
                    tally[("kept" if now else "lost") if before else ("gained" if now else "both_wrong")]+=1
                counts[key]=dict(tally)
            comp[f"{other}-{arm}"]=counts
    (ROOT/"metrics.json").write_text(json.dumps(dict(arms=stats,compared=comp),indent=2)+"\n",encoding="utf8")
    with (ROOT/"scores.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(reviews[0]));writer.writeheader();writer.writerows(reviews)
    sections=[]
    for rid,it in gold.items():
        cards=[]
        for arm in ARMS:
            row=records[arm,rid]; raw=data[rid,row["slot"]]["output"]
            cards.append(f'<article><h3>{arm}</h3><p>정답 {row["answer"]} · 검수 {row["mark"]}</p><pre>{html.escape(raw)}</pre></article>')
        sections.append(f'<section id="{rid}"><h2>{rid}</h2><p>{html.escape(it["question"])}</p><p>정답 {it["letter"]} ({it["truth"]}) · '+html.escape(" / ".join(it["options"]))+f'</p><img src="holdout/{it["image"]}" width="560" alt="차트"><div>'+''.join(cards)+'</div></section>')
    head='<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:16px/1.55 system-ui;max-width:1400px;margin:auto;padding:18px}div{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}article{background:#f2f5f4;padding:12px}pre{white-space:pre-wrap}img{max-width:100%}section{border-top:1px solid #abc}@media(max-width:840px){div{display:block}}</style>'
    legend='r/w/u는 첫 선택 정답·오답·불명, +는 ERA 연결까지 성공, !는 이미지로 실제 수정. 초안 검사 g는 근거 있는 유지·교체, g*는 교체는 맞지만 후속 확인에 오류, k는 맞는 초안을 유지했으나 경계 비교가 없거나 부정확, x는 최종 정답만, -는 오답.'
    (ROOT/"review.html").write_text('<!doctype html><html lang="ko">'+head+'<title>ERA 두 번째 확인 평가</title><h1>두 번째 확인 학습 평가</h1><p>'+legend+'</p>'+''.join(sections)+'</html>',encoding="utf8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--preview",nargs=2,type=int)
    parser.add_argument("--modes",default="rank,direct,draft")
    parser.add_argument("--slots",default="ABC")
    parser.add_argument("--clip",type=int,default=1400)
    args=parser.parse_args()
    if args.preview: preview(*args.preview,args.slots,args.modes.split(","),args.clip)
    else: score()
