"""Prepare image/contact review packets and assemble the two native-teacher reviews."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf8")


def native():
    inp={r["id"]:r for r in read(ROOT/"data/teacher_inputs.json")}
    gold=read(ROOT/"data/teacher_gold.json")
    return [(r,inp[r["id"]]) for r in sorted(gold,key=lambda r:r["id"]) if r["schema"]=="science_native"]


def sheets(rows):
    for offset in range(0,len(rows),4):
        out=Image.new("RGB",(1140,830),"white")
        dr=ImageDraw.Draw(out)
        for n,(g,r) in enumerate(rows[offset:offset+4]):
            x=(n%2)*570
            y=(n//2)*415
            with Image.open(ROOT/"data"/r["image"]) as im:
                im=ImageOps.contain(im.convert("RGB"),(558,380))
            out.paste(im,(x+(570-im.width)//2,y+25))
            dr.text((x+8,y+4),g["id"],fill="black")
        out.save(HERE/f"native_{offset//4}.jpg",quality=92)


def cards(rows, raw):
    preds={r["id"]:r for r in (json.loads(v) for v in raw.read_text(encoding="utf8").splitlines())}
    return [dict(id=g["id"],prompt=r["prompt"],gold=g["answer"],value=g["answer_text"],
                 reference=g["source_solution"],raw=preds[g["id"]]["raw"],finish=preds[g["id"]]["finish_reason"]) for g,r in rows if g["id"] in preds]


def assemble(rows,raw):
    allrows={}
    preds={r["id"]:r for r in (json.loads(v) for v in raw.read_text(encoding="utf8").splitlines())}
    for parity,subset in (("even",rows[::2]),("odd",rows[1::2])):
        labels=read(ROOT/f"data/review_native_{parity}.json")
        assert set(labels)=={g["id"] for g,_ in subset}
        for k,row in labels.items():
            assert isinstance(row["accept"],bool) and row["reason"].strip()
            if row["accept"]:
                assert all(row.get(field,"").strip() for field in ("evidence","rule","application","teacher_quote"))
                assert row["teacher_quote"] in preds[k]["raw"], (k,"teacher span not verbatim")
            allrows[k]=row
    assert len(allrows)==48
    save(ROOT/"data/review_native.json",allrows)


if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("mode",choices=["images","cards","assemble"])
    cli.add_argument("--raw",type=Path,default=HERE/"teacher_predictions.jsonl")
    cli.add_argument("--group",type=int)
    a=cli.parse_args()
    items=native()
    if a.mode=="images":
        sheets(items[1::2])
    elif a.mode=="cards":
        packet=cards(items[1::2],a.raw)
        if a.group is not None:
            packet=[x for x in packet if x["id"] in {g["id"] for g,r in items[1::2][4*a.group:4*a.group+4]}]
        for x in packet:
            print(x["id"],x["gold"],x["value"],"\n",x["prompt"],"\nSOURCE",x["reference"],"\nTEACHER",x["finish"],x["raw"],"\n")
    else:
        assemble(items,a.raw)
