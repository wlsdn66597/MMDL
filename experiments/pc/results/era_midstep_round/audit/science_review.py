"""Display complete blinded ScienceQA responses in source-ID order and emit reviewed predicates."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

HERE=Path(__file__).resolve().parent
SPEC=HERE.parent/"eval/spec/science"
LABELS=("E","R","A","consistent","answer","first_E_correct","image_corrected")


def load(p):
    return json.loads(p.read_text(encoding="utf8"))


def sheets():
    rows=sorted(load(SPEC/"inputs.json"),key=lambda r:r["id"])
    for i in range(0,len(rows),3):
        canvas=Image.new("RGB",(1320,530),"white")
        draw=ImageDraw.Draw(canvas)
        for j,item in enumerate(rows[i:i+3]):
            with Image.open(SPEC/item["image"]) as im:
                thumb=ImageOps.contain(im.convert("RGB"),(435,490))
            canvas.paste(thumb,(440*j+(440-thumb.width)//2,27))
            draw.text((440*j+10,4),item["id"],fill="black")
        canvas.save(HERE/f"science_{i//3}.jpg",quality=94)


def cards(f,group):
    rows=sorted((json.loads(x) for x in f.read_text(encoding="utf8").splitlines()),key=lambda r:(r["id"],r["key"]))
    ids=sorted({r["id"] for r in rows})
    for r in rows:
        if group is None or r["id"] in ids[group*3:group*3+3]:
            print(r["key"][:12],r["id"],r["finish"],r["reference"]["gold"],r["reference"]["visual_witness"])
            print(r["raw"],"\n")


def labels(file,decisions):
    rows=[json.loads(x) for x in file.read_text(encoding="utf8").splitlines()]
    lookup={r["key"][:12]:r for r in rows}
    assert len(rows)==len(lookup)
    out={}
    for line in decisions.read_text(encoding="utf8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        prefix,bits,reason=line.split("|")
        assert prefix in lookup and len(bits)==len(LABELS) and reason.strip(),line
        row=lookup[prefix]
        assert row["key"] not in out
        vals={k:{"1":True,"0":False,"?":None}[b] for k,b in zip(LABELS,bits)}
        out[row["key"]]=dict(key=row["key"],id=row["id"],**vals,reason=reason)
    assert len(out)==len(rows),(len(out),len(rows))
    dst=file.parent/"science_labels.jsonl"
    dst.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in out.values()),encoding="utf8")
    print(len(out),hashlib.sha256(dst.read_bytes()).hexdigest())


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=["images","cards","labels"])
    ap.add_argument("--group",type=int)
    ap.add_argument("--file",type=Path,default=HERE.parent/"eval/review/science.jsonl")
    ap.add_argument("--decisions",type=Path,default=HERE/"science_decisions.txt")
    args=ap.parse_args()
    if args.mode=="images":sheets()
    elif args.mode=="cards":cards(args.file,args.group)
    else:labels(args.file,args.decisions)
