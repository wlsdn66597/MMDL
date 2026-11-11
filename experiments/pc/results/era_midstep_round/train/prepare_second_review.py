"""Freeze a stratified quantitative QA sample and prepare cards for independent image/transcript review."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
DATA = HERE.parent/"data"
OUT = HERE/"second_review"
SCHEMAS = ("sum_then_subtract","mean_then_subtract","impulse")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calculate(r):
    t = {k:Decimal(v) for k,v in r["terms"].items()}
    if r["schema"] == "impulse":
        j1 = t["first_force"] * t["first_end"]
        j2 = t["second_force"] * (t["second_end"] - t["first_end"])
        return [j1,j2,j1+j2], t["initial_speed"]+(j1+j2)/t["mass"]
    a,b,c = t.values()
    middle = (a+b)/2 if r["schema"]=="mean_then_subtract" else a+b
    return [middle], middle-c


def make():
    OUT.mkdir(exist_ok=True)
    gold_file = DATA/"teacher_gold.json"
    gold = json.loads(gold_file.read_text(encoding="utf8"))
    inputs = {r["id"]:r for r in json.loads((DATA/"teacher_inputs.json").read_text(encoding="utf8"))}
    key = lambda r:hashlib.sha256(("r5-second-QA/"+r["id"]).encode()).digest()
    selected = [r for schema in SCHEMAS for r in sorted((r for r in gold if r["schema"]==schema),key=key)[:10]]
    assert len(selected)==30 and len({r["id"] for r in selected})==30
    for row in selected:
        inter,value = calculate(row)
        assert inter==list(map(Decimal,row["intermediate"])) and value==Decimal(row["answer_text"])
        assert row["options"][ord(row["answer"])-65]==row["answer_text"]
        inp=inputs[row["id"]]
        assert digest(DATA/inp["image"])==inp["image_sha256"]
    definition = dict(source_sha256=digest(gold_file), frozen_input_sha256=digest(DATA/"teacher_inputs.json"),
                      selection="First ten ascending SHA-256('r5-second-QA/'+id) per numeric multistep schema",
                      ids=[r["id"] for r in selected])
    dest=OUT/"selection.json"
    if dest.exists():
        assert json.loads(dest.read_text(encoding="utf8"))==definition
    else:
        dest.write_text(json.dumps(definition,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    for page in range(5):
        canvas=Image.new("RGB",(1500,1770),"white")
        draw=ImageDraw.Draw(canvas)
        for i,row in enumerate(selected[page*6:(page+1)*6]):
            dx=750*(i%2);dy=590*(i//2)
            draw.text((dx+8,dy+4),row["id"],fill="black")
            text=" | ".join(f"{k}: {v}" for k,v in row["terms"].items()).encode("ascii","replace").decode()
            draw.text((dx+8,dy+22),text[:110],fill="black")
            with Image.open(DATA/inputs[row["id"]]["image"]) as pic:
                image=ImageOps.contain(pic.convert("RGB"),(735,535))
            canvas.paste(image,(dx+(750-image.width)//2,dy+44))
        canvas.save(OUT/f"images_{page+1}.jpg",quality=95)
    return selected


if __name__ == "__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--predictions",type=Path)
    args=cli.parse_args()
    selected=make()
    if args.predictions:
        pred={r["id"]:r for r in map(json.loads,args.predictions.read_text(encoding="utf8").splitlines())}
        cards=[dict(id=r["id"],schema=r["schema"],terms=r["terms"],intermediate=r["intermediate"],
                    answer=r["answer"],answer_text=r["answer_text"],finish_reason=pred[r["id"]]["finish_reason"],
                    raw=pred[r["id"]]["raw"]) for r in selected]
        (OUT/"cards.json").write_text(json.dumps(cards,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
