"""Compact visual review sheets with question operands keyed to immutable chart IDs."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE=Path(__file__).resolve().parent


def main():
    gold={r["id"]:r for r in json.loads((HERE/"teacher_gold.json").read_text(encoding="utf8"))}
    ins={r["id"]:r for r in json.loads((HERE/"teacher_inputs.json").read_text(encoding="utf8"))}
    out=HERE/"review_sheets"; out.mkdir(exist_ok=True)
    tr=HERE.parent/"train/second_review/selection.json"
    roster=json.loads(tr.read_text(encoding="utf8")) if tr.is_file() else {}
    tag_ids={r["id"] for r in roster.get("items",[])} if isinstance(roster,dict) else set()
    ft=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",19)
    head=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",22)
    chosen=sorted(r["id"] for r in gold.values() if r["schema"]!="science_native")
    for start in range(0,len(chosen),6):
        ids=chosen[start:start+6]
        w,h=720,535
        canvas=Image.new("RGB",(w*2,h*3),"white");draw=ImageDraw.Draw(canvas)
        for j,sid in enumerate(ids):
            inp=ins[sid];row=gold[sid];x=(j%2)*w;y=(j//2)*h
            image=Image.open(HERE/inp["image"]).convert("RGB")
            thumb=ImageOps.contain(image,(w-16,h-82))
            canvas.paste(thumb,(x+(w-thumb.width)//2,y+76))
            draw.text((x+3,y+1),sid+"  "+row["schema"],font=head,fill="black")
            text=", ".join(f"{k}={v}" for k,v in row["terms"].items())
            draw.text((x+3,y+28),text[:65],font=ft,fill="#15394c")
            draw.text((x+3,y+51),text[65:]+f"    ⇒ {row['answer_text']}",font=ft,fill="#15394c")
        canvas.save(out/f"panel{start//6:02d}.jpg",quality=90)


if __name__=="__main__":
    main()
