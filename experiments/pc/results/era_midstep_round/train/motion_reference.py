"""Check all motion sources once, before inspecting any solver generation."""
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageOps

HERE=Path(__file__).resolve().parent
SPEC=HERE.parent/"eval/spec/charts"
items={r["id"]:r for r in json.loads((SPEC/"inputs.json").read_text(encoding="utf8"))}
sources=[r for r in json.loads((SPEC/"gold.json").read_text(encoding="utf8")) if r["domain"]=="motion"]
assert len(sources)==32
montage=Image.new("RGB",(1380,1380),"white")
draw=ImageDraw.Draw(montage)
scenes=sorted({r["scene"] for r in sources})
for g in sources:
    s=g["source"]
    rule=g["rule_index"]
    required=s["end"]-s["start"] if rule==0 else abs(s["turn"]-s["start"])+abs(s["end"]-s["turn"])
    assert int(g["value"])==required
    im=items[g["id"]]
    assert im["options"][ord(g["answer"])-65]==str(required)
    if g["order"]==0 and rule==0:
        x=g["variant"]*690
        y=scenes.index(g["scene"])*345
        draw.text((x+4,y+3),g["scene"]+f" variant {g['variant']}", fill="black")
        with Image.open(SPEC/im["image"]) as pic:
            drawn=ImageOps.contain(pic.convert("RGB"),(680,323))
        montage.paste(drawn,(x+4,y+20))
montage.save(HERE/"motion_reference.jpg",quality=94)
