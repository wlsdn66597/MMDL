"""Build reviewer contact sheets from the unchanged source ZIP for the selected 24."""
import argparse
from io import BytesIO
import json
from pathlib import Path
import zipfile
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parent
PRELIMINARY="v_6 v_8 v_13 v_15 v_36 v_45 v_49 v_57 v_58 v_106 v_147 v_198 c_27 c_38 c_131 c_158 c_239 c_285 c_292 c_380 c_683 c_816 c_936 c_962".split()


def make(path,ids):
    out=ROOT/"audit";out.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as z:
        index={r["id"]:r for r in json.loads(z.read("data.json"))}
        for off in range(0,len(ids),4):
            canvas=Image.new("RGB",(1330,950),"white");draw=ImageDraw.Draw(canvas)
            for j,rid in enumerate(ids[off:off+4]):
                r=index[rid]
                with Image.open(BytesIO(z.read(r["image_1"]))) as im:
                    thumb=ImageOps.contain(im.convert("RGB"),(650,440))
                x=j%2*665;y=j//2*475
                canvas.paste(thumb,(x+(665-thumb.width)//2,y+24))
                draw.text((x+7,y+5),rid,fill="black")
            canvas.save(out/f"group_{off//4}.jpg",quality=94)
        (out/"source_rows.json").write_text(json.dumps([index[k] for k in ids],ensure_ascii=False,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("zip",type=Path)
    ap.add_argument("--ids",nargs="+",default=PRELIMINARY)
    a=ap.parse_args()
    make(a.zip,a.ids)
