"""Render only comparison contacts. No archive extraction or model inference."""
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile
from PIL import Image,ImageDraw,ImageOps
import check_overlap as co

HERE=Path(__file__).resolve().parent

def draw(pairs,name):
    for offset in range(0,len(pairs),4):
        canvas=Image.new("RGB",(1080,325*min(4,len(pairs)-offset)),"#eeeff0")
        dr=ImageDraw.Draw(canvas)
        for n,(a,ia,b,ib) in enumerate(pairs[offset:offset+4]):
            for j,(label,im) in enumerate(((a,ia),(b,ib))):
                img=ImageOps.contain(im.convert("RGB"),(520,288))
                x=j*540;y=n*325
                canvas.paste(img,(x+(540-img.width)//2,y+25))
                dr.text((x+8,y+5),label,fill="black")
        canvas.save(HERE/f"{name}_{offset//4}.jpg",quality=92)

def main():
    D=co.ROOT
    olddirs={"donor_train200":D/"code/training/era_decoder_lora/data/pilot200","old_train449":D/"results/era_single_call_campaign/data",
             "D226":D/"results/d226_echeck_20261105/data"}
    mmmu={ (r["id"],r["field"]): Path(r["image_path"]) for r in co.read(D/"reports/era_abcd_review_20261020/mmmu_reference_fingerprints.json")}
    with ZipFile(co.DEFAULT_ZIP) as z:
        v={r["id"]:r for r in json.loads(z.read("data.json"))}
        def original(rid):
            return Image.open(BytesIO(z.read(v[rid]["image_1"])))
        cross=co.read(HERE/"cross_candidates.json")
        pairs=[]
        for rid,tag,other in (("c_771","MMMU900","validation_Psychology_8"),("i_438","MMMU900","validation_Marketing_1"),
            ("v_22","donor_train200","D_ScienceQA_1022"),("v_219","donor_train200","D_ScienceQA_10518"),("v_53","D226","dp_a53ca82111f21430")):
            r=next(r for r in cross if r["set"]==tag and r["other"]==other and r["id"]==rid)
            member=r["image"][0]["image"];path=(mmmu[other,member] if tag=="MMMU900" else olddirs[tag]/member)
            pairs.append((rid,original(rid),f"{tag} / {other}",Image.open(path)))
        draw(pairs,"external")
        internal=co.read(HERE/"internal_image_pairs.json")
        draw([(r["id"],original(r["id"]),r["other"],original(r["other"])) for r in internal],"internal")

if __name__=="__main__":main()
