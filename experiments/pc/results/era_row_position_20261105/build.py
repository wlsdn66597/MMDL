"""Freeze the 2x2 chart treatments, answers and bottom-row predictions before inference."""
import csv
import hashlib
import json
import random
import sys
from decimal import Decimal as D
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "era_visual_pairs_20261105"
sys.path.insert(0, str(PRIOR))
from build_pairs import paired_chart, fmt, dump, sha, REV

# Replace either the third or fourth smallest value to swap exactly those ranks.
SPECS = [
    ("30879216000290", "Suriname", "4.7", "%"),
    ("08995001000112", "No access to handwashing facility", "542", "deaths"),
    ("13704496008944", "All ages", "1.60", "deaths per 100,000"),
    ("20294671002019", "Nicaragua", "250", "deaths per 100,000"),
    ("20822516005121", "Malta", "3.0", "deaths per 100,000"),
    ("23366484000458", "World", "41.5", "%"),
    ("24259131001104", "Central and Western Asia", "29", "%"),
    ("34384593003052", "Finland", "22", "%"),
    ("38384570001899", "Sugar cane/beet", "5.2", "%"),
    ("39727570004541", "South America", "60.75", "grams per person per day"),
    ("42096136022189", "Low income economies", "210.1", "million"),
    ("47526188010356", "High income", "31.77", "%"),
]
SLOTS = [(3,2),(1,0),(2,3),(0,1),(3,0),(2,1),(0,3),(1,2),(3,1),(2,0),(1,3),(0,2)]

def rgb_hash(path):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()

def operands(vals, keys):
    return dict(names=list(keys),values=[fmt(vals[k]) for k in keys],
                total=fmt(sum(vals[k] for k in keys)))

def validate_outside_rows(a, b, count, changed_indices):
    diff = ImageChops.difference(a, b)
    assert diff.getbbox()
    # Check all pixels outside the two moved row slots, including title, axis and footer.
    for idx in changed_indices:
        lo = round(130 + 448*idx/count)
        hi = round(130 + 448*(idx+1)/count)
        assert diff.crop((0,lo,diff.width,hi)).getbbox()
        diff.paste((0,0,0), (0,lo,diff.width,hi))
    assert diff.getbbox() is None

def build():
    prior_pair = next(p for p in json.loads((PRIOR/"pairs.json").read_text(encoding="utf8")) if p["pair"]=="P08")
    q=prior_pair["source_question"]
    pilot=ROOT.parents[1]/"code/training/era_decoder_lora/data/pilot200/images"
    train_hashes={rgb_hash(img) for img in pilot.glob("*.png")}
    assert train_hashes
    train_records=ROOT.parent/"era_pilot_200_20261028_r2/source_records.jsonl"
    train_names={Path(json.loads(s)["image_path"]).name for s in train_records.read_text(encoding="utf8").splitlines()}
    fdir=ROOT/"images"; fdir.mkdir(exist_ok=True)
    inputs=[]; items=[]
    for i,(stem,changed_name,new_value,unit) in enumerate(SPECS):
        fid=f"F{i+1:02}"
        csv_path=ROOT/"sources"/(stem+".csv"); source_png=csv_path.with_suffix(".png")
        assert stem+".png" not in train_names and rgb_hash(source_png) not in train_hashes
        with csv_path.open(encoding="utf-8-sig",newline="") as f:
            r=csv.reader(f); _,title=next(r); rows=list(r)
        old={n:D(v) for n,v in rows}
        assert len(old)==len(rows)==len(set(old.values())) and len(old)>=4
        new=dict(old); new[changed_name]=D(new_value)
        assert len(set(new.values()))==len(old)
        o0=sorted(old,key=old.get,reverse=True); o1=sorted(new,key=new.get,reverse=True)
        assert list(old)==o0 and o0[:-4]==o1[:-4] and o0[-2:]==o1[-2:] and o0[-4:-2]==o1[-4:-2][::-1]
        limit_mag=D(10)**max(old.values()).adjusted()
        limit=(max(max(old.values()),max(new.values()))/limit_mag).to_integral_value(rounding="ROUND_CEILING")*limit_mag
        cells={}; rendered={}
        for v,vals in enumerate((old,new)):
            for o,order in enumerate((o0,o1)):
                cond=f"v{v}o{o}"; rid=f"{fid}_{cond}"
                path=fdir/(rid+".png"); vals_ordered={key:vals[key] for key in order}
                im=paired_chart(title,vals_ordered,limit,unit); im.save(path)
                rendered[v,o]=im
                truth=operands(vals,sorted(vals,key=vals.get)[:3]); bottom=operands(vals, order[-3:])
                aligned=set(truth["names"])==set(bottom["names"])
                assert aligned==(v==o) and (truth["total"]==bottom["total"])==aligned
                cells[cond]=dict(id=rid,aligned=aligned,rows={n:fmt(vals[n]) for n in order},
                                 truth=truth,bottom=bottom)
        for v in (0,1): validate_outside_rows(rendered[v,0],rendered[v,1],len(old),[len(old)-4,len(old)-3])
        for o,order in enumerate((o0,o1)): validate_outside_rows(rendered[0,o],rendered[1,o],len(old),[order.index(changed_name)])
        true0=cells["v0o0"]["truth"]["total"]; true1=cells["v1o0"]["truth"]["total"]
        assert true0!=true1
        if i==0:
            options=prior_pair["options"]
            assert sha(fdir/(fid+"_v0o0.png"))==sha(PRIOR/"images/P08_control.png")
            assert sha(fdir/(fid+"_v1o0.png"))==sha(PRIOR/"images/P08_changed.png")
        else:
            options=[None]*4
            a,b=SLOTS[i];options[a]=true0;options[b]=true1
            candidates=[cells["v0o1"]["bottom"]["total"],cells["v1o0"]["bottom"]["total"],
                        fmt(sum(sorted(old.values())[:2])),fmt(sum(sorted(old.values())[:4]))]
            for value in candidates:
                if None in options and value not in options: options[options.index(None)]=value
        assert len(set(options))==4 and None not in options
        for cell in cells.values():
            for mode in ("truth","bottom"):
                cell[mode]["letter"]="ABCD"[options.index(cell[mode]["total"])]
            image="images/"+cell["id"]+".png"
            inputs.append(dict(id=cell["id"],question=q,options=options,image=image,image_sha256=sha(ROOT/image)))
        items.append(dict(chart=fid,source_csv=csv_path.name,source_csv_sha256=sha(csv_path),source_png_sha256=sha(source_png),
                          question=q,options=options,changed=changed_name,old=fmt(old[changed_name]),new=fmt(D(new_value)),cells=cells))
    assert len({i["source_csv"] for i in items})==12 and len(inputs)==48
    random.Random(3407).shuffle(inputs)
    dump(ROOT/"items.json",items); dump(ROOT/"inputs.json",inputs)
    dump(ROOT/"design_receipt.json",dict(revision=REV,charts=len(items),inputs=len(inputs),pilot_exact_image_overlap=0,
         value_increases=sum(D(i["new"])>D(i["old"]) for i in items),value_decreases=sum(D(i["new"])<D(i["old"]) for i in items),
         train_pixel_hashes=len(train_hashes),items_sha256=sha(ROOT/"items.json"),inputs_sha256=sha(ROOT/"inputs.json")))

if __name__=="__main__": build()
