"""Build new chart families from published validation tables before any model inference."""
import argparse
import csv
from decimal import Decimal as D
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys

from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent
sys.path.insert(0,str(PREV/"era_visual_pairs_20261105"))
from build_pairs import paired_chart, fmt, sha, dump, REV

def rgb_hash(path):
    with Image.open(path) as im:
        rgb=im.convert("RGB")
        return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()

def choose(cache):
    prior={f.stem for folder in ("era_row_position_20261105","era_visual_pairs_20261105")
           for f in (PREV/folder/"sources").glob("*.csv")}
    pilot={Path(json.loads(l)["image_path"]).stem for l in (PREV/"era_pilot_200_20261028_r2/source_records.jsonl").read_text(encoding="utf8").splitlines()}
    eligible=[]
    for f in sorted(cache.glob("*.csv")):
        if f.stem in prior|pilot: continue
        rows=list(csv.reader(f.open(encoding="utf-8-sig",newline="")))
        if not 4<=len(rows)-1<=12 or any(len(r)!=2 for r in rows): continue
        try: values=[D(r[1]) for r in rows[1:]]
        except ArithmeticError: continue
        if not all(v.is_finite() and v>=0 and abs(v.as_tuple().exponent)<=3 for v in values): continue
        if len(set(values))!=len(values) or len(set(r[0] for r in rows[1:]))!=len(values):continue
        eligible.append(f)
    assert len(eligible)>=16
    return eligible[:16]

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--select",action="store_true")
    args=ap.parse_args()
    cache=PREV.parent/"tmp/d226_echeck_chart_sources"
    selected=choose(cache)
    if args.select:
        dump(cache/"selected.json",[p.stem for p in selected]);return
    evaldir=ROOT/"fresh";images=evaldir/"images";sources=evaldir/"sources"
    images.mkdir(exist_ok=True,parents=True);sources.mkdir(exist_ok=True)
    occupied=set()
    for folder in (PREV/"d_era_extension_20261105/images",PREV.parent/"code/training/era_decoder_lora/data/pilot200/images",
                   PREV/"era_visual_pairs_20261105/sources",PREV/"era_row_position_20261105/sources"):
        for p in folder.iterdir():
            if p.suffix.lower() in (".png",".jpg",".jpeg"):occupied.add(rgb_hash(p))
    source_pixels=set();items=[];inputs=[];rng=random.Random(89061)
    question="What is the sum of smallest three bar?"
    for index,f in enumerate(selected):
        name=f"N{index+1:02}"; raw=f.with_suffix(".png")
        pixelhash=rgb_hash(raw)
        assert pixelhash not in occupied|source_pixels
        source_pixels.add(pixelhash)
        for p in (raw,f):shutil.copyfile(p,sources/p.name)
        rows=list(csv.reader(f.open(encoding="utf-8-sig")));title=rows[0][1]
        old={r[0]:D(r[1]) for r in rows[1:]}; asc=sorted(old,key=old.get)
        low3,low4=asc[2],asc[3]; new=dict(old)
        if index%2==0:
            cell=low3
            margin=(old[asc[4]]-old[low4]) if len(asc)>4 else (old[low4]-old[low3])
            value=old[low4]+margin/2
        else:
            cell=low4; value=(old[low3]+old[asc[1]])/2
        candidate=value.quantize(D("0.01"))
        if index%2==0 and old[low4]<candidate and (len(asc)==4 or candidate<old[asc[4]]): value=candidate
        if index%2==1 and old[asc[1]]<candidate<old[low3]:value=candidate
        new[cell]=value
        o0=sorted(old,key=old.get,reverse=True);o1=sorted(new,key=new.get,reverse=True)
        assert o0[-4:-2]==o1[-4:-2][::-1] and o0[-2:]==o1[-2:] and o0[:-4]==o1[:-4]
        high=max(max(old.values()),max(new.values())); step=D(10)**high.adjusted()
        limit=(high/step).to_integral_value(rounding="ROUND_CEILING")*step
        variants={}; possibilities=set(); rendered=[]
        # Long source labels are wrapped via concise ellipsis; the question uses numerical rank only.
        labels={n:(n if len(n)<=31 else n[:28]+"...") for n in old}
        assert len(set(labels.values()))==len(labels)
        for v,values in enumerate((old,new)):
            rank=sorted(values,key=values.get)[:3]
            for o,order in enumerate((o0,o1)):
                cond=f"v{v}o{o}";rid=f"{name}_{cond}"
                im=paired_chart(title,{labels[n]:values[n] for n in order},limit,"Value")
                ip=images/(rid+".png");im.save(ip);rendered.append(im)
                truth=sum(values[n] for n in rank);bottom=sum(values[n] for n in order[-3:])
                assert (truth==bottom)==(v==o)
                possibilities.update([truth,bottom])
                variants[cond]=dict(id=rid,aligned=v==o,rows={n:fmt(values[n]) for n in order},
                    truth=dict(names=rank,values=[fmt(values[n]) for n in rank],total=fmt(truth)),
                    bottom=dict(names=order[-3:],values=[fmt(values[n]) for n in order[-3:]],total=fmt(bottom)))
        for first,second in [(0,1),(2,3)]:
            box=ImageChops.difference(rendered[first],rendered[second]).getbbox();n=len(old)
            assert box and box[1]>=round(130+448*(n-4)/n) and box[3]<=round(130+448*(n-2)/n)
        while len(possibilities)<4:
            for val in (sum(sorted(old.values())[:2]),sum(sorted(old.values())[:4]),max(old.values())+1):
                possibilities.add(val)
                if len(possibilities)==4:break
        assert len(possibilities)==4
        oldgold=variants["v0o0"]["truth"]["total"];newgold=variants["v1o0"]["truth"]["total"]
        opts=[None]*4; a=index%4;b=(a+1+(index//4)%3)%4
        opts[a]=oldgold;opts[b]=newgold
        remaining=sorted({fmt(x) for x in possibilities}-{oldgold,newgold});rng.shuffle(remaining)
        for j in range(4):
            if opts[j] is None:opts[j]=remaining.pop()
        for r in variants.values():
            for kind in ("truth","bottom"):r[kind]["letter"]="ABCD"[opts.index(r[kind]["total"])]
            path="images/"+r["id"]+".png"
            inputs.append(dict(id=r["id"],question=question,options=opts,image=path,image_sha256=sha(evaldir/path)))
        items.append(dict(chart=name,table=f.name,original_png_sha256=sha(raw),csv_sha256=sha(f),
            title=title,changed=cell,original_value=fmt(old[cell]),new_value=fmt(new[cell]),options=opts,cells=variants))
    assert len(items)==16 and len(inputs)==64
    rng.shuffle(inputs);dump(evaldir/"items.json",items);dump(evaldir/"inputs.json",inputs)
    dump(evaldir/"receipt.json",dict(source_revision=REV,charts=16,conditions=64,prior_image_pixels_matched=0,
        prior_source_stem_overlap=0,inputs_sha256=sha(evaldir/"inputs.json"),items_sha256=sha(evaldir/"items.json")))
if __name__=="__main__":main()
