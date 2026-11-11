"""Generate matched charts from published ChartQA validation data tables."""
import csv
import hashlib
import json
from decimal import Decimal as D
from pathlib import Path
import random
import shutil
import textwrap

from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
SOURCE = ROOT / "sources"
IMAGES = ROOT / "images"
REV = "044eabfc306abfe9340c5741f0093aefc5973d06"
# index, expression, target cell, replacement, wrong distractors, displayed unit
SPECS = [
    (43, "sum_all", "Cyprus", "2.34", ("5.13","8.13"), "deaths per 100,000"),
    (53, "sum_all", "Spain", "33", ("19","13"), "%"),
    (57, "difference_named", "Prevalence", "1230", ("3152","3352"), "cases"),
    (63, "range", "Unsafe water source", "1515", ("1454","1715"), "deaths"),
    (71, "mean_named", "24 hours", "82.1", ("210.9","110.9"), "n mi"),
    (79, "read", "50-69 years old", "7.88", ("4.27","3.58"), "deaths per 100,000"),
    (88, "max", "Laos", "296", ("180","68"), "deaths per 100,000"),
    (109, "sum_low3", "Suriname", "4.7", ("0.5","5.2"), "%"),
    (113, "mean_low2", "Myanmar", "21", ("26","28"), "%"),
    (125, "sum_named", "Low income economies", "151.1", ("407.8","388.1"), "million"),
]
NAMED = {"difference_named": ("Incidence","Prevalence"), "mean_named": ("24 hours","48 hours"),
         "sum_named": ("Low income economies","Sub-Saharan Africa")}

def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def fmt(number):
    return format(D(number).normalize(), "f")

def solve(op, vals, cell):
    ordered = sorted(vals, key=vals.get)
    if op == "read": keys = [cell]
    elif op == "max": keys = ordered[-1:]
    elif op == "range": keys = [ordered[-1],ordered[0]]
    elif op == "sum_low3": keys = ordered[:3]
    elif op == "mean_low2": keys = ordered[:2]
    elif op in NAMED: keys = list(NAMED[op])
    else: keys = list(vals)
    nums = [vals[k] for k in keys]
    if op in ("difference_named","range"): ans = nums[0]-nums[1]
    elif op.startswith("mean"): ans = sum(nums)/len(nums)
    else: ans = sum(nums)
    return {"operands":{k:fmt(vals[k]) for k in keys},"answer":fmt(ans)}

def paired_chart(title, vals, limit, unit):
    # Both panels use the same fixed axis, row order, type sizes, labels and colours.
    im = Image.new("RGB",(1160,640),"white")
    draw = ImageDraw.Draw(im)
    def font(size,bold=False):
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf",size)
    draw.multiline_text((30,18),"\n".join(textwrap.wrap(title,72)),font=font(24,True),fill="#142837")
    draw.text((30,90),unit,font=font(18),fill="#364858")
    left, right, top, bottom = 335, 1015, 130, 578
    for j in range(6):
        x=left+(right-left)*j/5
        draw.line((x,top,x,bottom), fill="#e0e5e8", width=2)
        draw.text((x,bottom+10),fmt(limit*D(j)/5),font=font(15),anchor="mt",fill="#37414b")
    spacing=(bottom-top)/len(vals)
    for i,(name,val) in enumerate(vals.items()):
        y=round(top+(i+0.5)*spacing)
        h=min(33,max(14,int(spacing*.55)))
        end=left+round(float(val/limit)*(right-left))
        draw.text((left-12,y),name,anchor="rm",font=font(18),fill="#202832")
        draw.rectangle((left,y-h//2,max(left,end),y+h//2), fill="#307285")
        draw.text((end+9,y),fmt(val),anchor="lm",font=font(19,True),fill="#182934")
    return im

def main():
    IMAGES.mkdir(exist_ok=True)
    pilot=PROJECT/"code/training/era_decoder_lora/data/pilot200"
    records=[json.loads(s) for s in (PROJECT/"results/era_pilot_200_20261028_r2/source_records.jsonl").read_text(encoding="utf8").splitlines()]
    assert len(records)==200
    train_names={Path(r["image_path"]).name for r in records}
    def pixels(path):
        with Image.open(path) as img:
            rgb=img.convert("RGB")
            return hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
    train_pixels={pixels(p) for p in (pilot/"images").glob("*.png")}
    questions=json.loads((SOURCE/"val_human.json").read_text(encoding="utf8"))
    pairs=[]; requests=[]; checks=[]
    # Balance old and new correct option positions separately without fixed offset.
    slots=[(0,2),(2,3),(3,1),(1,0),(0,1),(1,3),(2,0),(3,2),(0,3),(2,1)]
    for i,(idx,op,cell,repl,dist,unit) in enumerate(SPECS):
        q=questions[idx]; stem=Path(q["imgname"]).stem; pid=f"P{i+1:02}"
        raw=SOURCE/q["imgname"]; table=SOURCE/(stem+".csv")
        assert q["imgname"] not in train_names and pixels(raw) not in train_pixels
        with table.open(encoding="utf-8-sig", newline="") as stream:
            reader=csv.reader(stream); _,title=next(reader)
            rows=list(reader); old={name:D(v) for name,v in rows}
        assert len(old)==len(rows) and cell in old
        new=old.copy(); new[cell]=D(repl)
        assert [name for name in old if old[name]!=new[name]] == [cell]
        g0,g1=solve(op,old,cell),solve(op,new,cell)
        assert D(str(q["label"]))==D(g0["answer"])
        options=[None]*4; s0,s1=slots[i]; options[s0]=g0["answer"]; options[s1]=g1["answer"]
        for j,val in zip([j for j in range(4) if options[j] is None],dist): options[j]=val
        assert len({D(v) for v in options})==4
        magnitude=D(10)**max(old.values()).adjusted()
        limit=((max(max(old.values()),max(new.values()))/magnitude).to_integral_value(rounding="ROUND_CEILING"))*magnitude
        before=paired_chart(title,old,limit,unit); after=paired_chart(title,new,limit,unit)
        bbox=ImageChops.difference(before,after).getbbox()
        pos=list(old).index(cell); low=130+448*pos/len(old); high=130+448*(pos+1)/len(old)
        assert bbox and low <= bbox[1] and bbox[3] <= high
        panels={"original":raw,"control":IMAGES/(pid+"_control.png"),"changed":IMAGES/(pid+"_changed.png")}
        before.save(panels["control"]); after.save(panels["changed"])
        gold={"original":dict(**g0,letter="ABCD"[s0]),"control":dict(**g0,letter="ABCD"[s0]),"changed":dict(**g1,letter="ABCD"[s1])}
        pairs.append(dict(pair=pid,chartqa_val_index=idx,filename=q["imgname"],source_question=q["query"],source_answer=q["label"],table_sha256=sha(table),operation=op,changed_cell=cell,original_value=fmt(old[cell]),changed_value=repl,gold=gold,options=options,changed_pixel_bbox_xyxy=bbox))
        for panel,path in panels.items():
            requests.append(dict(id=f"{pid}_{panel}",question=q["query"],options=options,image=str(path.relative_to(ROOT)).replace("\\","/"),image_sha256=sha(path)))
        checks.append(dict(pair=pid,training_name_overlap=False,training_rgb_overlap=False,question_template_seen_in_pilot=any(r["question"].strip().lower()==q["query"].strip().lower() for r in records)))
    random.Random(3407).shuffle(requests)
    dump(ROOT/"pairs.json",pairs); dump(ROOT/"inputs.json",requests)
    dump(ROOT/"source_checks.json",dict(chartqa_revision=REV,validation_questions_sha256=sha(SOURCE/"val_human.json"),pilot_rows=len(records),distinct_training_png_pixels=len(train_pixels),checks=checks))

if __name__=="__main__": main()
