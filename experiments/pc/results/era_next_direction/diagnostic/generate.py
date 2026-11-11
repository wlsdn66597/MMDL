"""Render and freeze a 12-scene crossed rule/image/option diagnostic; run on CPU."""
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import re
import sys

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SEED = 17413087
FONT = Path("C:/Windows/Fonts/arial.ttf")
BOLD = Path("C:/Windows/Fonts/arialbd.ttf")
INK, BLUE, LINE = "#192e48", "#236079", "#c3d4da"
SITES = ["Alder", "Nori", "Cobalt", "Elara", "Kestrel", "Fjord", "Tamar", "Pax", "Nacre", "Lyra", "Haven", "Sol"]
ROWS = [("East", "West"), ("North", "South"), ("Lime", "Violet"), ("Day", "Night"),
        ("Case", "Stand"), ("Lens", "Mount"), ("Bowl", "Jug"), ("Sleeve", "Rack")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf8")


@lru_cache
def font(size, heavy=False):
    return ImageFont.truetype(str(BOLD if heavy else FONT), size)


def show(draw, xy, value, size=28, heavy=False, anchor="lm", color=INK):
    draw.text(xy, str(value), fill=color, font=font(size,heavy), anchor=anchor)


def pct(frac):
    return f"{(frac*100).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP):.1f}%"


def pooled(site, source, names):
    n1,a,n2,b = (source[k] for k in ("n1","a","n2","b"))
    p1,p2 = Decimal(a)/n1,Decimal(b)/n2
    qs = [f"At {site}, what percentage of all inspected units passed? Round the percentage to one decimal place.",
          f"At {site}, what is the unweighted average of the two batches' pass percentages? Round the percentage to one decimal place."]
    facts = dict(zip(names, [{"inspected":n1,"passed":a},{"inspected":n2,"passed":b}]))
    return [
        (qs[0], pct(Decimal(a+b)/(n1+n2)), facts, "sum passed / sum inspected",
         {"passed_total":a+b,"inspected_total":n1+n2}),
        (qs[1], pct((p1+p2)/2), facts, "arithmetic mean of the two unrounded batch percentages",
         {"batch_rates_as_fractions":[f"{a}/{n1}",f"{b}/{n2}"],"batch_count":2})]


def business(site, source, names):
    q,c,p,y,z,t,f = (source[k] for k in ("units1","cost1","price1","units2","cost2","price2","fixed"))
    a,b = q*(p-c),y*(t-z)
    facts = {names[0]:dict(units=q,price=p,unit_variable_cost=c),
             names[1]:dict(units=y,price=t,unit_variable_cost=z)}
    qs = [f"What is {site}'s total contribution margin from both products, in dollars?",
          f"What is {site}'s operating profit for the period, in dollars?"]
    intermediate = {"unit_margins":[p-c,t-z],"contributions":[a,b],"total_contribution":a+b}
    return [(qs[0],str(a+b),facts,"sum units × (unit price − unit variable cost)",intermediate),
            (qs[1],str(a+b-f),{**facts,"period_fixed_charge":f},"total contribution minus the period fixed charge",
             {**intermediate,"period_fixed_charge":f})]


def trajectory(site, source, unused_names):
    start,turn,end = [source[k] for k in ("start","turn","end")]
    assert (turn-start)*(end-turn)<0
    qs = [f"What is {site}'s signed displacement over the recorded interval, in metres?",
          f"How far does {site} travel along the one-dimensional track over the recorded interval, in metres?"]
    return [(qs[0],str(end-start),{"start":start,"end":end},"final position minus initial position",
             {"displacement":end-start}),
            (qs[1],str(abs(turn-start)+abs(end-turn)),source,"sum of the absolute position changes along each leg",
             {"first_leg":abs(turn-start),"second_leg":abs(end-turn)})]


def samples(index, rng):
    if index<4:
        source = dict(n1=rng.randrange(22,47),n2=rng.randrange(61,97))
        source.update(a=rng.randrange(7,source["n1"]-3),b=rng.randrange(17,source["n2"]-7))
        other = {**source,"a":source["a"]+rng.choice([2,3,4])}
        return "inspection", pooled, (source,other)
    if index<8:
        source = dict(units1=rng.randrange(15,43),cost1=rng.randrange(8,19),units2=rng.randrange(11,32),
                      cost2=rng.randrange(7,16),fixed=rng.randrange(53,160))
        source.update(price1=source["cost1"]+rng.randrange(4,16),price2=source["cost2"]+rng.randrange(3,12))
        other = {**source,"units1":source["units1"]+rng.choice([2,3,4])}
        return "operations", business, (source,other)
    sign = 1 if index%2==0 else -1
    start = rng.randrange(-7,8)
    turn = start+sign*rng.randrange(8,14)
    end = turn-sign*rng.randrange(5,16)
    return "motion", trajectory, (dict(start=start,turn=turn,end=end),
                                  dict(start=start,turn=turn,end=end-sign*rng.choice([2,3,4])))


def draw_image(path, task, site, names, data):
    img = Image.new("RGB",(900,500),"#ffffff")
    d = ImageDraw.Draw(img)
    titles = dict(inspection="Inspection record",operations="Operating statement",motion="Position over time")
    show(d,(46,50),f"{site}  /  {titles[task]}",34,True)
    d.line((46,92,854,92),fill=BLUE,width=3)
    if task in ("inspection","operations"):
        if task=="inspection":
            headers = ("Batch","Inspected","Passed")
            rows = [(names[0],data["n1"],data["a"]),(names[1],data["n2"],data["b"])]
            xs = (72,410,706)
            note = "Counts are units in the same inspection period."
        else:
            headers = ("Product","Units sold","Price / unit","Variable cost / unit")
            rows = [(names[0],data["units1"],f"${data['price1']}",f"${data['cost1']}"),
                    (names[1],data["units2"],f"${data['price2']}",f"${data['cost2']}")]
            xs = (72,350,545,737)
            note = f"Fixed charge for this period: ${data['fixed']}. No other costs."
        d.rectangle((46,130,854,203),fill="#e5eff2")
        for j,h in enumerate(headers):
            show(d,(xs[j],166),h,23,True,"lm" if j==0 else "mm")
        for i,row in enumerate(rows):
            y=242+i*83
            for j,v in enumerate(row):
                show(d,(xs[j],y),v,30,j==0,"lm" if j==0 else "mm")
            d.line((46,y+41,854,y+41),fill=LINE,width=2)
        show(d,(51,424),note,23)
    else:
        show(d,(46,119),"Position (m)",22,True)
        xs = [162,448,736]
        def y(v):
            return 283-v*5
        points = [(x,y(data[k])) for x,k in zip(xs,("start","turn","end"))]
        for v in range(-20,21,10):
            d.line((115,y(v),810,y(v)),fill=LINE,width=1)
            show(d,(95,y(v)),v,17,anchor="rm")
        d.line(points,fill=BLUE,width=4)
        for i,((x,pos),k) in enumerate(zip(points,("start","turn","end"))):
            v=data[k]
            d.ellipse((x-6,pos-6,x+6,pos+6),fill=BLUE)
            ty=pos-19 if (i!=1 or v>data["start"]) else pos+21
            box=d.textbbox((x,ty),str(v),font=font(22,True),anchor="mm")
            d.rectangle((box[0]-4,box[1]-1,box[2]+4,box[3]+1),fill="white")
            show(d,(x,ty),v,22,True,"mm")
            show(d,(x,416),f"{i} s",20,anchor="mm")
        show(d,(448,462),"Elapsed time on the one-dimensional track",21,anchor="mm")
    img.save(path)


def build():
    inputs,gold = [],[]
    image_dir = OUT/"images"
    image_dir.mkdir(exist_ok=True)
    for i,site in enumerate(SITES):
        names = ROWS[i] if i<len(ROWS) else ()
        rng = random.Random(f"{SEED}/{site}")
        for _ in range(100):
            task, calculate, tables = samples(i,rng)
            alternatives = [calculate(site,s,names) for s in tables]
            values = [entry[1] for alt in alternatives for entry in alt]
            if len(set(values))==4:
                break
        else:
            raise ValueError("Could not generate four distinct answers")
        diff = [k for k in tables[0] if tables[0][k] != tables[1][k]]
        assert len(diff)==1
        # Independently permute each group, then permute by one nonzero cyclic shift.
        rng.shuffle(values)
        # Balance answer letters per scene (every value serves once as a target in each order).
        for variant,source in enumerate(tables):
            scene = f"{task}-{site.lower()}"
            imagecode = hashlib.sha256(f"render:{SEED}:{scene}:{variant}".encode()).hexdigest()[:12]
            dest=image_dir/f"{imagecode}.png"
            draw_image(dest,task,site,names,source)
            for rule,(q,value,evidence,rule_text,intermediate) in enumerate(alternatives[variant]):
                for order in range(2):
                    offset = (1+i%3)*order
                    options = values[offset:]+values[:offset]
                    rid = hashlib.sha256(f"item:{SEED}:{site}:{variant}:{rule}:{order}".encode()).hexdigest()[:18]
                    inputs.append(dict(id=rid,image=f"images/{dest.name}",image_sha256=sha(dest),question=q,options=options))
                    gold.append(dict(id=rid,scene=scene,domain=task,variant=variant,rule_index=rule,order=order,
                                     source=source,changed_field=diff[0],evidence=evidence,rule=rule_text,
                                     intermediate=intermediate,value=value,answer="ABCD"[options.index(value)]))
    inputs.sort(key=lambda row:row["id"])
    gold.sort(key=lambda row:row["id"])
    assert len({r["id"] for r in inputs})==len(inputs)==96
    assert len({r["image_sha256"] for r in inputs})==24 and len({r["question"] for r in inputs})==24
    assert Counter(r["answer"] for r in gold)=={l:24 for l in "ABCD"}
    assert Counter(r["domain"] for r in gold)==dict(inspection=32,operations=32,motion=32)
    for s in set(r["scene"] for r in gold):
        gr=[r for r in gold if r["scene"]==s]
        assert len(gr)==8 and Counter(r["answer"] for r in gr)=={l:2 for l in "ABCD"}
    assert all(set(r)=={"id","image","image_sha256","question","options"} for r in inputs)
    write(OUT/"inputs.json",inputs)
    write(OUT/"gold.json",gold)
    # A view of three image pairs, preserving both variants at legible resolution.
    sheet = Image.new("RGB",(1350,1125),"white")
    for r,idx in enumerate([0,4,8]):
        scene_gold = next(g for g in gold if g["scene"].split("-")[1]==SITES[idx].lower() and g["variant"]==0)
        for v in (0,1):
            same = next(g for g in gold if g["scene"]==scene_gold["scene"] and g["variant"]==v)
            item = next(x for x in inputs if x["id"]==same["id"])
            with Image.open(OUT/item["image"]) as pic:
                sheet.paste(pic.resize((675,375),Image.Resampling.LANCZOS),(675*v,375*r))
    sheet.save(OUT/"representatives.png")
    return inputs,gold


def overlap(inputs):
    sys.path.insert(0,str(ROOT/"scripts/visual_groups"))
    from visual_groups import fingerprint, is_candidate
    def norm(q):
        return " ".join(re.findall(r"\w+|[%+−-]",q.lower()))
    last = ROOT/"results/era_a_pair_round"
    ref_path = ROOT/"reports/era_abcd_review_20261020/mmmu_reference_fingerprints.json"
    official = json.loads((ROOT/"results/mmmu_first_baseline_20261009/baseline_seed3407/results.json").read_text(encoding="utf8"))["records"]
    orig = json.loads(ref_path.read_text(encoding="utf8"))
    ref_images = [(r["id"],r) for r in orig]
    blocked = {norm(r["question"]) for r in official}
    for tag, path, imgs in (
            ("r4_train",last/"data_work/train.jsonl",last/"data_work"),
            ("r4_chart",last/"connection_work/spec/inputs.json",last/"connection_work/spec")):
        raw = path.read_text(encoding="utf8")
        rows = [json.loads(s) for s in raw.splitlines()] if path.suffix==".jsonl" else json.loads(raw)
        for row in rows:
            blocked.add(norm(row["question"]))
            ref_images.append((f"{tag}/{row['id']}",fingerprint(imgs/row["image"])))
    matches = {}
    for img in sorted(set(x["image"] for x in inputs)):
        fp = fingerprint(OUT/img)
        hits = [rid for rid,ref in ref_images if fp["pixel_sha256"]==ref["pixel_sha256"] or is_candidate(fp,ref)]
        if hits:
            matches[img]=hits
    return dict(original_questions=len(official),fingerprint_comparisons_per_image=len(ref_images),
                question_hits=sorted({r["question"] for r in inputs if norm(r["question"]) in blocked}),
                image_near_candidates=matches,reference_sha256=sha(ref_path))


if __name__=="__main__":
    items,refs = build()
    check = overlap(items)
    write(OUT/"manifest.json",dict(seed=SEED,items=96,scenes=12,images=24,independent_unit="scene",
                                   input_sha256=sha(OUT/"inputs.json"),gold_sha256=sha(OUT/"gold.json"),
                                   generator_sha256=sha(Path(__file__)),font_sha256=sha(FONT),
                                   bold_font_sha256=sha(BOLD),contact_sha256=sha(OUT/"representatives.png"),overlap=check))
