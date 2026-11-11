"""Construct the 192 input-only training requests independently of any model outcomes."""
from collections import Counter
import csv
from decimal import Decimal
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import re
import sys
import tempfile

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SCI=ROOT/"results/era_next_direction/retention"
sys.path.insert(0,str(SCI))
from audit_sources import checked_scope, collides, fingerprint, normalized, read, signatures
from native_dev import quality_exclusion
from fetch_chart_source import CACHE, REV

SUFFIX='Show the image readings and reasoning needed to justify your answer.\nOn the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'
PREFIX="ERA-midstep-r5-192-v1/"
IMG=HERE/"images"
MATH=Decimal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(value):
    return hashlib.sha256((PREFIX+value).encode()).hexdigest()


def fmt(number):
    return format(Decimal(number).normalize(),"f")


def dump(file,data):
    buf=(json.dumps(data,ensure_ascii=False,indent=2)+"\n").encode()
    if file.exists() and file.read_bytes()!=buf:
        raise RuntimeError(f"Existing frozen bytes differ: {file}")
    file.write_bytes(buf)


def input_image(source,image_id):
    image=Image.open(source).convert("RGB")
    target=IMG/(image_id+".png")
    image.save(target)
    return target


def options(values,answer,rng):
    unique=list(dict.fromkeys(str(v) for v in [answer,*values]))
    i=0
    while len(unique)<4:
        v=fmt(Decimal(answer)+(17+i))
        if v not in unique:
            unique.append(v)
        i+=1
    opts=unique[:4]
    rng.shuffle(opts)
    return opts


def record(sid,source,group,path,q,opts,correct,meta,hint=""):
    assert len(set(opts))==len(opts) and correct in opts
    letter=chr(65+opts.index(correct))
    prompt=(("Hint: "+hint.strip()+"\n") if hint.strip() else "")+"Question: "+q+"\nOptions:\n"+"\n".join(chr(65+i)+". "+v for i,v in enumerate(opts))+"\n"+SUFFIX
    inp=dict(id=sid,prompt=prompt,image=path.relative_to(HERE).as_posix(),image_sha256=sha(path))
    gold=dict(id=sid,source=source,group=group,question=q,options=opts,answer=letter,answer_text=correct,**meta)
    return inp,gold


def block_scope():
    questions,sigs=checked_scope()
    pilot=ROOT/"code/training/era_decoder_lora/data/pilot200"
    for line in (pilot/"exports/train_era.jsonl").read_text(encoding="utf8").splitlines():
        row=json.loads(line)
        q=re.search(r"Question: (.*?)\n(?:Options:|On the last line)",row["input_prompt"],re.S)
        if q:
            questions.add(normalized(q[1]))
        sigs.add(signatures(fingerprint(pilot/row["image_path"])))
    for root in (ROOT/"results/era_next_direction/diagnostic",SCI/"native_validation"):
        for row in read(root/"inputs.json"):
            sigs.add(signatures(fingerprint(root/row["image"])))
            text=row.get("question") or row["prompt"].split("Question: ")[-1].split("\nOptions:")[0]
            questions.add(normalized(text))
    return questions,sigs


def source_charts(block_questions,blocked):
    result=[]
    seen=[]
    seen_questions=set()
    for name in sorted(read(CACHE/"eligible.json"),key=key):
        path=CACHE/(name+".png")
        sig=signatures(fingerprint(path))
        if collides(sig,blocked) or collides(sig,seen):
            continue
        with (CACHE/(name+".csv")).open(encoding="utf-8-sig",newline="") as file:
            table=list(csv.reader(file))
        series=[(v[0].strip(),Decimal(v[1].strip().replace(",","").rstrip("%"))) for v in table[1:]]
        rng=random.Random(key("chart/"+name))
        rng.shuffle(series)
        a,b,c=series[:3]
        terms={k:fmt(v) for k,v in (a,b,c)}
        idx=len(result)
        kind="direct" if idx%7 in (0,2,4) else "multi"
        if kind=="direct":
            q=f'What numeric reading does the chart give for "{a[0]}"? Use the displayed scale.'
            answer=fmt(a[1])
            alt=[fmt(v) for _,v in series[1:]]
            schema="read_label"; intermediate=[]
            ev=f'The chart labels "{a[0]}" at {answer}.'
            rule="Match the requested category to its printed reading on the chart's numeric scale."
            compact=f'The value belonging to "{a[0]}" is {answer}.'
            preserved=compact
        elif idx%2:
            q=f'Using the printed numeric readings, add the values for "{a[0]}" and "{b[0]}", then subtract the value for "{c[0]}". What number results?'
            subtotal=a[1]+b[1]; net=subtotal-c[1]
            answer=fmt(net); alt=[fmt(subtotal),fmt(subtotal+c[1]),fmt(c[1]-subtotal)]
            schema="sum_then_subtract"; intermediate=[fmt(subtotal)]
            ev="; ".join(f'"{k}" is {v}' for k,v in terms.items())+"."
            rule="Add the two requested categories and subtract the third category once."
            compact=f'The two-category total less the third reading gives {answer}.'
            preserved=f'{fmt(a[1])} + {fmt(b[1])} = {fmt(subtotal)}; {fmt(subtotal)} - {fmt(c[1])} = {answer}.'
        else:
            q=f'Find the mean of the plotted readings for "{a[0]}" and "{b[0]}". Then subtract the reading for "{c[0]}" from that mean. What number results?'
            average=(a[1]+b[1])/2; net=average-c[1]
            answer=fmt(net); alt=[fmt(average),fmt(average+c[1]),fmt(a[1]+b[1]-c[1])]
            schema="mean_then_subtract"; intermediate=[fmt(average)]
            ev="; ".join(f'"{k}" is {v}' for k,v in terms.items())+"."
            rule="Take the arithmetic mean of the first two named readings, then subtract the third."
            compact=f'The mean of the two specified bars minus the third reading is {answer}.'
            preserved=f'({fmt(a[1])} + {fmt(b[1])}) / 2 = {fmt(average)}; {fmt(average)} - {fmt(c[1])} = {answer}.'
        if normalized(q) in block_questions or normalized(q) in seen_questions:
            continue
        sid="chart-"+name
        new=input_image(path,sid)
        meta=dict(stratum=kind,schema=schema,table=dict(series=[(x,fmt(y)) for x,y in series],heading=table[0]),terms=terms,
                  intermediate=intermediate,source_sha256=[sha(path),sha(CACHE/(name+".csv"))],source_revision=REV,
                  evidence=ev,rule=rule,compact=compact,preserved=preserved)
        result.append(record(sid,"ChartQA/train (new question on original table)","ChartQA/"+name,new,q,options(alt,answer,rng),answer,meta))
        seen.append(sig)
        seen_questions.add(normalized(q))
    return result


def source_science(block_q,blocked):
    metadata=read(ROOT/"reports/era_dataset_examples_20261014/scienceqa_problems.json")
    split=read(ROOT/"reports/era_dataset_examples_20261014/scienceqa_splits.json")["train"]
    ids=[x["scienceqa_id"] for x in read(SCI/"candidate_ids_after_native_holdout.json") if x["role"]=="train_candidate"]
    def visual_quality(q):
        skill=q["skill"]
        return any(x in skill.lower() for x in ("material","graphic organizer","food web","limbs","necks","skins","animal adaptations","tectonic","climate","grid","punnett","map:","force","chemical formula"))
    eligible=sorted((i for i in ids if visual_quality(metadata[i]) and not quality_exclusion(metadata[i]) and normalized(metadata[i]["question"]) not in block_q),key=key)
    source=ROOT/"tmp/era_prereview_20261021/scienceqa_train.parquet"
    assert sha(source)==read(SCI/"source_inventory.json")["source_sha256"]
    pool=set(eligible); raw={}; idx=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=128,columns=["image"]):
        for row in batch.to_pylist():
            sid=split[idx]; idx+=1
            if sid in pool:
                raw[sid]=row["image"]["bytes"]
    assert len(raw)==len(eligible)
    result=[]
    for sid in eligible:
        m=metadata[sid]; image=input_image(BytesIO(raw[sid]),"science-"+sid)
        if collides(signatures(fingerprint(image)),blocked):
            image.unlink();continue
        row=record("science-"+sid,"ScienceQA/train","ScienceQA/"+sid,image,m["question"],m["choices"],
                   m["choices"][m["answer"]],dict(stratum="direct",schema="science_native",skill=m["skill"],source_solution=m["solution"],
                   source_revision="f18b0a70359ebfb41f658fd564208d0355b013f4",source_id=sid),m["hint"])
        result.append(row)
    return result


def physics(n,kind,blocked):
    if n==0:
        return []
    f=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",24)
    title=ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf",28)
    rows=[]
    for i in range(n*3):
        rng=random.Random(key(f"physics/{kind}/{i}"))
        mass=rng.randint(2,8); v0=rng.randint(-7,8)
        d1=rng.randint(1,5); d2=rng.randint(2,7)
        f1=rng.randint(2,8)*mass; f2=-(rng.randint(1,6)*mass)
        t2=d1+d2; scale=max(abs(f1),abs(f2))
        exp_id=(1000 if kind=="multi" else 3000)+i
        im=Image.new("RGB",(790,435),"white"); dr=ImageDraw.Draw(im)
        dr.text((28,15),f"Trolley experiment {exp_id}   /   Force sensor",font=title,fill="#162d49")
        dr.text((28,65),f"Mass: {mass} kg        Initial velocity: {v0} m/s",font=f,fill="#101822")
        x0,x1,x2=90,90+int(580*d1/t2),670
        zero=260; y1=zero-int(110*f1/scale);y2=zero-int(110*f2/scale)
        dr.line((x0,zero,x2+50,zero),fill="#798591",width=2)
        dr.line((x0,138,x0,390),fill="#798591",width=2)
        dr.line(((x0,zero),(x0,y1),(x1,y1),(x1,y2),(x2,y2),(x2,zero)),fill="#25677c",width=4)
        dr.text(((x0+x1)/2,y1-16),f"+{f1} N",font=f,fill="#173d4f",anchor="ms")
        dr.text(((x1+x2)/2,y2+8),f"{f2} N",font=f,fill="#173d4f",anchor="ma")
        for x,t in ((x0,0),(x1,d1),(x2,t2)):
            dr.text((x,399),f"{t} s",font=f,fill="#101822",anchor="ms")
        image=IMG/f"force-{kind}-{i:03}.png"; im.save(image)
        if collides(signatures(fingerprint(image)),blocked):
            image.unlink();continue
        terms=dict(first_force=str(f1),second_force=str(f2),first_end=str(d1),second_end=str(t2),mass=str(mass),initial_speed=str(v0))
        if kind=="multi":
            j1=f1*d1;j2=f2*d2;j=j1+j2; net=v0+j//mass
            q=f"For trolley experiment {exp_id}, what is the final velocity at {t2} seconds, in metres per second? Positive force acts in the positive direction."
            ans=str(net); alt=[str(v0+(j1-j2)//mass),str(j//mass),str(v0+j1//mass)]
            ev=f"The trolley has mass {mass} kg and initial velocity {v0} m/s. Force is +{f1} N from 0 to {d1} s and {f2} N from {d1} to {t2} s."
            rule="Each constant force contributes force times duration to the impulse. Add both signed impulses, divide by mass, and add the initial velocity."
            compact=f"Using the two force intervals with the given mass and starting velocity yields {ans} m/s."
            preserved=f"The impulses are {f1} × {d1} = {j1} N·s and {f2} × ({t2} - {d1}) = {j2} N·s. Their sum is {j} N·s, so the final velocity is {v0} + ({j})/{mass} = {ans} m/s."
            inter=[str(j1),str(j2),str(j)]
        else:
            q=f"In trolley experiment {exp_id}, what force in newtons is printed over the first horizontal interval of the force sensor trace?"
            ans=str(f1);alt=[str(f2),str(-f1),str(mass)]
            ev=f"The first plateau is labeled +{f1} N.";rule="Read the signed height of the first force plateau."
            compact=f"The displayed first force is +{f1} N.";preserved=compact;inter=[]
        opts=options(alt,ans,rng)
        rows.append(record(image.stem,"newly generated classical impulse diagrams","force/"+kind+"/"+str(i),image,q,opts,ans,
                           dict(stratum=kind,schema="impulse" if kind=="multi" else "force_read",evidence=ev,rule=rule,compact=compact,preserved=preserved,terms=terms,intermediate=inter,
                                source_revision="ERA-midstep-r5-192-v1/synthetic")))
        if len(rows)==n:break
    assert len(rows)==n
    return rows


def main():
    IMG.mkdir(exist_ok=True,parents=True)
    if (HERE/"teacher_input_sha256.txt").exists():
        assert sha(HERE/"teacher_inputs.json")== (HERE/"teacher_input_sha256.txt").read_text().strip()
        return
    block_q,blocked=block_scope()
    chart=source_charts(block_q,blocked)
    direct=[r for r in chart if r[1]["stratum"]=="direct"][:48]
    multi=[r for r in chart if r[1]["stratum"]=="multi"][:64]
    science=source_science(block_q,blocked)[:48]
    direct+=science
    chosen=multi+direct+physics(96-len(multi),"multi",blocked)+physics(96-len(direct),"direct",blocked)
    chosen.sort(key=lambda x:key("order/"+x[0]["id"]))
    public=[x[0] for x in chosen]; private=[x[1] for x in chosen]
    assert len(public)==len({r["id"] for r in public})==len({r["image_sha256"] for r in public})==len({r["group"] for r in private})==192
    assert len({normalized(r["question"]) for r in private})==192
    assert all(normalized(r["question"]) not in block_q for r in private)
    assert Counter(r["stratum"] for r in private)=={"direct":96,"multi":96}
    used={r["image"] for r in public}
    for path in IMG.glob("*.png"):
        if path.relative_to(HERE).as_posix() not in used:path.unlink()
    dump(HERE/"teacher_inputs.json",public);dump(HERE/"teacher_gold.json",private)
    receipt=dict(n=192,input_sha256=sha(HERE/"teacher_inputs.json"),gold_sha256=sha(HERE/"teacher_gold.json"),source=dict(Counter(r["source"] for r in private)),
                 schema=dict(Counter(r["schema"] for r in private)),strata=dict(Counter(r["stratum"] for r in private)),eval_near_threshold="exact RGBA, dHash Hamming <=4 or pHash <=6",
                 excluded_signatures=len(blocked),excluded_questions=len(block_q),seed=PREFIX,raw_table_revision=REV,images=192)
    dump(HERE/"teacher_receipt.json",receipt)
    (HERE/"teacher_input_sha256.txt").write_text(receipt["input_sha256"]+"\n",encoding="ascii")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
