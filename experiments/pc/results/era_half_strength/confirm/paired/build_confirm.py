"""Produce disjoint confirmatory charts using the same operation families as the diagnostic."""
from itertools import permutations
import hashlib
import importlib.util
import json
from pathlib import Path
import random

from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[3]
diag=PROJECT/"results/era_a_only_round/paired"
spec=importlib.util.spec_from_file_location("diagnostic_source",diag/"make.py")
r9=importlib.util.module_from_spec(spec);spec.loader.exec_module(r9)
r9.NAMES="Alder Brant Coral Drift Ember Firth Glen Hazel Iris Jade Kelp Larch Moss North Onyx Pearl Quill Ridge Sage Tide Umber Vale Willow Yew".split()
SEED=67012571
INK="#192638"; PAPER="#fffdf5"; MUTED="#49576a"; RULE="#5d6874"
PALE=["#d7eaf5","#f2e0d1","#e2ead1","#e4dff1"]


def text(d,xy,x,size=22,bold=False,fill=INK,anchor="lm"):
    d.text(xy,str(x),font=r9.font(size,bold),fill=fill,anchor=anchor)


def header(name,subtitle):
    img=Image.new("RGB",(920,560),PAPER);d=ImageDraw.Draw(img)
    d.rectangle((0,0,920,116),fill=INK)
    text(d,(36,38),name,33,True,"white")
    text(d,(37,84),subtitle,20,False,"#e1e8f1")
    return img,d


def render_change(s,n):
    labels=s["labels"]; vals=s["values"];unit=s["unit"]
    img,d=header(s["title"],f"Travel log  /  signed {s['measure']} ({unit})")
    if n%2:
        x=[112+170*i for i in range(5)];y=[320-5*v for v in vals]
        for v in (-20,-10,0,10,20):
            yy=320-v*5;d.line((82,yy,850,yy),fill=RULE,width=2 if v==0 else 1)
            text(d,(65,yy),v,18,anchor="rm")
        d.line(list(zip(x,y)),fill="#994328",width=5)
        for xx,yy,l,v in zip(x,y,labels,vals):
            d.ellipse((xx-6,yy-6,xx+6,yy+6),fill=INK)
            d.rectangle((xx-24,yy-38,xx+24,yy-9),fill=PAPER)
            text(d,(xx,yy-25),v,22,True,anchor="mm");text(d,(xx,474),l,23,True,anchor="mm")
    else:
        text(d,(50,158),"Observation sequence",21,True)
        for j,(l,v) in enumerate(zip(labels,vals)):
            x=45+j*168
            d.rounded_rectangle((x,205,x+153,416),radius=12,fill=PALE[j%4])
            text(d,(x+77,254),l,23,False,anchor="mm")
            text(d,(x+77,345),v,31,True,anchor="mm")
    text(d,(45,525),"First observation at left; final observation at right.",18,fill=MUTED)
    return img


def render_mixture(s,n):
    labels=s["labels"];w=s["weights"];p=s["percent"];ingredient=s["ingredient"]
    img,d=header(s["title"],f"Mass of each lot and {ingredient} percentage by mass")
    if n%2:
        text(d,(38,152),"Lot and mass",20,True);text(d,(330,152),f"{ingredient.title()} content (0 to 100%)",20,True)
        for j,(l,weight,pct) in enumerate(zip(labels,w,p)):
            y=213+j*102
            text(d,(44,y),f"{l}   {weight} g",24,True)
            d.rectangle((330,y-15,721,y+15),outline=RULE,width=2)
            if pct: d.rectangle((331,y-14,331+int(3.86*pct),y+14),fill=["#456f86","#965a31","#4f7152"][j])
            text(d,(815,y),f"{pct}%",24,True,anchor="rm")
    else:
        for j,(l,weight,pct) in enumerate(zip(labels,w,p)):
            x=49+j*283
            d.rounded_rectangle((x,164,x+251,467),radius=16,fill=PALE[j])
            text(d,(x+22,211),l,27,True)
            text(d,(x+22,281),"Mass",19);text(d,(x+22,320),f"{weight} g",25,True)
            text(d,(x+22,376),ingredient.title(),19);text(d,(x+22,416),f"{pct}%",26,True)
    text(d,(45,528),"Composition is uniform within each lot.",18,fill=MUTED)
    return img


def render_rate(s,n):
    im,d=header(s["title"],"Observed counts for one reporting period")
    labs=s["labels"];num=s["numerators"];den=s["denominators"];event,exposure=s["fields"]
    if n%2:
        for y,c,vv in [(253,exposure,den),(382,event,num)]:
            text(d,(26,y),c,23,True)
            for j,(l,v) in enumerate(zip(labs,vv)):
                x=172+j*180
                d.rectangle((x,y-43,x+151,y+43),fill=PALE[j])
                text(d,(x+75,y),v,27,True,anchor="mm")
                if y==253:text(d,(x+75,173),l,23,True,anchor="mm")
    else:
        for j,(l,e,h) in enumerate(zip(labs,num,den)):
            y=175+83*j;d.rounded_rectangle((28,y-29,882,y+29),radius=13,fill=PALE[(j+1)%4])
            text(d,(50,y),l,24,True);text(d,(323,y),f"{e} {event.lower()}",22)
            text(d,(602,y),f"{h} {exposure.lower()}",22)
    return im


def render_route(s,n):
    origin,a,b,end=s["labels"]; v=s["times"];sa,sb=s["layovers"];unit=s["unit"]
    im,d=header(s["title"],f"Leg duration and intermediate-site service  /  {unit}")
    if n%2:
        for j,(mid,ab,be,service) in enumerate([(a,v[0],v[1],sa),(b,v[2],v[3],sb)]):
            y=185+j*112;d.rounded_rectangle((35,y-38,884,y+38),radius=7,fill=PALE[j])
            text(d,(50,y),origin,23,True);text(d,(202,y),f"{ab} {unit}  →",21)
            text(d,(395,y),mid,24,True);text(d,(554,y),f"{be} {unit}  →",21);text(d,(761,y),end,24,True)
            text(d,(402,y+49),f"Service at {mid}: {service} {unit}",17,fill=MUTED)
        d.line((38,413,881,413),fill=RULE,width=2)
        text(d,(52,473),f"Direct: {origin}  →  {end}     {v[4]} {unit}",27,True)
    else:
        text(d,(42,154),"Link",20,True);text(d,(372,154),"Transit",20,True);text(d,(628,154),"Site service",20,True)
        for j,(edge,t) in enumerate(zip(s["edges"],v)):
            y=211+j*65;d.rounded_rectangle((35,y-23,472,y+25),radius=5,fill=PALE[j%4])
            text(d,(54,y),edge.replace("/","   →   "),22)
            text(d,(390,y),f"{t} {unit}",22,True)
        for j,(name,t) in enumerate([(a,sa),(b,sb)]):
            y=232+j*100;d.rectangle((560,y-37,869,y+37),fill=PALE[(j+2)%4])
            text(d,(585,y),f"{name}    {t} {unit}",23,True)
    return im


def questions(k,s,pair,n):
    labs=s["labels"];p0,p1=pair
    if k=="change":
        first,last=labs[0],labs[-1];u=s["unit"]
        p0["question"]=(f"For {s['title']}, report the signed difference in {u} between the {first} reading and the {last} reading, final minus initial.")
        p1["question"]=(f"From {first} to {last} in {s['title']}, what distance in {u} is accumulated across the four segments, including each reversal?")
    if k=="mixture":
        intro=f"At {s['title']}, ";x=s["ingredient"]
        import re
        clip=re.search(r"\b(\d+) g",p0["question"]).group(1)
        p0["question"]=intro+f"a specimen uses exactly {clip} g of each lot. What percent of this three-lot specimen is {x}?"
        p1["question"]=intro+f"all material is blended. What percent by mass is {x} in the finished batch?"
    if k=="rate":
        event,exposure=s["fields"]; high=s["direction"]=="max";name=s["title"]
        word="greatest" if high else "lowest"
        p0["question"]=f"In {name}, which site recorded the {word} number of {event.lower()} cases?"
        p1["question"]=f"In {name}, which site has the {word} {event.lower()} rate per 100 {exposure.lower()} cases?"
    if k=="route":
        origin,_,_,end=labs;u=s["unit"];title=s["title"]
        p0["question"]=f"On {title}, how many {u} does the single link connecting {origin} to {end} require?"
        p1["question"]=f"On {title}, a trip from {origin} to {end} uses exactly one intermediate site and pays its service cost. What is the shortest elapsed time in {u}?"


def main():
    scenes=[];cards=[];(HERE/"images").mkdir(exist_ok=True)
    for i,(kind,generate,render) in enumerate([("change",r9.change,render_change),("mixture",r9.mixture,render_mixture),("rate",r9.rate,render_rate),("route",r9.route,render_route)]):
        slots=list(permutations(range(4),2));random.Random(SEED+i).shuffle(slots)
        for n,(x,y) in enumerate(slots,1):
            rng=random.Random(SEED+17_000*i+173*n)
            family,s,pair,options,_=generate(n,rng,r9.COLORS[(i+n)%6])
            s["title"]=f"{r9.NAMES[n]} {20+n} / "+{"change":"motion record","mixture":"batch inventory","rate":"field report","route":"transit chart"}[kind]
            questions(kind,s,pair,n)
            im=render(s,n)
            group=f"hc-{kind}-{n:02}";file=HERE/"images"/(group+".png")
            if n%4 in (2,3):pair.reverse()
            gold=[r["reference"]["result"] for r in pair]
            other=[z for z in options if z not in gold];rng.shuffle(other);it=iter(other)
            order=[gold[0] if j==x else gold[1] if j==y else next(it) for j in range(4)]
            im.save(file)
            scenes.append(dict(group=group,kind=kind,family=family,spec=s,rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
            for j,(letter,card) in enumerate(zip((x,y),pair),1):
                cards.append(dict(id=f"{group}-{j}",group=group,kind=kind,family=family,image="paired/images/"+file.name,options=order,gold="ABCD"[letter],**card))
    for f,rows in (("sources.jsonl",scenes),("items.jsonl",cards)):
        raw="".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows).encode()
        out=HERE/f
        if out.exists():assert out.read_bytes()==raw,out
        else:out.write_bytes(raw)
    oracle=importlib.util.spec_from_file_location("independent_oracle",diag/"verify.py")
    verify=importlib.util.module_from_spec(oracle);oracle.loader.exec_module(verify)
    verify.ROOT=HERE
    # The same scorer is used, including the independently derived fraction and path checks.
    for row in cards:
        srow=next(z for z in scenes if z["group"]==row["group"])
        op,value=verify.solve(row["kind"],srow["spec"],row["question"])
        assert op==row["reference"]["operation"] and row["reference"]["result"]==value==row["options"]["ABCD".index(row["gold"])]
    assert len(set(c["question"] for c in cards))==96
    assert len({g["rgb_sha256"] for g in scenes})==48
    for k in ("change","mixture","rate","route"):
        gs=[g for g in scenes if g["kind"]==k];assert len(gs)==12
        transitions={tuple(row["gold"] for row in cards if row["group"]==g["group"]) for g in gs}
        assert transitions==set(permutations("ABCD",2))
    verify.contact_sheets(scenes)
    receipt=dict(groups=48,n=96,n_per_kind=12,independent_oracle_correct=96,
                 inputs_sha256=verify.digest(HERE/"items.jsonl"),sources_sha256=verify.digest(HERE/"sources.jsonl"),
                 generator_sha256=verify.digest(Path(__file__)),reference_source_generator=verify.digest(diag/"make.py"),independent_oracle_sha256=verify.digest(diag/"verify.py"))
    (HERE/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    print(receipt)


if __name__=="__main__":main()
