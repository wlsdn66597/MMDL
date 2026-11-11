"""Build the R9 paired evaluation from a fixed, local-only source seed."""
from fractions import Fraction
from itertools import permutations
from pathlib import Path
import hashlib
import json
import random

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SEED = 9431327
LETTERS = "ABCD"
WIDTH, HEIGHT = 920, 560
PAPER, INK, MUTED, DIVIDER = "#fafbf7", "#152c36", "#425e67", "#b7c8c5"
COLORS = ["#135b69", "#66365e", "#805024", "#394c8c", "#22704d", "#834144"]
NAMES = "Vela Neri Leto Ossa Nilo Tula Zuri Cavo Heka Sora Ketu Yara Boro Mavi Runa Juno Faro Kira Daro Arvo".split()


def font(size, bold=False):
    paths = ("C:/Windows/Fonts/segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf")
    for filename in paths:
        try:
            return ImageFont.truetype(filename, size)
        except OSError:
            continue
    raise RuntimeError("An installed scalable font is required")


def label(d, x, y, value, size=23, bold=False, color=INK, anchor="lm"):
    d.text((x,y), str(value), font=font(size,bold), fill=color, anchor=anchor)


def canvas(title, subtitle, color):
    img = Image.new("RGB", (WIDTH,HEIGHT), PAPER)
    d = ImageDraw.Draw(img)
    d.rectangle((0,0,14,HEIGHT), fill=color)
    label(d,45,47,title,34,True)
    label(d,46,91,subtitle,21,color=MUTED)
    d.line((46,122,877,122), fill=DIVIDER,width=2)
    return img,d


def table(d, headers, content, xpos, top, row, color):
    d.rectangle((43,top-22,876,top+22), fill=color)
    for x,heading in zip(xpos,headers,strict=True):
        label(d,x,top,heading,21,True,"white")
    for i,fields in enumerate(content):
        y=top+(i+1)*row
        d.line((44,y+row//2,876,y+row//2),fill=DIVIDER)
        for j,(x,value) in enumerate(zip(xpos,fields,strict=True)):
            label(d,x,y,value,24,bold=j==0)


def make_card(question, facts, op, rule, work, result, unit):
    evidence = "; ".join(f"{key} = {value}" for key,value in facts)
    ref=dict(e=evidence,r=rule,a=work,operation=op,relation=op,
             source=[key for key,_ in facts],required_values=[value for _,value in facts],
             result=str(result),unit=unit)
    return dict(question=question,reference=ref,witness=f"{evidence}. {rule} {work}")


def four_choices(correct, distractors):
    out = list(dict.fromkeys(str(v) for v in [*correct,*distractors]))
    if len(out)<4:
        raise ValueError(out)
    return out[:4]


def change(n, rng, color):
    instrument = n%2==1
    family="bench_trace" if instrument else "gate_log"
    dev,measure,unit=("carriage","offset","mm") if instrument else ("test gate","elevation","cm")
    code=f"{rng.choice(NAMES)} {n+20}"
    labels=[f"T{i+n+2}" for i in range(5)] if instrument else [f"{h:02d}:00" for h in range(n+1,n+6)]
    while True:
        v=rng.sample(range(-19,22),5)
        net=v[-1]-v[0]
        steps=[b-a for a,b in zip(v,v[1:])]
        if (net>0)==(n%4 in (1,2)) and min(steps)<0<max(steps) and abs(net)>=3:
            break
    lengths=list(map(abs,steps))
    total=sum(lengths)
    s=dict(title=f"{code} / {'bench trace' if instrument else 'gate monitor'}",labels=labels,values=v,unit=unit,measure=measure)
    img,d=canvas(s["title"],f"Recorded {measure} ({unit}) at five consecutive readings",color)
    if instrument:
        coords=[]
        for i,val in enumerate(v):
            x=132+i*150; y=315-5*val
            coords.append((x,y))
        for val in [-20,-10,0,10,20]:
            y=315-5*val
            d.line((100,y,805,y),fill=DIVIDER,width=2 if val==0 else 1)
            label(d,84,y,val,18,anchor="rm")
        d.line(coords,fill=color,width=4)
        for (x,y),tag,val in zip(coords,labels,v,strict=True):
            d.ellipse((x-5,y-5,x+5,y+5), fill=color)
            d.rounded_rectangle((x-27,y-38,x+27,y-9), radius=4,fill=PAPER)
            label(d,x,y-24,val,23,True,color,"mm")
            label(d,x,463,tag,22,True,anchor="mm")
    else:
        table(d,["Order","Time","Elevation (cm)"],[(i+1,t,vv) for i,(t,vv) in enumerate(zip(labels,v))],
              [70,325,596],155,62,color)
    label(d,46,519,"Time runs from the first reading to the last.",18,color=MUTED)
    first,last=labels[0],labels[-1]
    net_questions=(
        f"How far above or below its initial {measure} was the {dev} at {last}? Use a signed value in {unit}, starting from {first}.",
        f"Record the {dev}'s signed {measure} change, in {unit}, from {first} to {last}.",
        f"The operator needs the final {measure} relative to the reading at {first}. What signed difference in {unit} is recorded at {last}?",
    )
    path_questions=(
        f"How many {unit} did the {dev} travel from {first} to {last} along the four segments shown, ending at {last}?",
        f"What is the combined length, in {unit}, of the four moves shown from {first} through {last}?",
        f"A travel counter starts at zero at {first} and stops at {last}. Counting every segment shown once, how many {unit} does it register?",
    )
    q=n%3
    net_card=make_card(net_questions[q],[(first,v[0]),(last,v[-1])],"endpoint change",
                       "Subtract the initial coordinate from the final coordinate.",
                       f"{v[-1]} - ({v[0]}) = {net} {unit}.",net,unit)
    path_card=make_card(path_questions[q],list(zip(labels,v)),"distance along path",
                        "Take the magnitude of each successive coordinate difference, then add all four.",
                        "The four lengths are "+" + ".join(map(str,lengths))+f" = {total} {unit}.",total,unit)
    return family,s,[net_card,path_card],four_choices([net,total],[max(v)-min(v),sum(lengths[:-1]),-net,abs(net)+1,total+1]),img


def mixture(n,rng,color):
    sector=n%2==1
    family="fiber_tiles" if sector else "alloy_inventory"
    ingredient,subject=("fiber","paper lots") if sector else ("nickel","alloy lots")
    names=rng.sample(NAMES,3)
    while True:
        pct=rng.sample(range(0,101,5),3)
        weights=[10*x for x in rng.sample(range(2,12),3)]
        same=Fraction(sum(pct),3)
        pool=Fraction(sum(w*p for w,p in zip(weights,pct)),sum(weights))
        if same.denominator==pool.denominator==1 and abs(same-pool)>=4:
            break
    same=int(same); pool=int(pool)
    spec=dict(title=f"{rng.choice(NAMES)} / {subject}",labels=names,weights=weights,percent=pct,unit="%",ingredient=ingredient)
    img,d=canvas(spec["title"],f"Lot mass and {ingredient} content by mass",color)
    if sector:
        for i,(name,m,p) in enumerate(zip(names,weights,pct)):
            x=45+283*i
            d.rounded_rectangle((x,173,x+265,459),radius=12,fill="#eef2ec")
            label(d,x+20,209,name,29,True,color)
            label(d,x+20,274,"Mass",20,color=MUTED)
            label(d,x+20,317,f"{m} g",29,True)
            label(d,x+20,372,"Fiber",20,color=MUTED)
            label(d,x+20,416,f"{p}%",29,True)
    else:
        table(d,["Lot","Nickel content","Total mass"],[(l,f"{p}%",f"{w} g") for l,w,p in zip(names,weights,pct)],
              [70,367,656],182,78,color)
    label(d,46,520,"Each lot has a uniform composition.",18,color=MUTED)
    clip=rng.choice([2,3,4,5])
    equal_questions=(
        f"A sample contains {clip} g from each of the three listed lots. What percent of that sample is {ingredient}?",
        f"A test portion mixes the same {clip} g from every listed lot. Find the {ingredient} percentage of that portion.",
        f"Each lot contributes {clip} g to a small specimen. What is the specimen's {ingredient} content as a percent by mass?",
    )
    pool_questions=(
        f"After the three lots are combined in full, what percent of their total mass is {ingredient}?",
        f"The entire inventory shown is poured into one container. What {ingredient} percentage should its label show?",
        f"What percentage of the stock is {ingredient} if all material shown from every lot is used?",
    )
    same_facts=[(f"{k}/percent",p) for k,p in zip(names,pct)]
    pool_facts=[pair for k,w,p in zip(names,weights,pct) for pair in [(f"{k}/mass (g)",w),(f"{k}/percent",p)]]
    work=" + ".join(f"{w}×{p}" for w,p in zip(weights,pct))
    q=n%3
    subject_note=f"The listed lots are {names[0]}, {names[1]} and {names[2]}. "
    equal_card=make_card(subject_note+equal_questions[q],same_facts,"equal contributions",
                         "With equal masses from three uniform lots, average their three mass percentages.",
                         f"({' + '.join(map(str,pct))}) / 3 = {same}%.",same,"%")
    pooled_card=make_card(subject_note+pool_questions[q],pool_facts,"pooled composition",
                          "Weight each lot's percentage by its mass, then divide by the total lot mass.",
                          f"({work}) / {sum(weights)} = {pool}%.",pool,"%")
    return family,spec,[equal_card,pooled_card],four_choices([same,pool],[*pct,100-pool,100-same]),img


def rate(n,rng,color):
    high=n%2==1
    family="inspection_columns" if high else "repair_tallies"
    event,exposure=("Sealed","Checked") if high else ("Reopened","Repaired")
    labels=rng.sample(NAMES,4)
    cmp=max if high else min
    target=((n-1)//2+(0 if high else 2))%4
    rate_target=(target+1+((n-1)//4)%3)%4
    while True:
        dnom=rng.sample(range(41,139),4)
        wins=[rng.randint(8,d-4) for d in dnom]
        ratio=[Fraction(w,d) for w,d in zip(wins,dnom)]
        if len(set(wins))<4 or len(set(ratio))<4:
            continue
        if wins.index(cmp(wins))==target and ratio.index(cmp(ratio))==rate_target:
            order=sorted(ratio,reverse=high)
            if abs(order[0]-order[1])>Fraction(1,30):
                break
    site_name=NAMES[(n*3)%len(NAMES)]
    s=dict(title=f"{site_name} / {'seal inspection' if high else 'repair follow-up'}",
           labels=labels,numerators=wins,denominators=dnom,fields=[event,exposure],direction="max" if high else "min",unit="cases")
    img,d=canvas(s["title"],"Observed cases for the same reporting window",color)
    if high:
        table(d,["Line","Units checked","Units sealed"],list(zip(labels,dnom,wins)),[72,362,655],165,66,color)
    else:
        for i,(name,w,h) in enumerate(zip(labels,wins,dnom)):
            x=45+(i%2)*424;y=149+(i//2)*158
            d.rounded_rectangle((x,y,x+401,y+139),radius=8,fill="#eef2ec")
            label(d,x+17,y+32,name,25,True,color)
            label(d,x+17,y+77,f"Reopened  {w}",22)
            label(d,x+17,y+114,f"Repaired  {h}",22)
    site=f"at {site_name}"
    if high:
        qs1=[f"Which line {site} sealed the greatest number of units in the observed window?",
             f"Which line recorded the highest sealed-unit total {site}?",
             f"For the observed period {site}, which line supplied the most sealed units?"]
        qs2=[f"If every line {site} checked an equal number of units at its observed seal rate, which line would seal the most?",
             f"If inspection volumes are equal {site}, which line has the best sealed-unit yield?",
             f"Using the rates recorded {site}, which line would seal the most per 100 checked?"]
    else:
        qs1=[f"Which team {site} had the fewest reopened cases in this window?",
             f"The supervisor {site} wants the smallest reopened-case total. Which team recorded it?",
             f"On observed reopened-case count alone {site}, which team ranks lowest?"]
        qs2=[f"With an equal number of new repairs per team {site}, which team would have the fewest reopened cases at its observed rate?",
             f"If comparing reopened cases per 100 repairs {site}, which team has the lowest rate?",
             f"For equal repair volumes {site}, which team has the smallest reopened share?"]
    relation="greatest" if high else "smallest"
    facts=[(f"{l}/{event}",w) for l,w in zip(labels,wins)]
    rat_facts=[pair for l,w,h in zip(labels,wins,dnom) for pair in [(f"{l}/{event}",w),(f"{l}/{exposure}",h)]]
    chosen=labels[target]; chosen_rate=labels[rate_target]
    comparisons=", ".join(f"{l} {w}/{h}" for l,w,h in zip(labels,wins,dnom))
    observed=make_card(qs1[n%3],facts,"observed count",f"Select the {relation} observed {event.lower()} count.",
                       f"{chosen} has the {relation} count, {wins[target]}.",chosen,"count")
    normalized=make_card(qs2[n%3],rat_facts,"per-opportunity comparison",f"Compare all four {event.lower()}/{exposure.lower()} fractions and select the {relation}.",
                         f"{comparisons}. The {relation} fraction is {wins[rate_target]}/{dnom[rate_target]} at {chosen_rate}.",chosen_rate,"fraction")
    return family,s,[observed,normalized],labels,img


def route(n,rng,color):
    canal=n%2==1
    family="lock_network" if canal else "signal_hops"
    names=rng.sample(NAMES,4); origin,first,second,end=names
    signs=("lock","vessel","min") if canal else ("relay","signal","ms")
    node,traveler,unit=signs
    while True:
        d=rng.sample(range(5,22),4)
        delays=rng.sample(range(2,16),2)
        raw=[sum(d[:2]),sum(d[2:])]
        via=[raw[i]+delays[i] for i in range(2)]
        if len(set(raw+via))<4 or via.index(min(via))!=int(n%4 in (0,1)):
            continue
        if (raw.index(min(raw))!=via.index(min(via)))!=(n%4 in (2,3)):
            continue
        direct=min(via)+rng.choice([-5,-4,-3] if n%4<2 else [3,4,5])
        if direct>0 and direct not in raw+via:
            break
    edges=[origin+"/"+first,first+"/"+end,origin+"/"+second,second+"/"+end,origin+"/"+end]
    vals=d+[direct]
    title=f"{rng.choice(NAMES)} / {'lock network' if canal else 'signal links'}"
    spec=dict(title=title,labels=names,edges=edges,times=vals,layovers=delays,unit=unit)
    img,drawer=canvas(title,f"Travel time per link ({unit}) and mandatory service at an intermediate {node}",color)
    if canal:
        centers=[(115,286),(442,177),(442,386),(775,286)]
        seq=[(0,1),(1,3),(0,2),(2,3),(0,3)]
        for (i,j),time in zip(seq,vals):
            x1,y1=centers[i];x2,y2=centers[j]
            drawer.line((x1,y1,x2,y2),fill=color,width=3)
            xx=(x1+x2)//2; yy=(y1+y2)//2
            drawer.rounded_rectangle((xx-25,yy-16,xx+25,yy+16),radius=4,fill=PAPER)
            label(drawer,xx,yy,time,23,True,color,"mm")
        for (x,y),name in zip(centers,names):
            drawer.rounded_rectangle((x-55,y-23,x+55,y+23),radius=8,fill="#e1ebe6",outline=color,width=2)
            label(drawer,x,y,name,23,True,anchor="mm")
        label(drawer,48,480,f"Lock service     {first}   {delays[0]} min        /        {second}   {delays[1]} min",23,True)
    else:
        table(drawer,["Link","Travel time"],[(e.replace("/","  →  "),f"{v} ms") for e,v in zip(edges,vals)],[72,393],162,54,color)
        drawer.rounded_rectangle((582,204,851,370),radius=8,fill="#e4ece7")
        label(drawer,600,238,"Relay processing",20,True,color)
        for i,(name,delay) in enumerate(zip(names[1:3],delays)):
            label(drawer,600,287+i*46,f"{name}   {delay} ms",20,True)
    direct_q=[f"The {traveler} goes from {origin} to {end} on the single link between them. How many {unit} elapse?",
              f"How long in {unit} is the uninterrupted direct link from {origin} to {end}?",
              f"A direct transfer from {origin} to {end} bypasses both intermediate sites. What is its duration in {unit}?"]
    via_q=[f"The {traveler} must visit exactly one intermediate {node} from {origin} to {end}, including its full service time. What is the shortest elapsed time in {unit}?",
           f"From {origin} to {end}, the {traveler} must stop once and incur the service shown at that site. What is the minimum total in {unit}?",
           f"The direct link is unavailable from {origin} to {end}. Of the two allowed one-stop journeys, including the mandatory intermediate service, how many {unit} does the faster take?"]
    one=make_card(direct_q[n%3],[(edges[-1],direct)],"no intermediate stop","Use the link directly between the endpoints.",
                  f"{origin} to {end} on the direct link takes {direct} {unit}.",direct,unit)
    facts=list(zip(edges[:4],d))+[(first+"/service",delays[0]),(second+"/service",delays[1])]
    best=via.index(min(via))
    two=make_card(via_q[n%3],facts,"minimum one-stop elapsed time","Compare both feasible two-link routes, including the mandatory service of their own intermediate site.",
                  f"Via {first}: {d[0]}+{delays[0]}+{d[1]}={via[0]}. Via {second}: {d[2]}+{delays[1]}+{d[3]}={via[1]}. Via {names[best+1]} is fastest at {via[best]} {unit}.",
                  via[best],unit)
    return family,spec,[one,two],four_choices([direct,via[best]],[max(via),min(raw),max(raw)]),img


def write_rows(name,rows):
    (ROOT/name).write_bytes("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode())


def main():
    (ROOT/"images").mkdir(parents=True,exist_ok=True)
    cards=[]; scenes=[]
    for kind_index,(kind,builder) in enumerate([("change",change),("mixture",mixture),("rate",rate),("route",route)]):
        slots=list(permutations(range(4),2))
        random.Random(SEED+kind_index).shuffle(slots)
        for n,(left,right) in enumerate(slots,1):
            rng=random.Random(SEED+kind_index*10_000+n*53)
            group=f"ir9-{kind}-{n:02d}"
            family,spec,pair,options,img=builder(n,rng,COLORS[(kind_index+n)%len(COLORS)])
            if n%4 in (0,1):
                pair.reverse()
            gold=[q["reference"]["result"] for q in pair]
            distractors=[v for v in options if v not in gold]; rng.shuffle(distractors)
            assert len(set(options))==4 and len(set(gold))==2 and len(distractors)==2
            other=iter(distractors)
            order=[gold[0] if i==left else gold[1] if i==right else next(other) for i in range(4)]
            file=ROOT/"images"/f"{group}.png"
            img.save(file)
            scenes.append(dict(group=group,kind=kind,family=family,spec=spec,rgb_sha256=hashlib.sha256(img.tobytes()).hexdigest()))
            for j,(answer,card) in enumerate(zip([left,right],pair)):
                cards.append(dict(id=f"{group}-{j+1}",group=group,kind=kind,family=family,image=f"paired/images/{file.name}",options=order,gold=LETTERS[answer],**card))
    write_rows("items.jsonl",cards);write_rows("sources.jsonl",scenes)


if __name__=="__main__":
    main()
