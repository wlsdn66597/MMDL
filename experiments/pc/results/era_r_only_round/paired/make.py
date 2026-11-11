"""Generate the R8 independent rule-choice pairs from the fixed source seed."""
from fractions import Fraction
from pathlib import Path
import hashlib
import itertools
import json
import random

from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
SEED=835117
LETTERS="ABCD"
KINDS=("change","rate","route","mixture")
NAMES=("Cairn","Zephyr","Harbor","Mica","Amber","Flint","Gable","Robin","Fawn","Aster","Coral","Vista","Morrow","Finch","Echo","Pine","Quill","Elm","Opal","Sable","Acorn","Iris","Swift","Cedar")
PALETTES=(("#0c5276","#d68532"),("#007661","#95406a"),("#573996","#b9681d"),("#8b453b","#31798c"),("#3e6383","#99702c"),("#2b6a71","#994a47"))
BG="#f8f7f0"; INK="#172b39"; MUTED="#53616c"; W=860; H=540


def font(size,bold=False):
    choices=("C:/Windows/Fonts/arialbd.ttf","DejaVuSans-Bold.ttf") if bold else ("C:/Windows/Fonts/arial.ttf","DejaVuSans.ttf")
    for f in choices:
        try:return ImageFont.truetype(f,size)
        except OSError:continue
    raise RuntimeError("Audit image requires a scalable font")


def line(d,xy,text,size=22,bold=False,color=INK,anchor="lm"):
    d.text(xy,str(text),font=font(size,bold),fill=color,anchor=anchor)


def num(x):
    if isinstance(x,Fraction):
        assert x.denominator==1
        return str(x.numerator)
    return str(x)


def ask(question,keys,values,op,result,rule,work,unit):
    facts="; ".join(f"{k} = {v}" for k,v in zip(keys,values,strict=True))
    return dict(question=question,reference=dict(e=facts,r=rule,a=work,operation=op,relation=op,source=keys,required_values=values,result=num(result),unit=unit),
                witness=f"{facts}. {rule} {work}")


def differing(options,answers):
    unique=list(dict.fromkeys(map(num,[*answers,*options])))
    j=1
    while len(unique)<4:
        v=str(abs(int(answers[1]))+j);j+=1
        if v not in unique:unique.append(v)
    return unique[:4]


def change(n,rng):
    layout="station_profile" if n%2 else "checkpoint_cards"
    ticks=rng.sample([["Dawn","Noon","Dusk","Close"],["Alpha","Bravo","Delta","Sigma"],["East","Gate","Pier","Bay"],["E1","E2","E3","E4"]],1)[0]
    start=rng.randint(-9,7)
    while True:
        v=[start,start+rng.randint(2,11),start-rng.randint(3,9),start + (1 if n%4 in (0,1) else -1)*rng.randint(2,7)]
        net=v[-1]-v[0]; dist=sum(abs(y-x) for x,y in zip(v,v[1:]))
        if len(set(v))==4:break
    site=rng.choice(NAMES); unit="levels" if layout=="station_profile" else "m"
    spec=dict(title=site+(" lift log" if layout=="station_profile" else " sled record"),labels=ticks,values=v,unit=unit)
    q1=(f"At {ticks[-1]}, how many {unit} higher is the {'lift' if layout=='station_profile' else 'sled'} than at {ticks[0]}? Record a negative answer if it is lower."
        if n%3 else f"A supervisor cares only about where the trip finished relative to {ticks[0]}. What signed change in {unit} would they record at {ticks[-1]}?")
    q2=(f"A wheel counter follows every leg from {ticks[0]} through {ticks[-1]}. What distance would its counter register in {unit}?"
        if n%3 else f"Between {ticks[0]} and {ticks[-1]}, how many {unit} of movement must the mechanism withstand, including its return trips?")
    steps=[abs(y-x) for x,y in zip(v,v[1:])]
    qs=[ask(q1,[ticks[0],ticks[-1]],[v[0],v[-1]],"endpoint change",net,"Compare arrival with departure, preserving the direction.",f"{v[-1]} - ({v[0]}) = {net} {unit}.",unit),
        ask(q2,ticks,v,"distance along path",dist,"Use each consecutive pair and sum the positive segment lengths.",f"{' + '.join(map(str,steps))} = {dist} {unit}.",unit)]
    return layout,spec,qs,differing([max(v)-min(v),sum(steps[:2]),abs(net),v[-1]],(net,dist))


def rate(n,rng):
    layout="germination_slips" if n%2 else "invitation_register"
    noun="greenhouse" if n%2 else "chapter"
    sent,arrived=("Seeds sown","Seedlings") if n%2 else ("Invites sent","Attendees")
    names=rng.sample(NAMES,4)
    while True:
        denomin=rng.sample(range(36,140),4);numer=[rng.randint(12,d-5) for d in denomin]
        ratios=[Fraction(x,y) for x,y in zip(numer,denomin)]
        if len(set(numer))==4 and len(set(ratios))==4 and numer.index(max(numer))!=ratios.index(max(ratios)) and sorted(ratios)[-1]-sorted(ratios)[-2]>Fraction(1,30):break
    spec=dict(title="Germination record" if n%2 else "Invitation responses",labels=names,fields=[sent,arrived],denominators=denomin,numerators=numer,unit="seeds" if n%2 else "people")
    cq=(f"Which {noun} produced the most seedlings in this record?" if n%2 else "Which chapter supplied the most attendees to this event?")
    rq=("If each greenhouse sowed the same number of new seeds, which one would produce the most seedlings if its observed performance held?" if n%2 else
        "If each chapter sent the same number of new invitations, which chapter's current response pattern would bring in the most people?")
    ck=[s+"/"+arrived for s in names]
    rk=[v for s in names for v in (s+"/"+arrived,s+"/"+sent)]
    vals=[v for pair in zip(numer,denomin) for v in pair]
    winner1=names[numer.index(max(numer))];winner2=names[ratios.index(max(ratios))]
    comparisons=", ".join(f"{s}: {x}/{y} ({float(x/y)*100:.1f}%)" for s,x,y in zip(names,numer,denomin))
    qs=[ask(cq,ck,numer,"observed count",winner1,"Compare the observed outcome counts across all sites.",f"{max(numer)} is largest at {winner1}.","count"),
        ask(rq,rk,vals,"per-opportunity comparison",winner2,"At equal future exposure, compare each site's outcomes per opportunity in its own row.",f"{comparisons}; {winner2} has the largest fraction.","fraction")]
    return layout,spec,qs,names


def route(n,rng):
    layout="rail_transfer" if n%2 else "trail_checkpoint"
    node=rng.sample(NAMES,4);u,t,b,v=node
    while True:
        times=rng.sample(range(3,15),4);wait=rng.sample(range(2,10),2)
        legs=[times[0]+times[1],times[2]+times[3]]
        dwell=[l+w for l,w in zip(legs,wait)]
        if len(set(dwell))<2 or len({*legs,*dwell})<4:continue
        direct=max(dwell)+rng.randint(1,6) if n%4<2 else min(dwell)-rng.randint(1,3)
        if direct not in {*legs,*dwell} and direct>3:break
    edges=[u+"/"+t,t+"/"+v,u+"/"+b,b+"/"+v,u+"/"+v]
    times=times+[direct]
    spec=dict(title="Transfer timetable" if n%2 else "Trail timing card",labels=node,edges=edges,times=times,layovers=wait,unit="min")
    cq=(f"A parcel must go from {u} to {v} without an intermediate handoff. What time in minutes should be booked?" if n%2 else
        f"A walker from {u} to {v} must avoid both middle checkpoints. How many minutes will the walk take?")
    rq=(f"The parcel must change trains exactly once on the way from {u} to {v}. What is its earliest possible arrival time in minutes from departure?"
        if n%2 else f"A walker from {u} to {v} must visit one middle checkpoint and stay for the posted inspection time. What is the earliest finish in minutes?")
    keys=edges[:4]+[t+"/stop",b+"/stop"];w=min(dwell)
    qs=[ask(cq,[edges[-1]],[direct],"no intermediate stop",direct,"Read the non-stop link joining the named endpoints.",f"{u}→{v} = {direct} min.","min"),
        ask(rq,keys,times[:4]+wait,"minimum one-stop elapsed time",w,"For each allowed stop, add both legs and its posted stop time, then take the shorter feasible elapsed time.",f"Via {t}: {times[0]}+{wait[0]}+{times[1]}={dwell[0]}; via {b}: {times[2]}+{wait[1]}+{times[3]}={dwell[1]}; earliest {w} min.","min")]
    return layout,spec,qs,differing([min(legs),max(dwell),max(legs)],(direct,w))


def mixture(n,rng):
    layout="batch_samples" if n%2 else "shift_loads"
    nx,ny=rng.sample(NAMES,2)
    while True:
        p1,p2=rng.sample([0,10,20,30,40,50,60,70,80,90,100],2)
        a,b=rng.sample([10,20,30,40,50,60,70,80,90],2)
        equal=Fraction(p1+p2,2);pooled=Fraction(a*p1+b*p2,a+b)
        if pooled.denominator==equal.denominator==1 and abs(equal-pooled)>=4:break
    spec=dict(title="Batch composition" if n%2 else "Shift quality record",labels=[nx,ny],weights=[a,b],percent=[p1,p2],unit="%")
    if n%2:
        spec.update(weight_label="Batch volume (L)",percent_label="Dye by volume (%)")
        cq=f"A chemist takes one litre from {nx} and one litre from {ny}, then blends the samples. What percent of the blend is dye?"
        rq=f"The full batches from {nx} and {ny} are poured together. What percent of all the liquid is dye?"
    else:
        spec.update(weight_label="Parts checked",percent_label="Passing (%)")
        cq=f"A report gives {nx} and {ny} equal influence on a headline passing percentage. What percentage will it print?"
        rq=f"A second report describes every part checked across {nx} and {ny} as a single inspection lot. What percentage of those parts passed?"
    keys=[nx+"/percent",ny+"/percent"]
    qs=[ask(cq,keys,[p1,p2],"equal contributions",equal,"Equal sample volumes or equal reporting weights give the two percentages the same weight.",f"({p1}+{p2})/2={num(equal)}%.","%"),
        ask(rq,[nx+"/size",nx+"/percent",ny+"/size",ny+"/percent"],[a,p1,b,p2],"pooled composition",pooled,"Weight each percentage by its source quantity and divide by the combined quantity.",f"({a}×{p1}+{b}×{p2})/({a}+{b})={num(pooled)}%.","%")]
    return layout,spec,qs,differing([p1,p2,100-int(pooled),100-int(equal)],(equal,pooled))


GENERATORS=dict(change=change,rate=rate,route=route,mixture=mixture)


def heading(d,title,subtitle,col):
    d.rectangle((0,0,W,7),fill=col)
    line(d,(37,42),title,34,True)
    line(d,(39,83),subtitle,19,color=MUTED)


def graph(d,s,layout,col,other):
    vals=s["values"];labs=s["labels"]
    xy=[(88+i*211,int(330-v*8)) for i,v in enumerate(vals)]
    if layout=="station_profile":
        heading(d,s["title"],"Position on arrival at each call  ·  levels relative to lobby",col)
        for n in range(-20,26,5):
            y=330-n*8;d.line((56,y,808,y),fill="#b7c3c4",width=1)
            line(d,(48,y),n,14,anchor="rm")
        d.line(xy,fill=col,width=5)
        for (x,y),lab,val in zip(xy,labs,vals):
            d.ellipse((x-7,y-7,x+7,y+7),fill=col);line(d,(x,y-22),val,23,True,col,"mm")
            line(d,(x,500),lab,19,True,anchor="mm")
    else:
        heading(d,s["title"],"Position marker at successive checks  ·  metres relative to base",col)
        for i,(lab,val) in enumerate(zip(labs,vals)):
            x=30+i*208
            d.rounded_rectangle((x,199,x+177,382),radius=14,fill="#fffefe",outline=col,width=2)
            line(d,(x+88,235),lab,23,True,anchor="mm");line(d,(x+88,309),str(val)+" m",33,True,col,"mm")
            if i<3:line(d,(x+192,292),"›",29,True,other,"mm")
        line(d,(35,449),"Checks ordered from departure to arrival",18,color=MUTED)


def count_scene(d,s,layout,col,other):
    labels=s["labels"];f=s["fields"]
    heading(d,s["title"],"Measured totals for the same reporting period",col)
    if layout=="germination_slips":
        for i,(name,n,m) in enumerate(zip(labels,s["denominators"],s["numerators"])):
            x=37+(i%2)*405;y=135+(i//2)*184
            d.rounded_rectangle((x,y,x+376,y+158),12,fill="#fffefe",outline=col,width=2)
            line(d,(x+15,y+28),name,25,True,col)
            line(d,(x+15,y+78),f[0],20);line(d,(x+344,y+78),n,24,True,anchor="rm")
            line(d,(x+15,y+120),f[1],20);line(d,(x+344,y+120),m,24,True,anchor="rm")
    else:
        for i,h in enumerate(["Chapter",f[1],f[0]]):line(d,(75+i*265,156),h,22,True,col)
        for i,(name,n,m) in enumerate(zip(labels,s["denominators"],s["numerators"])):
            y=222+i*68
            d.line((57,y-28,801,y-28),fill="#bac7cc",width=1)
            for j,val in enumerate([name,m,n]):line(d,(75+j*265,y),val,25,bold=j==0)


def transit(d,s,layout,col,other):
    u,t,b,v=s["labels"];vals=s["times"];w=s["layovers"]
    heading(d,s["title"],"Travel and required stop times, in minutes  ·  west to east",col)
    centers=[(89,300),(417,150),(417,410),(760,300)]
    seq=[(0,1),(1,3),(0,2),(2,3),(0,3)]
    if layout=="rail_transfer":
        for (a,b2),numv in zip(seq,vals):
            ax,ay=centers[a];bx,by=centers[b2];d.line((ax,ay,bx,by),fill=col,width=3)
            x=(ax+bx)//2;y=(ay+by)//2
            d.rounded_rectangle((x-21,y-14,x+21,y+14),5,fill=BG)
            line(d,(x,y),numv,21,True,col,"mm")
        for (x,y),name in zip(centers,s["labels"]):
            d.rounded_rectangle((x-50,y-19,x+50,y+19),10,fill="#e0ebe9",outline=col)
            line(d,(x,y),name,19,True,anchor="mm")
        line(d,(35,494),f"Change time at {t}: {w[0]} min     ·     at {b}: {w[1]} min",20,True,other)
    else:
        line(d,(41,141),"Trail section",22,True,col);line(d,(423,141),"Walk (min)",22,True,col)
        for i,((l,r),time) in enumerate(zip(seq,vals)):
            y=197+i*52;d.line((35,y-22,557,y-22),fill="#b1c1c5",width=1)
            line(d,(42,y),f"{s['labels'][l]}   →   {s['labels'][r]}",21)
            line(d,(466,y),time,22,True,anchor="mm")
        d.rounded_rectangle((577,152,825,401),12,fill="#e8ece5",outline=other)
        line(d,(597,183),"Inspection stop",22,True,other)
        for i,(name,numv) in enumerate(zip((t,b),w)):
            line(d,(601,244+i*83),name,21,True);line(d,(804,244+i*83),str(numv)+" min",19,anchor="rm")


def blend(d,s,layout,col,other):
    heading(d,s["title"],"Individually measured lots",col)
    for i,(name,size,perc) in enumerate(zip(s["labels"],s["weights"],s["percent"])):
        x=52+i*399;y=126
        d.rounded_rectangle((x,y,x+357,470),16,fill="#fffdf8",outline=col,width=2)
        line(d,(x+20,169),name,29,True,col)
        line(d,(x+20,224),s["weight_label"],19)
        line(d,(x+20,264),size,32,True)
        line(d,(x+20,330),s["percent_label"],19)
        line(d,(x+20,376),str(perc)+"%",32,True,other)
        d.rectangle((x+20,418,x+325,433),fill="#d1dbd9")
        if perc:d.rectangle((x+20,418,x+20+305*perc//100,433),fill=other)


def save_jsonl(path,items):
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in items),encoding="utf8")


def main():
    (HERE/"images").mkdir(exist_ok=True,parents=True)
    cards=[]; scenes=[]
    for ix,kind in enumerate(KINDS):
        transitions=list(itertools.permutations(range(4),2));random.Random(SEED+ix*101).shuffle(transitions)
        for zero,(a,b) in enumerate(transitions):
            n=zero+1;group=f"ir8-{kind}-{n:02d}"
            rng=random.Random(SEED+ix*3130+n*73)
            layout,s,q,opts=GENERATORS[kind](n,rng)
            answers=[card["reference"]["result"] for card in q]
            assert len(set(opts))==4 and len(set(answers))==2 and set(answers)<=set(opts)
            # Counterbalance which rule occurs first as well as each answer letter.
            if n%4 in (0,3):q.reverse();answers.reverse()
            opts=[v for v in opts if v not in answers];rng.shuffle(opts);remaining=iter(opts)
            order=[answers[0] if j==a else answers[1] if j==b else next(remaining) for j in range(4)]
            im=Image.new("RGB",(W,H),BG);d=ImageDraw.Draw(im)
            col,other=rng.choice(PALETTES)
            render={"change":graph,"rate":count_scene,"route":transit,"mixture":blend}[kind]
            render(d,s,layout,col,other)
            pic=HERE/"images"/(group+".png");im.save(pic)
            scenes.append(dict(group=group,kind=kind,family=layout,spec=s,rgb_sha256=hashlib.sha256(im.tobytes()).hexdigest()))
            for j,card in enumerate(q):
                cards.append(dict(id=group+f"-{j+1}",group=group,kind=kind,family=layout,image="paired/images/"+pic.name,
                    options=order,gold=LETTERS[[a,b][j]],**card))
    save_jsonl(HERE/"items.jsonl",cards);save_jsonl(HERE/"sources.jsonl",scenes)


if __name__=="__main__":main()
