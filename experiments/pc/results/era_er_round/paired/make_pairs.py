"""Create the independent chart-pair set from a seed reserved for development."""
import hashlib
import json
import math
import random
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SEED = 714963
KINDS = ("E", "R", "A", "choice")
FAMILIES = ("ledger","trace","route","bars")
LETTERS = "ABCD"
ROWS = ("Alder","Beck","Cove","Dale","Elk","Firth","Glen","Heath","Inlet","Junco","Kite","Lark","Marsh","Nook","Pike","Rill","Sage","Tern","Vail","Wick","Yew","Zinnia")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle(op, vals, labels):
    if op == "read": return str(vals[0])
    if op == "sum": return str(sum(vals))
    if op == "delta": return str(vals[-1]-vals[0])
    if op == "travel": return str(sum(abs(b-a) for a,b in zip(vals,vals[1:])))
    if op == "gap": return str(abs(vals[0]-vals[1]))
    if op == "larger":
        assert vals[0] != vals[1]
        return labels[int(vals[1]>vals[0])]
    if op == "larger_rate":
        a,b=Fraction(vals[0],vals[1]),Fraction(vals[2],vals[3])
        assert a != b
        return labels[int(b>a)]
    if op == "combined_pending": return str(vals[0]-vals[1]+vals[2]-vals[3])
    if op == "shorter_route":
        assert vals[0]+vals[1]!=vals[2]+vals[3]
        return labels[int(vals[0]+vals[1]>vals[2]+vals[3])]
    raise ValueError(op)


def make_spec(family,rng):
    labels=rng.sample(ROWS,4)
    if family=="ledger":
        done=rng.sample(range(22,68),4)
        requested=[v+rng.randint(3,42) for v in done]
        data={f"{label}/{col}":v for label,x,y in zip(labels,done,requested) for col,v in (("Closed",x),("Received",y))}
        return dict(labels=labels,data=data,title="Work order register",unit="orders",
                    fields=rng.sample(["Received","Closed"],2))
    if family=="trace":
        traces=rng.sample(["Reef","Mesa","Drum","Vale","Opal","Lake","Pond","Reed","Oak"],2)
        stations=list("JKLM") if rng.randrange(2) else list("PQRS")
        data={f"{s}/{t}":v for s in traces for t,v in zip(stations,rng.sample(range(-12,22),4))}
        return dict(labels=traces,stations=stations,data=data,title="Probe readings by check",unit="units")
    if family=="route":
        x,y,z,w=labels
        edges=[(x,y),(y,w),(x,z),(z,w),(x,w)]
        vals=rng.sample(range(2,22),5)
        return dict(labels=labels,edges=edges,data={f"{a}/{b}":v for (a,b),v in zip(edges,vals)},title="Delivery routes",unit="min")
    shifts=rng.sample(["Day","Evening","Night","Morning"],2)
    data={f"{l}/{s}":v for l in labels for s,v in zip(shifts,rng.sample(range(8,56),2))}
    return dict(labels=labels,shifts=shifts,data=data,title="Dispatch totals by shift",unit="parcels")


def ask(spec, question, keys, op, rule, labels=None):
    vals=[spec["data"][k] for k in keys]
    opts=labels or []
    answer=oracle(op,vals,opts)
    e=", ".join(f"{k.replace('/', ' / ')} = {v}" for k,v in zip(keys,vals))
    rel=dict(read="read",sum="add",delta="signed last minus first",travel="sum absolute successive differences",
             gap="absolute difference",larger="compare counts",larger_rate="compare each group's fraction",
             combined_pending="sum received minus closed",shorter_route="compare two route sums")[op]
    return dict(question=question,op=op,keys=keys,labels=opts,result=answer,
                reference=dict(e=e,r=rule,a=f"{rel} for [{', '.join(map(str,vals))}] gives {answer}.",
                               required_values=vals,relation=rel,result=answer,source=keys,unit=spec["unit"]))


def ledger_pair(s,kind):
    l,m,n,_=s["labels"]; a=lambda x:f"{x}/Closed"; b=lambda x:f"{x}/Received"
    if kind=="E":
        return [ask(s,f"How many orders are shown as {field.lower()} for {row}?",[k],"read","Look up the named row and status field.")
                for field,row,k in (("closed",l,a(l)),("received",m,b(m)))]
    if kind=="R":
        return [
            ask(s,f"Of {l} and {m}, which team closed more orders?",[a(l),a(m)],"larger","Compare the two closed counts.",[l,m]),
            ask(s,f"Of {l} and {m}, which team closed a greater fraction of its own received orders?",
                [a(l),b(l),a(m),b(m)],"larger_rate","Divide each closed count by that team's own received count, then compare.",[l,m])]
    pending=lambda x,y:ask(s,f"Counting {x} and {y} together, how many received orders were still open?",
            [b(x),a(x),b(y),a(y)],"combined_pending","For each named team subtract closed from received, then add the two pending counts.")
    return [pending(l,m), pending(m,n)] if kind=="A" else [pending(l,m)]*2


def trace_pair(s,kind):
    x,y=s["labels"]; ts=s["stations"]; points=lambda z: [f"{z}/{t}" for t in ts]
    if kind=="E":
        return [ask(s,f"What reading is labeled for probe {z} at check {t}?",[f"{z}/{t}"],"read","Read the named trace at the named horizontal-axis check.")
                for z,t in ((x,ts[1]),(y,ts[2]))]
    delta=lambda z,i,j:ask(s,f"For {z}, what is the signed change from check {ts[i]} to check {ts[j]}?",
                 [points(z)[i],points(z)[j]],"delta","Subtract the starting reading from the ending reading.")
    travel=lambda z:ask(s,f"For {z}, what is the total change traveled over all three segments, counting increases and decreases by their magnitudes?",
                 points(z),"travel","Add the absolute change across each consecutive pair.")
    if kind=="R": return [delta(x,0,3),travel(x)]
    if kind=="A": return [delta(y,0,3),delta(y,3,0)]
    return [travel(y)]*2


def route_pair(s,kind):
    x,y,z,w=s["labels"]; xy,yw,xz,zw,xw=[f"{a}/{b}" for a,b in s["edges"]]
    direct=lambda a,b,key:ask(s,f"How many minutes are on the direct arrow from {a} to {b}?",[key],"read","Use the number on the directed edge with those endpoints.")
    via=lambda node,keys:ask(s,f"How many minutes does the trip from {x} to {w} take when it must pass through {node}?",keys,"sum","Add only the two arrows along the required route.")
    if kind=="E":return [direct(x,y,xy),direct(z,w,zw)]
    if kind=="R":return [direct(x,w,xw),via(y,[xy,yw])]
    if kind=="A":return [via(y,[xy,yw]),via(z,[xz,zw])]
    pick=ask(s,f"Between {y} and {z}, which intermediate stop gives a shorter two-leg trip from {x} to {w}?",
             [xy,yw,xz,zw],"shorter_route","Compare the totals of the two permitted two-leg routes; exclude the direct arrow.",[y,z])
    return [pick]*2


def bar_pair(s,kind):
    x,y,z,_=s["labels"]; d,t=s["shifts"]
    pair=lambda row:[f"{row}/{d}",f"{row}/{t}"]
    if kind=="E":
        return [ask(s,f"How many parcels are displayed at {row} for the {shift.lower()} shift?",[f"{row}/{shift}"],"read","Read the bar for the named series and terminal.")
                for row,shift in ((y,d),(z,t))]
    total=lambda row:ask(s,f"Across both shifts, how many parcels did {row} dispatch?",pair(row),"sum","Add the two shift counts for this terminal.")
    gap=lambda row:ask(s,f"How many parcels apart are the two shift counts for {row}?",pair(row),"gap","Subtract the smaller shift count from the larger one.")
    if kind=="R":return [total(x),gap(x)]
    signed=lambda row,u,v:ask(s,f"At {row}, what is the signed difference when the {v.lower()} count is taken minus the {u.lower()} count?",
             [f"{row}/{u}",f"{row}/{v}"],"delta","Subtract the named baseline from the named target.")
    if kind=="A":return [signed(y,d,t),signed(y,t,d)]
    return [total(z)]*2


GENERATORS=dict(ledger=ledger_pair,trace=trace_pair,route=route_pair,bars=bar_pair)


def font(size,bold=False):
    names=(["arialbd.ttf","DejaVuSans-Bold.ttf"] if bold else ["arial.ttf","DejaVuSans.ttf"])
    for name in names:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    raise IOError("An Arial or DejaVu Sans font is required.")


def draw_image(spec,family,rng,path):
    ink="#202b35"; muted="#455967"; paper="#fcfcf9"; grid="#cad6db"
    colors=rng.choice([("#315f91","#9a4926"),("#22695c","#754780"),("#963e5f","#256e82")])
    im=Image.new("RGB",(780,540),paper); dr=ImageDraw.Draw(im)
    def label(xy,val,size=19,align="mm",bold=False,color=ink):
        dr.text(xy,str(val),font=font(size,bold),fill=color,anchor=align)
    label((30,39),spec["title"],29,"lm",True)
    d=spec["data"]; labels=spec["labels"]
    if family=="ledger":
        xs=[90,400,623]; field=spec["fields"]
        for j,k in enumerate(["Team"]+field):
            label((xs[j],122),k,22,"lm" if not j else "mm",True)
        for i,l in enumerate(labels):
            yy=179+i*75
            dr.rectangle((55,yy-29,729,yy+31), fill="#e9edef" if i%2==0 else paper)
            label((xs[0],yy),l,24,"lm");label((xs[1],yy),d[f"{l}/{field[0]}"],24);label((xs[2],yy),d[f"{l}/{field[1]}"],24)
        label((58,484),"Counts are orders; each row is one team.",17,"lm",color=muted)
    elif family=="trace":
        ts=spec["stations"]; xs=[125,309,493,677]
        yp=lambda v: int(458-(v+15)*8.8)
        for v in range(-15,26,5):
            y=yp(v);dr.line((90,y,706,y),fill=grid,width=1);label((77,y),v,17,"rm",color=muted)
        for j,t in enumerate(ts):label((xs[j],493),t,20)
        # Labels are attached to the point, offset by series and opposite the other reading at that time.
        for j,s in enumerate(labels):
            ps=[(xx,yp(d[f"{s}/{t}"])) for xx,t in zip(xs,ts)]
            dr.line(ps,fill=colors[j],width=4)
            legend_x=500+j*118
            dr.line((legend_x,42,legend_x+24,42),fill=colors[j],width=4)
            label((legend_x+30,42),s,18,"lm",True,colors[j])
            for (xx,yy),t in zip(ps,ts):
                dr.ellipse((xx-5,yy-5,xx+5,yy+5),fill=colors[j])
                vc=d[f"{s}/{t}"]; other=d[f"{labels[1-j]}/{t}"]
                tx=xx+(-15 if j==0 else 15); ty=yy+(-15 if vc>other else 16)
                dr.text((tx,ty),str(vc),font=font(17,True),fill=colors[j],anchor="mm",stroke_width=2,stroke_fill=paper)
        label((22,79),"Units",15,"lm",color=muted)
    elif family=="route":
        pos=[(103,291),(385,132),(385,451),(664,291)]
        if rng.randrange(2): pos[1],pos[2]=pos[2],pos[1]
        pts=dict(zip(labels,pos))
        for a,b in spec["edges"]:
            ax,ay=pts[a]; bx,by=pts[b]; dx,dy=bx-ax,by-ay; h=math.hypot(dx,dy)
            ux,uy=dx/h,dy/h; start=(ax+42*ux,ay+42*uy); end=(bx-44*ux,by-44*uy)
            dr.line([start,end],fill=colors[0],width=3); ex,ey=end
            dr.polygon([(ex,ey),(ex-13*ux+6*uy,ey-13*uy-6*ux),(ex-13*ux-6*uy,ey-13*uy+6*ux)],fill=colors[0])
            mx=(ax+bx)/2; my=(ay+by)/2
            dr.rounded_rectangle((mx-27,my-14,mx+27,my+14),radius=4,fill=paper,outline=grid)
            label((mx,my),d[f"{a}/{b}"],19)
        for l,(x,y) in pts.items():
            dr.rounded_rectangle((x-39,y-18,x+39,y+18),radius=6,fill="#e4edf2",outline=colors[0])
            label((x,y),l,17,bold=True)
        label((38,505),"Numbers on arrows give travel time in minutes.",17,"lm",color=muted)
    else:
        shifts=spec["shifts"]; xs=[130,301,472,643]; yp=lambda v:446-v*5.6
        for v in range(0,61,10):
            y=yp(v);dr.line((67,y,736,y),fill=grid)
            label((57,y),v,17,"rm",color=muted)
        for j,s in enumerate(shifts):
            dr.rectangle((422+163*j,63,444+163*j,80),fill=colors[j])
            label((452+163*j,72),s,17,"lm")
            for x,l in zip(xs,labels):
                v=d[f"{l}/{s}"]; bx=x-36+40*j
                dr.rectangle((bx,yp(v),bx+30,yp(0)),fill=colors[j]);label((bx+15,yp(v)-13),v,18,bold=True)
        for x,l in zip(xs,labels):label((x,477),l,19)
    im.save(path)


def choices(spec,pair,letters,rng,kind):
    results=[c["result"] for c in pair]
    if kind!="choice" and len(set(results))!=2: raise ValueError("same result")
    if pair[0]["labels"]:
        pool=spec["labels"][:]
    else:
        pool=sorted(set(spec["data"].values()))
        pool=[str(v) for v in pool]
        for c in pair:
            v=int(c["result"])
            pool.extend(str(v+i) for i in (-3,-2,-1,1,2,3))
    distractors=sorted(set(pool)-set(results))
    rng.shuffle(distractors)
    if kind!="choice":
        base=[None]*4
        for l,v in zip(letters,results):base[l]=v
        for j in range(4):
            if base[j] is None:base[j]=distractors.pop()
        return [base,base.copy()]
    baseline=distractors[:3]
    baseline.insert(letters[0],results[0])
    second=[None]*4; second[letters[1]]=results[0]
    order=distractors[:3];rng.shuffle(order)
    for j in range(4):
        if second[j] is None:second[j]=order.pop()
    return [baseline,second]


def main():
    rng=random.Random(SEED)
    dest=HERE/"images";dest.mkdir(parents=True,exist_ok=True)
    cards=[]; scenes=[]
    for kind in KINDS:
        # Each ordered pair of different answer positions occurs exactly once in each stratum.
        transitions=[(l,r) for l in range(4) for r in range(4) if l!=r]
        rng.shuffle(transitions)
        design=FAMILIES*3;design=list(design);rng.shuffle(design)
        for j,(family,transition) in enumerate(zip(design,transitions),1):
            while True:
                s=make_spec(family,rng)
                try:
                    q=GENERATORS[family](s,kind)
                    if family=="trace" and any(s["data"][f"{s['labels'][0]}/{t}"]==s["data"][f"{s['labels'][1]}/{t}"] for t in s["stations"]):continue
                    options=choices(s,q,transition,rng,kind)
                    break
                except (AssertionError, ValueError):continue
            group=f"ep-{kind.lower()}-{j:02d}"
            # Generate a fresh independent scene per group. Only the two questions in a group share pixels.
            path=dest/(group+".png");draw_image(s,family,rng,path)
            scenes.append(dict(group=group,kind=kind,family=family,spec=s,rgb_sha256=hashlib.sha256(Image.open(path).tobytes()).hexdigest()))
            for pos,card in enumerate(q):
                ref=card["reference"]
                cards.append(dict(id=group+f"-{pos+1}",group=group,kind=kind,family=family,image="paired/images/"+path.name,
                                  question=card["question"],options=options[pos],gold=LETTERS[transition[pos]],
                                  reference=ref,witness=ref["e"]+". "+ref["r"]+" "+ref["a"],
                                  oracle=dict(op=card["op"],keys=card["keys"],labels=card["labels"])))
    for filename,data in (("items.jsonl",cards),("sources.jsonl",scenes)):
        (HERE/filename).write_text("".join(json.dumps(c,ensure_ascii=False)+"\n" for c in data),encoding="utf-8")


if __name__=="__main__":main()
