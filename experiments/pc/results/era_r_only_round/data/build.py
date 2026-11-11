"""Build paired R-only training targets from original schematics and six ScienceQA train images."""
import hashlib
from fractions import Fraction
from io import BytesIO
import json
import math
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageFont
from photo_items import RECORDS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
INK="#18273a"; SOFT="#e6ebf0"; GREEN="#18665c"; ORANGE="#a84129"
FONT_DIR=Path("C:/Windows/Fonts")


def font(size,bold=False):
    path=FONT_DIR/("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(path),size=size)


def label(draw,x,y,text,size=23,anchor="mm",color=INK,bold=False):
    draw.text((x,y),str(text),font=font(size,bold),fill=color,anchor=anchor)


def board(title,subtitle,header):
    im=Image.new("RGB",(860,465),"white");d=ImageDraw.Draw(im)
    d.rectangle((0,0,860,81),fill=header);label(d,28,25,title,27,"lm","white",True)
    label(d,28,62,subtitle,17,"lm","white")
    return im,d


def sha(data):return hashlib.sha256(data).hexdigest()
def js(data):return (json.dumps(data,ensure_ascii=False,indent=2)+"\n").encode()


def line_image(s):
    im,d=board(s["title"],s["unit"]+" at successive test checkpoints","#243746")
    ys=[*s['values'],*s['other']];lo,hi=min(ys)-4,max(ys)+4
    X=[105,316,527,737]
    def Y(n):return round(365-(n-lo)/(hi-lo)*230)
    for val in range(math.ceil(lo/5)*5,hi+1,5):
        yy=Y(val);d.line((75,yy,777,yy),fill=SOFT,width=1);label(d,57,yy,val,15,"rm")
    for gx,t in zip(X,s["stations"]):label(d,gx,396,t,20)
    for track,vals,col,j in ((s["channel"],s["values"],GREEN,0),(s["reference"],s["other"],ORANGE,1)):
        xx=157+300*j;d.line((xx-58,104,xx-15,104),fill=col,width=4);label(d,xx,104,track,19,"lm",col,True)
        points=[(gx,Y(v)) for gx,v in zip(X,vals)]
        d.line(points,fill=col,width=4)
        for (x,y),v in zip(points,vals):
            d.ellipse((x-5,y-5,x+5,y+5),fill=col)
            label(d,x+(12 if j else -12),y+(15 if j else -15),v,18,"mm",col,True)
    label(d,434,439,"Checkpoint",17)
    return im


def ledger_image(s):
    im,d=board(s["title"],s["scale"],"#354942")
    head=[s["entity"],s["top"],s["bottom"]]
    col=[175,432,684]
    for i,h in enumerate(head):label(d,col[i],131,h,23,bold=True)
    for j,(name,a,b) in enumerate(s["data"]):
        y=185+j*75
        d.rectangle((48,y-28,812,y+33),fill=("#ecf3ef" if j%2==0 else "#f9fafa"))
        for k,v in enumerate([name,a,b]):label(d,col[k],y,v,24)
    label(d,49,431,"Counts refer to a single reporting period.",17,"lm")
    return im


def arrow(draw,p1,p2,col):
    x,y=p1;xx,yy=p2;angle=math.atan2(yy-y,xx-x)
    draw.line((x,y,xx,yy),fill=col,width=4)
    tri=[(xx,yy)]+[(xx-14*math.cos(angle+u),yy-14*math.sin(angle+u)) for u in (-.44,.44)]
    draw.polygon(tri,fill=col)


def circuit_image(s):
    im,d=board(s["title"],"Arrow labels show signal delay in microseconds.", "#343456")
    positions=[(117,253),(417,129),(417,376),(739,253)]
    for (a,b,n) in s['edges']:
        x,y=positions[a];xx,yy=positions[b];ang=math.atan2(yy-y,xx-x)
        start=(x+58*math.cos(ang),y+30*math.sin(ang))
        end=(xx-61*math.cos(ang),yy-31*math.sin(ang))
        arrow(d,start,end,GREEN)
        mx,my=(x+xx)//2,(y+yy)//2
        d.rounded_rectangle((mx-20,my-16,mx+20,my+16),radius=4,fill="#fffbea",outline="#abb79b");label(d,mx,my,n,19)
    for (x,y),name in zip(positions,s["nodes"]):
        d.rounded_rectangle((x-58,y-24,x+58,y+24),radius=9,fill="#f0f2fb",outline="#8989b3",width=2);label(d,x,y,name,22,bold=True)
    return im


def symbol_image(s):
    im,d=board(s["title"],"Locator key | boxes mark positions in the bay", "#424047")
    for i,mark in enumerate(s["marks"]):
        x=91+(i%2)*349;y=111+(i//2)*146
        d.rounded_rectangle((x,y,x+294,y+119),radius=6,fill=("#f4f0eb" if i%2 else "#e9f1ee"),outline="#a1aba4",width=2)
        label(d,x+75,y+55,mark,46,"mm",INK,True);label(d,x+185,y+55,s["objects"][i],22)
    label(d,434,433,"Each letter identifies the fixture in its own box.",18)
    return im


def write_image(name,im):
    dst=HERE/"images"/(name+".png");im.save(dst);return dst


def q(stem,result,ordinary,enhanced,e,a,alternatives,op):
    return dict(stem=stem,result=str(result),ordinary=ordinary,enhanced=enhanced,e=e,a=a,alternatives=list(map(str,alternatives)),op=op)


def r_specs():
    groups=[]
    lines=[
      ("Radar alignment bench","Echo","Bias",["P1","P2","P3","P4"],[-6,7,3,10],[4,-2,13,-5]),
      ("Oven control deviation","Therm","Pilot",["D0","D1","D2","D3"],[14,2,8,-3],[-3,10,-4,5]),
      ("Receiver gain checks","Main","Spare",["C1","C2","C3","C4"],[3,-9,1,-2],[-5,2,-8,6]),
      ("Motor feedback voltage","Drive","Idle",["S1","S2","S3","S4"],[-4,12,-2,6],[7,1,9,-6])]
    for i,(title,track,other,stations,vs,control) in enumerate(lines,1):
        group=f"r8-tr-signal-{i}"
        s=dict(title=title,channel=track,reference=other,stations=stations,values=vs,other=control,unit="Instrument readings (units)")
        first,last=stations[0],stations[-1]
        signed=vs[-1]-vs[0];seg=[abs(b-a) for a,b in zip(vs,vs[1:])];total=sum(seg)
        pool=[signed,total,abs(signed) if signed<0 else -signed,total+2]
        e=f"{track} has readings "+", ".join(f"{t}={v}" for t,v in zip(stations,vs))+"."
        qs=[q(f"For {track}, how far above or below its {first} reading does it finish at {last}, with sign?",signed,
              "Compute the difference between the last and first values on the named series.",
              "For signed change, take last minus first regardless of intermediate reversals.",e,
              f"The endpoint difference is {vs[-1]} minus ({vs[0]}) = {signed} units.",pool,"signed_change"),
            q(f"How much does {track} move across the checkpoints if every rise and fall counts toward the total?",total,
              "Add the size of every step between neighboring points on the named series.",
              "For movement through reversals, sum each adjacent change in absolute value.",e,
              "The successive magnitudes are "+" + ".join(map(str,seg))+f" = {total} units.",pool,"travel")]
        groups.append((group,"R","signal",s,qs,line_image(s)))
    books=[
        ("Book returns by branch","Branch","Returned","Loaned", [("Anvil",24,41),("Lemon",19,22),("Hollow",12,30)]),
        ("Casting inspections","Batch","Accepted","Inspected",[("North",32,35),("Cedar",45,67),("Ruby",28,66)]),
        ("Repair requests closed","Desk","Closed","Assigned",[("Pearl",42,76),("Loom",37,42),("Navy",26,44)]),
        ("Archive checks completed","Team","Verified","Assigned",[("Elm",38,43),("Solar",56,79),("Grain",27,35)])]
    for i,(title,entity,top,bottom,vals) in enumerate(books,1):
        group=f"r8-tr-register-{i}";s=dict(title=title,entity=entity,top=top,bottom=bottom,data=vals,scale="Counts by "+entity.lower())
        a,b=vals[:2];win=max((a,b),key=lambda r:r[1]);rw=max((a,b),key=lambda r:Fraction(r[1],r[2]))
        assert win!=rw
        pool=[a[0],b[0],vals[2][0],"They are tied"];e=f"{a[0]} has {a[1]} {top.lower()} out of {a[2]} {bottom.lower()}; {b[0]} has {b[1]} out of {b[2]}."
        qs=[q(f"Between {a[0]} and {b[0]}, which has the larger {top.lower()} count?",win[0],
              "Compare the values in the named count column to find the larger entry.",
              "For counts rather than shares, compare the named numerator entries directly.",e,
              f"{win[1]} exceeds {min(a[1],b[1])}, so the larger count belongs to {win[0]}.",pool,"count"),
            q(f"Between {a[0]} and {b[0]}, which has a larger {top.lower()} share of its own {bottom.lower()} count?",rw[0],
              "Compute both fractions from this table, then choose the larger resulting value.",
              f"For each share, divide {top.lower()} by that row's {bottom.lower()}, then compare.",e,
              f"{a[0]}: {a[1]}/{a[2]} ≈ {a[1]/a[2]:.3f}; {b[0]}: {b[1]}/{b[2]} ≈ {b[1]/b[2]:.3f}. {rw[0]} has the larger share.",pool,"ratio")]
        groups.append((group,"R","register",s,qs,ledger_image(s)))
    paths=[
        ("Amplifier test network",["Feed","Buffer","Filter","Output"],[3,8,9,4,14],1),
        ("Control-bus paths",["Latch","Mux","Gate","Probe"],[2,7,6,5,13],2),
        ("Telemetry switch fabric",["Input","Mixer","Relay","Sink"],[7,4,8,9,15],1),
        ("Audio patch timing",["Source","Amp","Codec","Meter"],[8,4,6,9,11],2)]
    for i,(title,names,lens,sel) in enumerate(paths,1):
        group=f"r8-tr-board-{i}";edges=[(0,1,lens[0]),(1,3,lens[1]),(0,2,lens[2]),(2,3,lens[3]),(0,3,lens[4])]
        s=dict(title=title,nodes=names,edges=edges);start,end=names[0],names[-1];mid=names[sel]
        a,b=lens[2*(sel-1):2*sel];direct=lens[-1]; via=a+b
        pool=[direct,via,a,b];assert len(set(pool))==4
        e=f"The {start}→{end} link shows {direct} µs. {start}→{mid} shows {a} µs and {mid}→{end} shows {b} µs."
        qs=[q(f"What delay (µs) is printed on the direct {start}-to-{end} signal link?",direct,
              "Read the delay on the single edge connecting the two specified nodes.",
              "The direct-link request uses only the arrow joining the named endpoints.",e,
              f"The single arrow from {start} to {end} is {direct} µs.",pool,"direct"),
            q(f"What is the total delay (µs) if the signal from {start} to {end} must go through {mid}?",via,
              "Add the delays along the two links traversed by this signal.",
              "A required intermediate node means adding both hops through it, excluding the direct link.",e,
              f"The route {start}→{mid}→{end} takes {a} + {b} = {via} µs.",pool,"via")]
        groups.append((group,"R","circuit",s,qs,circuit_image(s)))
    return groups


def symbols():
    groups=[]
    sets=[("Recording studio",["Microphone","Monitor","Mixer","Recorder"]),
          ("Instrument bay",["Voltmeter","Breaker","Battery","Transducer"]),
          ("Pump station",["Motor","Valve","Drain","Filter"]),
          ("Laser workbench",["Mirror","Aperture","Prism","Shutter"]),
          ("Biology laboratory",["Centrifuge","Incubator","Scale","Microscope"]),
          ("Optics station",["Detector","Grating","Source","Lens"])]
    for i,(title,objs) in enumerate(sets,1):
        rng=random.Random(8119+i);marks=list("ABCD");rng.shuffle(marks);pos=(2*i+1)%4
        target=marks[pos];obj=objs[pos];s=dict(title=title,marks=marks,objects=objs)
        e=f"The box labeled {obj} contains the locator letter {target}."
        rule="Find the named fixture on the locator plan, then choose the option whose text is its letter."
        qs=[q(f"Which locator letter marks the {obj.lower()} in the {title.lower()}?",target,rule,rule,e,"",marks,"locator"),
            q(f"Select the locator letter shown in the box for the {obj.lower()}.",target,rule,rule,e,"",marks,"locator")]
        groups.append((f"r8-tr-symbol-{i}","SYMBOL","locator",s,qs,symbol_image(s)))
    return groups


def photographs():
    sys.path.insert(0,str(ROOT/".runtime/era_review_deps"))
    import pyarrow.parquet as pq
    metadata=ROOT/"reports/era_dataset_examples_20261014"
    ids=json.loads((metadata/"scienceqa_splits.json").read_text(encoding="utf8"))["train"]
    path=ROOT/"tmp/era_prereview_20261021/scienceqa_train.parquet"
    wanted={ids.index(r[0]):r[0] for r in RECORDS};images={}
    offset=0
    for batch in pq.ParquetFile(path).iter_batches(columns=["image"],batch_size=256):
        for ix,sid in wanted.items():
            if offset<=ix<offset+len(batch):images[sid]=batch.column(0)[ix-offset].as_py()["bytes"]
        offset+=len(batch)
    assert len(images)==6 and offset==len(ids)
    groups=[]
    for i,(sid,note,e,questions) in enumerate(RECORDS,1):
        im=Image.open(BytesIO(images[sid])).convert("RGB")
        s=dict(source="ScienceQA/train",original_id=sid,original_bytes_sha256=sha(images[sid]),field_note=note)
        qs=[q(note+" "+stem,alts[0],rule,rule,e,app,alts,"witness") for stem,alts,rule,app in questions]
        groups.append((f"r8-tr-photo-{i}","GENERAL","photo",s,qs,im))
    return groups


def main():
    (HERE/"images").mkdir(exist_ok=True,parents=True)
    groups=r_specs()+symbols()+photographs()
    qrows=[];refs=[]
    for i,(group,kind,family,s,qs,im) in enumerate(groups):
        picture=write_image(group,im);imgsha=sha(picture.read_bytes())
        firstpos=i%4
        secondpos=(firstpos+(i//4+1 if i<12 else 2))%4
        pair_options=None
        if kind=="R":
            primary,secondary=(q["result"] for q in qs)
            remaining=[v for v in qs[0]["alternatives"] if v not in (primary,secondary)]
            assert primary!=secondary and len(remaining)==2
            pair_options=[None]*4
            pair_options[firstpos]=primary;pair_options[secondpos]=secondary
            random.Random(811917+i*43).shuffle(remaining)
            for index in range(4):
                if pair_options[index] is None:pair_options[index]=remaining.pop()
        for j,v in enumerate(qs):
            group_pos=(firstpos,secondpos)[j]
            core=list(v["alternatives"]);assert len(core)==len(set(core))==4 and v["result"] in core
            rng=random.Random(811917+i*43+j);core.remove(v["result"]);rng.shuffle(core)
            core.insert(group_pos,v["result"]);gold="ABCD"[group_pos];e=v['e'];a=v['a']
            if pair_options:core=pair_options.copy()
            if kind=="SYMBOL":a=f"The {s['objects'][s['marks'].index(v['result'])]} is marked {v['result']}; option {gold} contains {v['result']}."
            else:a+=f" This corresponds to option {gold}."
            qrows.append(dict(id=f"{group}-q{j+1}",group=group,kind=kind,family=family,image=picture.relative_to(HERE.parent).as_posix(),
                image_sha256=imgsha,question=v["stem"],options=core,gold=gold,
                ordinary=dict(e=e,r=v["ordinary"],a=a),enhanced=dict(e=e,r=v["enhanced"],a=a)))
        refs.append(dict(group=group,kind=kind,family=family,source=s,image_sha256=imgsha,image_size=im.size,
                         oracle=[dict(op=v["op"],answer=v["result"]) for v in qs]))
    (HERE/"rows.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in qrows),encoding="utf8")
    (HERE/"sources.json").write_bytes(js(refs))


if __name__=="__main__":main()
