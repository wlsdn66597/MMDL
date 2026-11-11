"""Construct the 24 new A-intervention training images and their matched targets."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / ".runtime/tokenizers"))
from tokenizers import Tokenizer

LETTERS = "ABCD"
COLORS = {"change":"#315673","mix":"#656046","rate":"#3a6661",
          "locator":"#654768","record":"#465978"}
FONT_ROOT = Path("C:/Windows/Fonts")


def text(draw, x, y, s, size=24, *, bold=False, color="#16293a", anchor="lm"):
    font = ImageFont.truetype(str(FONT_ROOT / ("arialbd.ttf" if bold else "arial.ttf")), size)
    draw.text((x,y), str(s), font=font, fill=color, anchor=anchor)


def chart(src):
    """Every cell printed comes directly from the same public source used by the oracle."""
    w,h = 850,460
    image=Image.new("RGB",(w,h),"#ffffff")
    d=ImageDraw.Draw(image)
    color = COLORS[src["family"]]
    d.rectangle((0,0,w,84),fill=color)
    text(d,30,29,src["title"],30,bold=True,color="white")
    text(d,30,62,src["subtitle"],18,color="white")
    if src["family"]=="locator":
        for i,(part,mark) in enumerate(src["rows"]):
            x,y=38+392*(i%2),105+155*(i//2)
            d.rounded_rectangle((x,y,x+365,y+127),radius=12,fill="#f4eef4",outline=color,width=2)
            d.ellipse((x+17,y+29,x+88,y+100),fill=color)
            text(d,x+52,y+65,mark,29,bold=True,anchor="mm",color="white")
            text(d,x+114,y+65,part,23,bold=True)
        return image
    xs = [163,433,682] if len(src["columns"])==3 else [215,620]
    for j,val in enumerate(src["columns"]):
        text(d,xs[j],118,val,21,bold=True,anchor="mm")
    for i,row in enumerate(src["rows"]):
        top=147+i*73
        d.rounded_rectangle((30,top,818,top+57),radius=6,fill="#eef3f6" if i%2==0 else "#f7f8f9")
        for j,val in enumerate(row):
            text(d,xs[j],top+29,val,24,bold=j==0,anchor="mm")
    return image


def js(data):
    return json.dumps(data,ensure_ascii=False,indent=2)+"\n"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(family,kind,n,title,subtitle,columns,rows,qs):
    gid=f"r9-fit-{family}-{n+1:02d}"
    return dict(group=gid,family=family,kind=kind,title=title,subtitle=subtitle,
                columns=columns,rows=rows),qs


def question(question, truth, e, r, usual, detailed, distractors, op):
    return dict(stem=question,answer=str(truth),e=e,r=r,ordinary=usual,enhanced=detailed,
                distractors=[str(x) for x in distractors],op=op)


def change_groups():
    fixtures=[
        ("West gantry offset",["Seal","Stop","Loop","Dock"],[11,4,15,-2]),
        ("Cutter slide log",["Start","Pin","Jaw","End"],[-11,1,-3,8]),
        ("Probe axis position",["Zero","Rack","Tip","Park"],[6,-8,-1,2]),
        ("Hoist register",["Nest","Load","Rise","Rest"],[-2,13,5,7])]
    for i,(title,stages,values) in enumerate(fixtures):
        x,y,z,u=values
        e="The logged readings are "+", ".join(f"{s} {v} mm" for s,v in zip(stages,values))+", in time order."
        r1="Subtract the earliest position from the latest position, retaining the sign."
        r2="Sum the magnitudes of the differences between neighboring positions along the logged route."
        net=u-x
        seg=[abs(b-a) for a,b in zip(values,values[1:])]
        travel=sum(seg)
        opts=[net,travel,-net,travel+abs(net)]
        s,t=stages[0],stages[-1]
        qs=[
          question(f"What is the signed change in position from {s} to {t}, in mm?",net,e,r1,
                   f"The readings at the two endpoints are {x} and {u} mm. The final value minus the initial value gives {net} mm.",
                   f"{t} minus {s} is ({u}) - ({x}) = {net} mm. The signed endpoint change is {net} mm.",opts,"net"),
          question(f"How many mm were traveled over only the three logged moves from {s} through {t}?",travel,e,r2,
                   f"The successive distances in the log are {seg[0]}, {seg[1]}, and {seg[2]} mm. Their sum gives a travel distance of {travel} mm.",
                   f"{stages[0]} to {stages[1]}: {seg[0]} mm; {stages[1]} to {stages[2]}: {seg[1]} mm; {stages[2]} to {stages[3]}: {seg[2]} mm. Total = {seg[0]}+{seg[1]}+{seg[2]} = {travel} mm.",opts,"travel")]
        yield group("change","A",i,title,"First to last, with no additional moves","Log point|Position (mm)".split("|"),list(zip(stages,values)),qs)


def mix_groups():
    fixtures=[
        ("Rinse solution stock",["Cress","Tamar"],[15,30],[12,54]),
        ("Bath concentrate log",["Wisp","Kelp"],[30,10],[18,66]),
        ("Tank assay record",["Dale","Crux"],[18,12],[8,68]),
        ("Storage blend sheet",["Umber","Lark"],[12,36],[24,56])]
    for i,(title,names,vol,pct) in enumerate(fixtures):
        p,q=pct;v,w=vol;n,m=names
        eq=(p+q)//2
        all_=int(Fraction(v*p+w*q,v+w))
        opts=[eq,all_,p,q]
        e=f"{n} holds {v} L at {p}% and {m} holds {w} L at {q}% active component by volume."
        r1="For equal volumes, the concentration is the arithmetic mean of the two percentages."
        r2="For a pooled stock, weight each concentration by its full volume and divide by the combined volume."
        qs=[
          question(f"A technician pours one litre of {n} and one litre of {m}. What percent of the sample is active component?",eq,e,r1,
                   f"The equal-sized samples receive equal weights. The mean percentage of {p} and {q} is ({p} + {q})/2 = {eq}%.",
                   f"{n} contributes {p}%/2 = {p//2} percentage points and {m} contributes {q}%/2 = {q//2}. The blend is {p//2}+{q//2} = {eq}%.",opts,"equal"),
          question(f"If the entire {n} and {m} stocks are emptied into one tank, what percentage of the combined volume is active component?",all_,e,r2,
                   f"Using every litre gives ({v} × {p} + {w} × {q}) / ({v} + {w}). The pooled sample has an active concentration of {all_}%.",
                   f"{n} supplies {v}×{p}% = {v*p/100:g} L active; {m} supplies {w}×{q}% = {w*q/100:g} L. Together {((v*p+w*q)/100):g}/{v+w} = {all_}%.",opts,"pooled")]
        yield group("mix","A",i,title,"Volumes and active-component percentages of two full stocks",
                    ["Vessel","Stock volume (L)","Active (%)"],list(zip(names,vol,pct)),qs)


def rate_groups():
    fixtures=[
        ("Packing quality checks",["Alder","Cirro","Firth","Duet"],[(42,60),(44,80),(27,30),(34,50)]),
        ("Parts released",["Mauve","Nectar","Sienna","Wick"],[(30,40),(35,50),(24,40),(64,100)]),
        ("Parcel scans",["Lynx","Cairn","Twine","Pearl"],[(22,40),(54,60),(39,50),(63,90)]),
        ("Valve trials",["Plait","Indigo","Yarrow","Fable"],[(69,100),(36,50),(24,30),(38,50)])]
    for i,(title,names,pairs) in enumerate(fixtures):
        numbers=[x for x,y in pairs]
        perc=[int(Fraction(x*100,y)) for x,y in pairs]
        ci=numbers.index(max(numbers));ri=perc.index(max(perc));assert ci!=ri
        countwin=names[ci];ratewin=names[ri]
        evidence="; ".join(f"{n} passed {x} of {y}" for n,(x,y) in zip(names,pairs))+"."
        r1="Compare the observed passed counts themselves across every station."
        r2="Compare passed per tested for every station; equal future test counts preserve the order of these fractions."
        qs=[
          question(f"In the {title.lower()} report, which station had the largest observed number of passes?",countwin,evidence,r1,
                   f"The observed counts in chart order are {', '.join(map(str,numbers))}. The largest is {numbers[ci]}, belonging to {countwin}.",
                   ", ".join(f"{n} {x}" for n,x in zip(names,numbers))+f". The maximum is {countwin} with {numbers[ci]} passes.",names,"count"),
          question(f"Using the {title.lower()} report, if every station tests the same number of new items at its observed pass rate, which station leads in expected passes?",ratewin,evidence,r2,
                   f"The four stations pass {', '.join(str(p)+'%' for p in perc)}, respectively. The maximum is {perc[ri]}%, so {ratewin} leads.",
                   ", ".join(f"{n} {x}/{y}={p}%" for n,(x,y),p in zip(names,pairs,perc))+f". Highest: {ratewin}.",names,"rate")]
        yield group("rate","A",i,title,"Passed checks by station in the same reporting period",
                    ["Station","Passed","Tested"],[(n,*p) for n,p in zip(names,pairs)],qs)


def protection_groups():
    designs=[
        ("Film archive",["Splicer","Cooler","Scanner","Rewinder"]),
        ("Ceramic studio",["Shelf","Shim","Chisel","Kiln"]),
        ("Bicycle shop",["Crank","Caliper","Spindle","Pulley"]),
        ("Survey station",["Tripod","Target","Level","Staff"]),
        ("Printing room",["Inker","Platen","Roller","Feeder"]),
        ("Water test cart",["Gauge","Sampler","Sieve","Burette"])]
    for i,(title,objects) in enumerate(designs):
        marks=["G","J","M","P"]
        random.Random(9123+i).shuffle(marks)
        k=(3*i+2)%4; mark=marks[k]; obj=objects[k]
        e=f"The {obj.lower()} placard displays {mark}."
        r="Read the marker attached to the named object, then locate that marker among the options."
        a=f"The marker for the {obj.lower()} is {mark}."
        qs=[question(stem,mark,e,r,a,a,marks,"mark") for stem in (
            f"Which marker is printed next to the {obj.lower()}?",f"Choose the placard marker for the {obj.lower()} in this layout.")]
        yield group("locator","SYMBOL",i,title,"Equipment placards in four bays",["Object","Marker"],list(zip(objects,marks)),qs)
    fixtures=[
        ("Mineral sample log","Mass (g)","Colour",["Jet","Salt","Mica","Chalk"],["black","white","silver","cream"]),
        ("Cloth inspection","Width (cm)","Weave",["Bolt U","Bolt F","Bolt K","Bolt J"],["plain","twill","satin","basket"]),
        ("Seed lot cards","Pods","Condition",["Lot N","Lot H","Lot I","Lot E"],["dry","fresh","sealed","open"]),
        ("Cable inventory","Length (m)","Jacket",["Cord X","Cord Z","Cord V","Cord W"],["navy","gold","teal","ivory"]),
        ("Tile samples","Width (mm)","Surface",["Tile K","Tile M","Tile T","Tile F"],["smooth","ribbed","matte","glazed"]),
        ("Packet log","Count","Shelf",["Tray Y","Tray R","Tray P","Tray A"],["high","low","middle","floor"])]
    for i,(title,num_col,cat_col,names,tags) in enumerate(fixtures):
        vals=[16+i*3,41+i*2,28+i,10+2*i]
        a=i%4;b=(i+2)%4
        ev=f"The {names[a]} row records {vals[a]} in the {num_col} column."
        rv="Read the numeric entry in the specified row and column."
        av=f"The recorded value for {names[a]} is {vals[a]}."
        ec=f"In the {cat_col} column, {tags[b]} belongs to {names[b]}."
        rc="Restrict attention to the row whose categorical entry matches the question."
        ac=f"The row meeting that condition is {names[b]}."
        qs=[question(f"What number appears in the {num_col} column for {names[a]}?",vals[a],ev,rv,av,av,vals,"direct_number"),
            question(f"Which entry is labeled {tags[b]} in the {cat_col} column?",names[b],ec,rc,ac,ac,names,"restricted")]
        yield group("record","GENERAL",i,title,"One row is recorded for each item in this collection",
                    ["Item",num_col,cat_col],list(zip(names,vals,tags)),qs)


def render(t,gold):
    return f"Evidence: {t['e']}\nRule: {t['r']}\nApplication: {t['a']}\nFinal answer: {gold}"


def main():
    image_dir=HERE/"images"
    image_dir.mkdir(exist_ok=True,parents=True)
    tok=Tokenizer.from_file(str(ROOT/".runtime/qwen_tokenizer/tokenizer.json"))
    groups=[*change_groups(),*mix_groups(),*rate_groups(),*protection_groups()]
    entries=[];sources=[];tokens=Counter()
    for i,(s,qs) in enumerate(groups):
        g=s["group"];img=chart(s);img_path=image_dir/(g+".png");img.save(img_path)
        pic=sha(img_path)
        first=i%4;second=(first+1+(i//4)%3)%4
        positions=(first,second)
        shared_opts=None
        if s["kind"]=="A":
            opts=list(qs[0]["distractors"])
            assert len(set(opts))==4 and set(opts)==set(qs[1]["distractors"])
            shared_opts=[None]*4
            for slot,q in zip(positions,qs):
                shared_opts[slot]=q["answer"]; opts.remove(q["answer"])
            random.Random(912113+i).shuffle(opts)
            for slot in range(4):
                if shared_opts[slot] is None:shared_opts[slot]=opts.pop()
        for j,(pos,q) in enumerate(zip(positions,qs)):
            opts=list(q["distractors"]); assert len(set(opts))==4
            if shared_opts:opts=shared_opts.copy()
            else:
                opts.remove(q["answer"]);random.Random(912113+i*83+j).shuffle(opts)
                opts.insert(pos,q["answer"])
            gold=LETTERS[pos]
            assert opts[pos]==q["answer"]
            row=dict(id=f"{g}-q{j+1}",group=g,kind=s["kind"],family=s["family"],
                     image=img_path.relative_to(HERE.parent).as_posix(),image_sha256=pic,
                     question=q["stem"],options=opts,gold=gold)
            for arm in ("ordinary","enhanced"):
                a=q[arm]
                if arm=="enhanced" or s["kind"]!="A":
                    a+=f" This is option {gold}."
                else:
                    a+=f" The listed choice for {q['answer']} is option {gold}."
                row[arm]=dict(e=q["e"],r=q["r"],a=a)
                if s["kind"]=="A":
                    tokens[(s["family"],arm)]+=len(tok.encode(render(row[arm],gold),add_special_tokens=False).ids)+1
            entries.append(row)
        sources.append(dict(group=g,kind=s["kind"],family=s["family"],source=s,image_sha256=pic,
                            image_size=img.size,oracle=[dict(op=q["op"],answer=q["answer"]) for q in qs]))
    (HERE/"rows.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in entries),encoding="utf8")
    (HERE/"sources.json").write_text(js(sources),encoding="utf8")
    for index in range(4):
        sheet=Image.new("RGB",(1210,3*320),"white");draw=ImageDraw.Draw(sheet)
        for slot,s in enumerate(sources[index*6:index*6+6]):
            im=ImageOps.contain(Image.open(image_dir/(s["group"]+".png")),(589,291))
            x=(slot%2)*605;y=(slot//2)*320
            sheet.paste(im,(x+(605-im.width)//2,y+23));text(draw,x+10,y+12,s["group"],14)
        sheet.save(HERE/f"sheet_{index}.jpg",quality=91)
    print(dict(tokens))


if __name__=="__main__":
    main()
