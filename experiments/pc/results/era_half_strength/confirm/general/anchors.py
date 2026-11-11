"""Sixteen independent diagram-code and measurement-scope questions."""
from io import BytesIO
from PIL import Image,ImageDraw,ImageFont

INK="#17253a";WHITE="#fffef8"
PALE=["#d8e1f3","#f2caa5","#bfe4d9","#e5d6ea"]
LABELS=[
 ("Observatory dial",["Axis","Rotor","Lens","Brace"],["F","U","K","J"]),
 ("Survey instruments",["Probe","Level","Tape","Gauge"],["H","M","B","W"]),
 ("Vessel fittings",["Cleat","Spar","Keel","Crank"],["P","Q","V","N"]),
 ("Turbine assembly",["Vane","Hub","Blade","Seal"],["L","G","Y","C"]),
 ("Map compartments",["Atlas","Maps","Pins","Grids"],["X","B","J","D"]),
 ("Mosaic modules",["Wing","Arc","Pearl","Ring"],["N","Q","W","V"]),
 ("Theatre rig",["Beam","Fly","Rail","Deck"],["G","A","Z","H"]),
 ("Survey site legend",["Copse","Dune","Ford","Peak"],["K","T","P","W"])]
SCOPES=[
 ("Orchid count","East bed","West bed","blooms",[18,26],"West had eight more blooms at this count.",["East had more blooms.","All blooms had opened.","Eight blooms opened each hour."]),
 ("Package tally","Before noon","After noon","parcels",[33,25],"Eight more parcels were counted before noon.",["The parcels were all the same mass.","After noon had the larger count.","Eight parcels were damaged."]),
 ("Reservoir checks","Tuesday","Friday","litres",[42,59],"The Friday reading was seventeen litres higher.",["The reservoir rose every day.","Tuesday's reading was higher.","Seventeen litres entered every hour."]),
 ("Shelf inventory","Upper shelf","Lower shelf","tins",[21,37],"The lower shelf held sixteen more tins.",["The upper shelf held more tins.","The lowest price was on the lower shelf.","Every tin had the same contents."]),
 ("Site tally","Cedar","Birch","saplings",[53,44],"Cedar had nine more saplings in the census.",["Birch had more saplings.","The trees were all the same species.","Cedar will always have more saplings."]),
 ("Evening riders","Line A","Line B","riders",[29,47],"Eighteen more riders were counted on Line B.",["Line A had more riders.","Every rider paid the same fare.","Eighteen riders changed lines."]),
 ("Stack inspection","Stack X","Stack Z","plates",[74,58],"Stack X had sixteen more plates.",["The two stacks had equal counts.","Stack Z had more plates.","All plates were clean."]),
 ("Instrument readings","Run 1","Run 2","seconds",[63,76],"Run 2 lasted thirteen seconds longer.",["Run 1 lasted longer.","Both runs had the same speed.","Every future run will last longer."])]


def font(n):
    for f in ("C:/Windows/Fonts/arial.ttf","DejaVuSans.ttf"):
        try:return ImageFont.truetype(f,n)
        except OSError:pass
    raise RuntimeError("Font unavailable")


def put(d,xy,s,n=24,anchor="lm",color=INK):
    d.text(xy,str(s),fill=color,font=font(n),anchor=anchor)


def canvas(title,subtitle):
    im=Image.new("RGB",(820,490),WHITE);d=ImageDraw.Draw(im)
    d.rectangle((0,0,820,109),fill=INK)
    put(d,(32,37),title,34,color="white");put(d,(32,82),subtitle,20,color="white")
    return im,d


def make():
    for i in range(16):
        pos=(i*3+1)%4
        if i<8:
            scene,names,tags=LABELS[i]
            im,d=canvas(scene,"Item-name and code assignments")
            target=(i*3)%4
            for j,(name,tag) in enumerate(zip(names,tags)):
                if i%2:
                    x=34+(j%2)*392;y=146+(j//2)*143
                    d.polygon([(x,y),(x+340,y),(x+370,y+55),(x+340,y+111),(x,y+111)],fill=PALE[(j+i)%4])
                    put(d,(x+18,y+34),name);put(d,(x+263,y+70),tag,32)
                else:
                    x=32+j*193;y=151+(j%2)*72
                    d.rounded_rectangle((x,y,x+172,y+210),radius=22,fill=PALE[(j+i)%4])
                    put(d,(x+86,y+51),tag,36,"mm");put(d,(x+86,y+145),name,24,"mm")
            question=f"On the {scene.lower()} sheet, which code is assigned to {names[target]}?"
            ans=tags[target];opts=[t for t in tags if t!=ans];typ="label"
            witness="The printed pairs are "+", ".join(n+"–"+t for n,t in zip(names,tags))+"."
            rule="Read the one code printed with the requested item."
        else:
            scene,one,two,unit,(a,b),ans,alts=SCOPES[i-8]
            im,d=canvas(scene,unit.capitalize()+" recorded at these two observations")
            opts=list(alts);question=f"Which statement follows from the two observations in the {scene.lower()}?"
            if i%2:
                for j,(l,v) in enumerate([(one,a),(two,b)]):
                    x=96+356*j;d.line((x+96,411,x+96,411-int(225*v/max(a,b))),fill=INK,width=8)
                    y=411-int(225*v/max(a,b));d.ellipse((x+40,y-39,x+152,y+39),fill=PALE[(j+i)%4],outline=INK,width=3)
                    put(d,(x+96,y),v,27,"mm");put(d,(x+96,446),l,24,"mm")
            else:
                for j,(l,v) in enumerate([(one,a),(two,b)]):
                    y=190+138*j;put(d,(36,y),l,25);w=int(455*v/max(a,b))
                    d.rectangle((255,y-31,255+w,y+31),fill=PALE[(i+j)%4],outline=INK,width=2)
                    put(d,(265+w,y),v,25)
            typ="qualifier";witness=f"{one} is {a} and {two} is {b} {unit}. {ans}"
            rule="Compare precisely the two measured quantities, in the indicated units and window."
        opts.insert(pos,ans);buf=BytesIO();im.save(buf,format="PNG")
        yield dict(id=f"hc-guard-{i+1:02}",source="new authored confirmation control",
            source_id=f"hc-guard-{i+1:02}",kind="retention anchor",question=question,
            options=opts,gold="ABCD"[pos],sentinel=True,sentinel_type=typ,
            reference=dict(visual_witness=witness,rule=rule)),buf.getvalue()
