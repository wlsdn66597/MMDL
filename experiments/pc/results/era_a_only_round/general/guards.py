"""Sixteen held-out protection questions with source-grounded labels or qualified claims."""
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

PALETTE = ["#afcbd1", "#e7bd8c", "#c0c9a0", "#c4b7d4"]
INK = "#172b34"

def font(size):
    for name in ("C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()

LABELS = [
    ("Loom threading", ["Comb", "Hook", "Treadle", "Lever"], ["S", "K", "R", "J"]),
    ("Garden beds", ["Thyme", "Sage", "Chive", "Dill"], ["F", "D", "Q", "H"]),
    ("Kiln zones", ["Preheat", "Anneal", "Glaze", "Exhaust"], ["G", "E", "Y", "U"]),
    ("Dock cranes", ["Tug", "Skiff", "Barge", "Cutter"], ["W", "M", "T", "H"]),
    ("Sample drawer", ["Flint", "Marl", "Chalk", "Quartz"], ["N", "P", "C", "V"]),
    ("Roof inlets", ["Tower", "Porch", "Loft", "Atrium"], ["B", "L", "H", "K"]),
    ("Lock controls", ["Paddle", "Hinge", "Winch", "Latch"], ["F", "Q", "G", "E"]),
    ("Terrace bays", ["South", "Upper", "Inner", "Lower"], ["R", "M", "T", "V"]),
]
CLAIMS = [
    ("Spring meter reads", "Mon 08:00", "Mon 18:00", "volume (L)", [12, 19], "Which conclusion is supported for the two Monday readings?", "The evening reading was seven litres higher.", ["Both readings were the same.", "The morning reading exceeded the evening reading.", "Volume increased every hour of the day."]),
    ("Cabinet checks", "Bay A", "Bay B", "sealed jars", [21, 13], "What is established by this cabinet check?", "Eight more sealed jars were counted in Bay A.", ["Bay B held more sealed jars.", "Every jar at both sites was sealed.", "Bay A held twice as many sealed jars as Bay B."]),
    ("Cycle stand count", "Tue 09:00", "Thu 09:00", "parked cycles", [9, 15], "Which claim applies to the two snapshot counts?", "Six more cycles were parked at the Thursday snapshot.", ["Fewer cycles were present on Thursday.", "Each cycle belonged to a different rider.", "Parking increased on Wednesday."]),
    ("Swatch test", "Woven", "Felted", "water retained (g)", [32, 24], "Which claim follows from this one swatch test?", "The woven swatch retained eight grams more.", ["The felted swatch retained the most.", "Every woven fabric always retains more water.", "The test measured evaporation rates."]),
    ("Seed tray tally", "Rack K", "Rack J", "open pods", [17, 10], "What do the observed pod counts support?", "Rack K had seven more open pods at the inspection.", ["Rack J had more open pods.", "All pods at Rack K had opened.", "Rack J contained fewer seeds in total."]),
    ("Gauge survey", "Site R", "Site B", "water depth (cm)", [28, 41], "Which statement matches these two measurements?", "Water at Site B measured thirteen centimetres deeper.", ["Site R was deeper than Site B.", "Site B was deeper at every earlier visit.", "The measurements record stream speed."]),
    ("Archive batch", "Batch V", "Batch M", "folders scanned", [35, 26], "What can the desk log show about these two batches?", "Nine more folders were scanned in Batch V.", ["Batch M had more folders scanned.", "Every folder contained one page.", "Batch V was scanned more quickly per hour."]),
    ("Festival receipts", "First set", "Last set", "tickets collected", [23, 38], "Which conclusion is warranted for the two sets shown?", "Fifteen more tickets were collected during the last set.", ["Only fifteen people came to the last set.", "Tickets were collected at a constant rate.", "The first set yielded more tickets."]),
]

def canvas(title, sub):
    im=Image.new("RGB",(800,470),"#fbfaf4")
    d=ImageDraw.Draw(im)
    d.text((35,20),title,fill=INK,font=font(29))
    d.text((35,66),sub,fill=INK,font=font(18))
    return im,d

def label_image(i, meta):
    title,names,tags=meta
    im,d=canvas(title,"Each part has its own printed inspection code.")
    # Alternate circular, slanted, staggered and vertical diagrams.
    layouts = [[(196,191),(517,166),(251,352),(608,317)],
               [(129,186),(303,246),(476,314),(665,364)],
               [(178,349),(283,195),(488,347),(622,167)],
               [(139,191),(336,344),(528,179),(646,346)]]
    loc=layouts[i%4]
    if i>=4: loc=[(800-x, y+((j%2)*13-6)) for j,(x,y) in enumerate(loc)]
    for j,((x,y),name,tag) in enumerate(zip(loc,names,tags)):
        color=PALETTE[(j+i)%4]
        if i%3==0:
            d.ellipse((x-62,y-46,x+62,y+46),fill=color,outline=INK,width=3)
        elif i%3==1:
            d.rounded_rectangle((x-65,y-40,x+65,y+40),radius=10,fill=color,outline=INK,width=3)
        else:
            d.polygon([(x-74,y),(x,y-48),(x+74,y),(x,y+48)],fill=color,outline=INK,width=2)
        d.text((x,y-10),name,font=font(19),fill=INK,anchor="mm")
        d.text((x,y+17),tag,font=font(24),fill=INK,anchor="mm")
    return im

def claim_image(i,m):
    title,left,right,unit,vals,_,_,_=m
    im,d=canvas(title,unit+"  |  observations recorded at the marked sites or times")
    if i in (1,7):
        for j,(name,val) in enumerate(zip((left,right),vals)):
            x=(165 if i==1 else 310)+j*(370 if i==1 else 220); h=round((195 if i==1 else 230)*val/max(vals))
            d.rectangle((x,367-h,x+115,367),fill=PALETTE[(i+j)%4],outline=INK,width=2)
            d.text((x+58,367-h-18),str(val),font=font(27),anchor="mm",fill=INK)
            d.text((x+58,397),name,font=font(21),anchor="mm",fill=INK)
        d.line((105,369,728,369),fill=INK,width=2)
    elif i in (0,):
        for j,(name,val) in enumerate(zip((left,right),vals)):
            y=139+j*122
            d.rectangle((55,y,727,y+73),fill=PALETTE[(i+j)%4],outline=INK,width=2)
            d.text((90,y+36),name,font=font(24),anchor="lm",fill=INK)
            d.text((665,y+36),str(val),font=font(28),anchor="rm",fill=INK)
    elif i in (2,5):
        for j,(name,val) in enumerate(zip((left,right),vals)):
            x=(120 if i==2 else 255)+j*(400 if i==2 else 310)
            top=(153 if i==2 else 183)+j*(52 if i==2 else -42)
            d.rounded_rectangle((x-74,top,x+118,top+152),radius=20,fill=PALETTE[(i+j)%4],outline=INK,width=3)
            d.text((x+22,top+44),name,font=font(20),anchor="mm",fill=INK)
            d.text((x+22,top+102),str(val),font=font(34),anchor="mm",fill=INK)
    elif i in (3,):
        for j,(name,val) in enumerate(zip((left,right),vals)):
            y=150+j*139
            d.text((45,y),name,font=font(24),fill=INK)
            width=round(385*val/max(vals))
            d.rectangle((194,y,194+width,y+53),fill=PALETTE[(i+j)%4],outline=INK,width=2)
            d.text((206+width,y+26),str(val),font=font(24),anchor="lm",fill=INK)
    elif i==4:
        d.text((49,142),"Observation",fill=INK,font=font(20))
        d.text((49,304),"Count",fill=INK,font=font(20))
        for j,(name,val) in enumerate(zip((left,right),vals)):
            x=219+246*j
            d.rectangle((x,125,x+210,226),fill=PALETTE[(j+i)%4],outline=INK,width=2)
            d.text((x+105,175),name,font=font(21),anchor="mm",fill=INK)
            d.text((x+105,322),str(val),font=font(32),anchor="mm",fill=INK)
    else:
        d.line((90,386,710,386),fill=INK,width=3)
        for j,(name,val) in enumerate(zip((left,right),vals)):
            x=225+295*j
            top=380-int(val*5)
            d.line((x,386,x,top),fill=INK,width=5)
            d.ellipse((x-42,top-42,x+42,top+42),fill=PALETTE[(j+i)%4],outline=INK,width=2)
            d.text((x,top),str(val),fill=INK,font=font(28),anchor="mm")
            d.text((x,413),name,fill=INK,font=font(20),anchor="mm")
    return im

def records():
    for i in range(16):
        pos=(i+1)%4
        if i<8:
            title,names,tags=LABELS[i]
            node=(i*3+1)%4
            opts=[t for t in tags if t!=tags[node]]
            opts.insert(pos,tags[node])
            question=f"Which inspection code is printed on the {names[node]} part in the {title.lower()} diagram?"
            witness=f"The {names[node]} part bears code {tags[node]}; the four part/code pairs are "+", ".join(f"{n}–{t}" for n,t in zip(names,tags))+"."
            rule="Read the code assigned to the requested part and match it to the option."
            im=label_image(i,LABELS[i]); kind="label"
        else:
            j=i-8;m=CLAIMS[j]
            title,left,right,unit,vals,question,right_answer,alts=m
            opts=list(alts);opts.insert(pos,right_answer)
            witness=f"{left} measures {vals[0]}, and {right} measures {vals[1]} ({unit}). {right_answer}"
            rule="Compare the displayed observations, quantities and scope of the measurement."
            im=claim_image(j,m);kind="qualifier"
        b=BytesIO();im.save(b,format="PNG")
        yield dict(id=f"s9-guard-{i+1:02}",kind="retention anchor",source="R9 held-out authored",source_id=f"guard-{i+1:02}",question=question,hint="",options=opts,gold=chr(65+pos),sentinel=True,sentinel_type=kind,reference=dict(visual_witness=witness,rule=rule)),b.getvalue()
