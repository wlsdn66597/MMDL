"""Original, protected general items. Each drawing and its answer are defined before inference."""
import io
from PIL import Image, ImageDraw, ImageFont

INK="#152e38"; MUTE="#607782"; BLUE="#206b85"; RUST="#ab4837"; WIDTH=780; HEIGHT=490
def font(n,strong=False):
    for name in (("C:/Windows/Fonts/arialbd.ttf" if strong else "C:/Windows/Fonts/arial.ttf"),"DejaVuSans.ttf"):
        try:return ImageFont.truetype(name,n)
        except OSError:pass
    return ImageFont.load_default()
def text(d,xy,s,n=19,fill=INK,anchor="la",strong=False):
    d.text(xy,str(s),font=font(n,strong),fill=fill,anchor=anchor)
def base(title,subtitle):
    im=Image.new("RGB",(WIDTH,HEIGHT),"#fdfdf9"); d=ImageDraw.Draw(im)
    text(d,(32,22),title,29,strong=True);text(d,(33,72),subtitle,17,fill=MUTE)
    return im,d
def dot(d,xy,c,fill=BLUE):
    x,y=xy
    d.ellipse((x-18,y-18,x+18,y+18),fill=fill,outline="white",width=2)
    text(d,(x,y),c,19,fill="white",anchor="mm",strong=True)
def coords_map(title,subtitle,points,landmark,kind):
    im,d=base(title,subtitle)
    if kind=="river":
        d.rectangle((36,117,744,443),fill="#ecead4",outline="#a2ae94",width=2)
        path=[(60,135),(176,169),(215,202),(310,214),(360,281),(480,336),(705,389)]
        d.line(path,fill="#83bdd1",width=18);d.line((542,116,492,219,509,275,674,440),fill="#b59677",width=7)
        d.line((246,402,350,347,389,305,422,236),fill="#99896b",width=5)
    else:
        d.rectangle((36,117,744,443),fill="#c4e4e9",outline="#9bbcc5",width=2)
        d.polygon([(36,117),(575,117),(490,158),(423,185),(418,226),(348,274),(365,332),(289,382),(240,443),(36,443)],fill="#eadebb")
        d.polygon([(534,266),(599,244),(632,264),(616,289),(547,297)],fill="#eadebb")
        d.polygon([(569,348),(633,324),(684,343),(672,385),(610,396)],fill="#eadebb")
    for lab,xy in points.items():dot(d,xy,lab)
    text(d,landmark[:2],landmark[2],16,fill=INK,strong=True)
    return im
def wire(title,nodes,target):
    im,d=base(title,"Round tags identify test points. Component labels are printed beside the wire.")
    if target=="switch":
        y=219; d.line((85,360,85,y,235,y),fill=INK,width=5); d.line((235,y,295,185),fill=INK,width=5)
        d.line((306,y,660,y,660,360,85,360),fill=INK,width=5)
        d.ellipse((460,y-22,504,y+22),fill="#fff7cf",outline=INK,width=3);d.line((467,y-14,497,y+15),fill=INK,width=2);d.line((467,y+14,497,y-15),fill=INK,width=2)
        text(d,(225,150),"switch",17);text(d,(461,165),"lamp",17)
    else:
        d.line((107,190,665,190,665,368,107,368,107,190),fill=INK,width=5)
        d.rectangle((298,174,375,206),fill="#f5cfb5",outline=INK,width=3)
        d.rectangle((460,350,530,385),fill="#d3e3c6",outline=INK,width=3)
        text(d,(302,131),"R1",18,strong=True);text(d,(478,399),"R2",18,strong=True)
    for lab,xy in nodes.items():dot(d,xy,lab,RUST)
    return im
def core(title,subtitle,labs,layers):
    im,d=base(title,subtitle)
    for j,(name,color) in enumerate(layers):
        high=119+j*79;d.rectangle((100,high,705,high+77),fill=color,outline="#586e6e",width=1)
        text(d,(130,high+38),name,21,anchor="lm",strong=True)
        dot(d,(500+(j%2)*68,high+38),labs[j])
    return im
def layout(title,subtitle,rooms):
    im,d=base(title,subtitle)
    for name,lab,coords,color in rooms:
        d.rectangle(coords,fill=color,outline="#435862",width=4)
        x0,y0,x1,y1=coords;text(d,((x0+x1)//2,(y0+y1)//2-18),name,20,anchor="mm",strong=True);dot(d,((x0+x1)//2,(y0+y1)//2+24),lab)
    return im
def encoded(im):
    b=io.BytesIO();im.save(b,format="PNG");return b.getvalue()
def newitem(i,kind,question,opts,gold,witness,rule,image):
    return dict(id=f"s8-guard-{i:02}",group=f"s8-guard-{i:02}",kind="retention anchor",source="R8 authored held-out",
                source_id=f"guard-{i:02}",question=question,hint="",options=opts,gold=gold,sentinel=True,sentinel_type=kind,
                reference=dict(visual_witness=witness,rule=rule,application=f"{witness} Therefore choose option {gold}: {opts[ord(gold)-65]}.")), encoded(image)

def label_items():
    rule="Find the requested object in the drawing, read its marker, then match the marker to the content of an option."
    pts={"A":(177,327),"D":(298,193),"B":(565,337),"C":(611,191)}
    im=coords_map("Hollow Brook trail map","Look for the tag next to the named trail stop.",pts,(275,147,"Ferry"),"river")
    yield newitem(1,"label","Which tag marks the Ferry stop?",["D","B","C","A"],"A","The Ferry label is next to map tag D.",rule,im)
    pts={"D":(198,204),"B":(359,282),"A":(580,270),"C":(628,365)}
    im=coords_map("Windward islands","Island points and mainland stations are labeled separately.",pts,(543,214,"North Isle"),"isle")
    yield newitem(2,"label","Which site marker sits on the island labeled North Isle?",["C","A","D","B"],"B","North Isle contains marker A.",rule,im)
    pts={"C":(138,219),"B":(372,219),"A":(610,219),"D":(312,360)}
    yield newitem(3,"label","Which tag is on the wire between the open switch and the lamp?",["D","C","B","A"],"C","Marker B lies on the upper wire between the switch and lamp.",rule,wire("Courtyard lamp circuit",pts,"switch"))
    pts={"D":(177,190),"C":(423,190),"A":(581,368),"B":(290,368)}
    yield newitem(4,"label","Which tag lies on the upper wire just to the right of R1?",["B","D","A","C"],"D","On the upper wire, marker C is immediately right of R1.",rule,wire("Two-resistor test board",pts,"resistor"))
    im=core("North face · core sample","The shallowest layer is at the top.",["B","A","C","D"],[("Clay","#d2b087"),("Limestone","#d1d9bd"),("Sandstone","#c98065"),("Shale","#acbdc5")])
    yield newitem(5,"label","What is the marker in the sandstone layer?",["C","B","D","A"],"A","The sandstone band contains tag C.",rule,im)
    im=core("Coastal transect","Surface to base. Each band names the sampled deposit.",["D","B","A","C"],[("Peat","#bdac91"),("Fine sand","#f0d8a9"),("Pebbles","#bdd5d7"),("Bedrock","#d8b5ad")])
    yield newitem(6,"label","Which label identifies the fine sand in the coastal transect?",["A","B","D","C"],"B","The fine sand band contains tag B.",rule,im)
    rooms=[("Gallery","A",(50,125,378,342),"#dfeadb"),("Print room","D",(378,125,730,286),"#cadddf"),("Stores","C",(378,286,730,432),"#ead6b9"),("Hall","B",(50,342,378,432),"#ece5ce")]
    yield newitem(7,"label","Which drawing tag is inside the Print room?",["C","B","D","A"],"C","Print room in the upper right contains marker D.",rule,layout("Workshop · first floor","Plan symbols indicate inspection positions.",rooms))
    rooms=[("Cool room","B",(50,132,315,430),"#c4ddec"),("Lab","C",(315,132,730,275),"#e7ddbb"),("Office","A",(315,275,535,430),"#dce6d5"),("Bay","D",(535,275,730,430),"#dfc7c2")]
    yield newitem(8,"label","Which monitoring symbol is located in the Bay?",["B","A","C","D"],"D","The lower-right Bay contains tag D.",rule,layout("Field station rooms","Each monitoring symbol is printed in its assigned room.",rooms))

EVIDENCE=[
 ("Cliff watch","Counts during two observation windows","table",[("09:00 · upper ledge",7),("09:00 · lower ledge",4),("16:00 · upper ledge",3),("16:00 · lower ledge",6)],"birds",
  "Which statement is supported by the counts?",["More birds were counted on the upper ledge at 09:00.","More birds were counted on the upper ledge at 16:00.","The lower ledge count at 09:00 was six.","The counts show that a change in temperature drove birds downhill."],"A","At 09:00 the upper ledge count is 7 and the lower count 4."),
 ("Workshop trial","Three test panels under the same lamp","bars",[("Matte",12),("Satin",19),("Gloss",16)],"light reading",
  "Which claim agrees with the recorded panel readings?",["The gloss reading was the lowest.","Satin had the largest reading in this trial.","Matte exceeded satin by seven units.","The lamp gave all three panels the same reading."],"B","Satin reads 19, above matte 12 and gloss 16 in this single trial."),
 ("Basin samples","A single afternoon sampling visit","dots",[("North",5),("East",9),("West",7)],"visible shells",
  "What do the samples establish for that visit?",["The north sample had more shells than the west sample.","The east sample had the fewest shells.","The east sample contained more shells than the north sample.","The shell count doubled from north to east."],"C","East has 9 visible shells, north 5, and west 7."),
 ("Train desk tally","Tickets checked on Tuesday morning","table",[("Hill · before 09:00",18),("Vale · before 09:00",22),("Hill · after 09:00",29),("Vale · after 09:00",17)],"tickets",
  "Which statement describes the observations shown?",["Vale had 29 tickets checked after 09:00.","Hill had fewer tickets checked after 09:00 than before.","Vale exceeded Hill in both periods.","Hill exceeded Vale after 09:00 on Tuesday."],"D","After 09:00 Hill has 29 tickets to Vale's 17; the table covers Tuesday morning."),
 ("Enclosure survey","Flowers observed in one square metre on May 6","dots",[("Shade",6),("Edge",10),("Open",8)],"open flowers",
  "Which conclusion is supported for these sampled plots?",["The Edge plot had the most open flowers in this survey.","The Shade plot had ten open flowers.","Open ground prevented flowers from opening.","All three plots had at least eight open flowers."],"A","The Edge sample is 10, while Shade is 6 and Open is 8."),
 ("Reservoir log","Weekly staff-gauge readings","line",[("Week 1",8),("Week 2",5),("Week 3",9),("Week 4",6)],"gauge marks",
  "Which claim can be checked from this four-week log?",["The gauge rose every week.","Week 2 had the lowest recorded gauge reading.","Week 4 had the highest recorded gauge reading.","The log shows why the gauge fell in week 2."],"B","Week 2 reads 5, lower than 8, 9 and 6. The log records levels only."),
 ("Night watch","Microphones at three sites, 21:00–22:00","bars",[("Birch",11),("Alder",16),("Ash",9)],"calls recorded",
  "Which statement is supported for the recorded hour?",["Alder had fewer calls than Birch.","Ash had 11 calls.","More calls were recorded at Alder than at either other site.","Alder necessarily had more individual animals."],"C","Alder has 16 calls, Birch 11 and Ash 9; calls are not counts of individual animals."),
 ("Greenhouse bench","One reading at noon and one at 16:00","line",[("Noon · left",6),("Noon · right",9),("16:00 · left",10),("16:00 · right",7)],"moisture units",
  "What can be concluded from the two time points?",["The right sensor increased from noon to 16:00.","The left sensor was wetter at noon than at 16:00.","The data prove that sunlight caused the difference.","At 16:00 the left sensor read higher than the right."],"D","At 16:00 the left reading is 10 and the right reading is 7.")
]
def observations(title,sub,kind,pairs,unit):
    im,d=base(title,sub)
    if kind=="table":
        for j,(label,num) in enumerate(pairs):
            y=132+72*j
            d.rectangle((60,y,702,y+59),fill="#dbe8eb" if j%2==0 else "#f2e8d4")
            text(d,(80,y+29),label,22,anchor="lm");text(d,(655,y+29),f"{num} {unit}",22,anchor="rm",strong=True)
    elif kind=="dots":
        for j,(label,num) in enumerate(pairs):
            y=157+j*91
            text(d,(58,y),label,19,anchor="lm")
            for k in range(num):
                x=211+45*k;d.ellipse((x-11,y-11,x+11,y+11),fill=[BLUE,RUST,"#4a7c61"][j])
            text(d,(706,y),num,18,anchor="rm")
        text(d,(60,425),f"Each dot represents one of the {unit}.",16,fill=MUTE)
    elif kind=="bars":
        for j,(label,num) in enumerate(pairs):
            y=143+j*90
            text(d,(58,y+25),label,20,anchor="lm")
            x=225+int(409*num/max(n for _,n in pairs));d.rectangle((225,y,x,y+49),fill=[BLUE,RUST,"#53826e"][j])
            text(d,(x+10,y+25),num,21,anchor="lm")
        text(d,(225,425),unit,18,fill=MUTE)
    else:
        ps=[(114+j*172,377-int(203*num/max(n for _,n in pairs))) for j,(_,num) in enumerate(pairs)]
        d.line((75,134,75,382,729,382),fill=MUTE,width=3)
        if title.startswith("Greenhouse"):
            for i in range(2):d.line([ps[i],ps[i+2]],fill=[BLUE,RUST][i],width=4)
        else:d.line(ps,fill=BLUE,width=4)
        for (x,y),(label,num) in zip(ps,pairs):
            d.ellipse((x-6,y-6,x+6,y+6),fill=RUST);text(d,(x,y-13),num,19,anchor="mb")
            text(d,(x,397),label,16,anchor="ma")
        text(d,(75,110),unit,16,fill=MUTE)
    return im
def records():
    yield from label_items()
    for i,(title,sub,kind,vals,unit,q,opts,gold,witness) in enumerate(EVIDENCE,9):
        yield newitem(i,"qualifier",q,opts,gold,witness,
                      "Test each option against the recorded measurements and the indicated time or sampling scope.",
                      observations(title,sub,kind,vals,unit))
