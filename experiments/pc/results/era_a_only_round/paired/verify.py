"""Answer every released question from the printed source fields, without generator helpers."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import hashlib
import json

from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent


def rows(file):
    return [json.loads(s) for s in file.read_text(encoding="utf8").splitlines() if s]


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def solve(kind,s,question):
    """Select operations from question text; do not use generated answer, key or witness."""
    q=question.lower()
    if kind=="change":
        coords=s["values"]
        if "signed" in q:
            return "endpoint change",str(coords[-1]-coords[0])
        assert any(word in q for word in ("four segments","four moves","every segment")),q
        gain=sum(max(0,b-a) for a,b in zip(coords,coords[1:]))
        loss=sum(max(0,a-b) for a,b in zip(coords,coords[1:]))
        return "distance along path",str(gain+loss)
    if kind=="mixture":
        if any(key in q for key in ("specimen","portion","a sample contains")):
            op="equal contributions"
            masses=[1]*len(s["labels"])
        else:
            assert any(word in q for word in ("all material","in full","entire inventory")),q
            op="pooled composition"
            masses=s["weights"]
        grams=sum(Fraction(m*p,100) for m,p in zip(masses,s["percent"]))
        return op,str(100*grams/sum(masses))
    if kind=="rate":
        low=any(word in q for word in ("smallest","fewest","lowest"))
        high=any(word in q for word in ("greatest","highest","most","best"))
        assert low!=high,q
        normalized=any(word in q for word in ("equal","per 100"))
        denominator=s["denominators"] if normalized else [1]*4
        candidates=list(zip(s["numerators"],denominator))
        # Compare n/d with a/b by cross multiplication; no decimal approximation.
        leaders=[i for i,(n,d) in enumerate(candidates)
                 if all((n*b<=a*d if low else n*b>=a*d) for a,b in candidates)]
        assert len(leaders)==1
        return ("per-opportunity comparison" if normalized else "observed count"),s["labels"][leaders[0]]
    assert kind=="route"
    origin,a,b,dest=s["labels"]
    travel=dict(zip(s["edges"],s["times"]))
    if any(phrase in q for phrase in ("single link","uninterrupted direct","bypasses both")):
        return "no intermediate stop",str(travel[f"{origin}/{dest}"])
    assert any(phrase in q for phrase in ("stop once","exactly one intermediate","two allowed one-stop")),q
    alternatives=[]
    for node,service in zip((a,b),s["layovers"]):
        alternatives.append(travel[f"{origin}/{node}"]+service+travel[f"{node}/{dest}"])
    assert alternatives[0]!=alternatives[1]
    return "minimum one-stop elapsed time",str(min(alternatives))


def contact_sheets(scenes):
    for start in range(0,48,6):
        page=Image.new("RGB",(1380,604),"white")
        d=ImageDraw.Draw(page)
        for idx,scene in enumerate(scenes[start:start+6]):
            x=(idx%3)*460; y=(idx//3)*302
            with Image.open(ROOT/"images"/(scene["group"]+".png")) as img:
                img.thumbnail((460,280));page.paste(img,(x,y))
            d.text((x+9,y+279),scene["group"]+"  /  "+scene["family"],fill="black",
                   font=ImageFont.load_default(size=15))
        page.save(ROOT/f"sheet_{start//6}.jpg",quality=92)


def main():
    cards=rows(ROOT/"items.jsonl");scenes=rows(ROOT/"sources.jsonl")
    assert len(cards)==len({r["id"] for r in cards})==len({r["question"] for r in cards})==96
    assert len(scenes)==len({r["group"] for r in scenes})==len({r["rgb_sha256"] for r in scenes})==48
    assert Counter(r["kind"] for r in scenes)==dict.fromkeys(["change","rate","mixture","route"],12)
    assert set(Counter(r["family"] for r in scenes).values())=={6}
    option_pairs={}; photos={};operations={}
    for s in scenes:
        g=s["group"];pair=[r for r in cards if r["group"]==g]
        assert len(pair)==2
        first,second=pair
        assert first["image"]==second["image"] and first["options"]==second["options"]
        assert len(set(first["options"]))==4 and first["gold"]!=second["gold"]
        file=ROOT.parent/first["image"]
        with Image.open(file) as img:
            assert img.size==(920,560) and img.mode=="RGB"
            assert hashlib.sha256(img.tobytes()).hexdigest()==s["rgb_sha256"]
        photos[g]=digest(file)
        option_pairs.setdefault(s["kind"],Counter())[first["gold"]+second["gold"]]+=1
        for item in pair:
            op,result=solve(s["kind"],s["spec"],item["question"])
            ref=item["reference"]
            assert op==ref["operation"] and result==ref["result"]==item["options"]["ABCD".index(item["gold"])]
            assert all(k in ref["e"] for k in ref["source"]) and len(ref["source"])==len(ref["required_values"])
            operations.setdefault(s["kind"],Counter())[op]+=1
    expected={a+b:1 for a in "ABCD" for b in "ABCD" if a!=b}
    assert all(d==expected for d in option_pairs.values())
    assert all(sorted(d.values())==[12,12] for d in operations.values())
    contact_sheets(scenes)
    result=dict(n_items=96,n_groups=48,n_per_kind=12,independently_recalculated=96,unique_question_texts=96,
                families=dict(Counter(x["family"] for x in scenes)),operations_per_kind=operations,
                transitions_per_kind=expected,items_sha256=digest(ROOT/"items.jsonl"),
                sources_sha256=digest(ROOT/"sources.jsonl"),image_sha256=photos,
                generator_sha256=digest(ROOT/"make.py"),oracle_sha256=digest(Path(__file__)))
    (ROOT/"receipt.json").write_bytes((json.dumps(result,indent=2)+"\n").encode())
    print(result["items_sha256"],result["sources_sha256"],digest(ROOT/"receipt.json"))


if __name__=="__main__":
    main()
