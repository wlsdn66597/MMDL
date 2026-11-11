"""Recompute reference answers from the published questions and untokenized source data."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independently_solve(scene,question):
    family=scene["family"];s=scene["spec"];d=s["data"]
    names=re.findall(r"\b(?:"+"|".join(s["labels"])+r")\b",question)
    if family=="ledger":
        if "as closed" in question: return str(d[names[0]+"/Closed"])
        if "as received" in question:return str(d[names[0]+"/Received"])
        if "fraction" in question:return max(names,key=lambda n:Fraction(d[n+"/Closed"],d[n+"/Received"]))
        if "more orders" in question:return max(names,key=lambda n:d[n+"/Closed"])
        return str(sum(d[n+"/Received"]-d[n+"/Closed"] for n in names))
    if family=="trace":
        station=re.findall(r"\b(?:"+"|".join(s["stations"])+r")\b",question)
        trace=[d[names[0]+"/"+t] for t in s["stations"]]
        if "labeled" in question:return str(d[names[0]+"/"+station[0]])
        if "signed" in question:return str(d[names[0]+"/"+station[1]]-d[names[0]+"/"+station[0]])
        return str(sum(abs(q-p) for p,q in zip(trace,trace[1:])))
    if family=="route":
        if "direct arrow" in question:return str(d["/".join(names)])
        if "must pass" in question:
            begin,end,mid=names
            return str(d[begin+"/"+mid]+d[mid+"/"+end])
        m1,m2,start,end=names
        return min((m1,m2),key=lambda mid:d[start+"/"+mid]+d[mid+"/"+end])
    byname=names[0]
    shifts=re.findall(r"\b(?:"+"|".join(s["shifts"])+r")\b",question,re.I)
    values=[d[byname+"/"+v] for v in s["shifts"]]
    if "displayed" in question:return str(d[byname+"/"+shifts[0].title()])
    if "apart" in question:return str(max(values)-min(values))
    if "minus" in question:return str(d[byname+"/"+shifts[0].title()]-d[byname+"/"+shifts[1].title()])
    return str(sum(values))


def main():
    cards=read(HERE/"items.jsonl");scenes=read(HERE/"sources.jsonl")
    assert len(cards)==len({r["id"] for r in cards})==96
    assert len(scenes)==len({s["group"] for s in scenes})==48
    assert len({s["rgb_sha256"] for s in scenes})==48
    transitions={};pics={}
    for s in scenes:
        group=s["group"];a,b=[r for r in cards if r["group"]==group]
        assert a["kind"]==b["kind"]==s["kind"]
        assert a["image"]==b["image"]
        path=HERE.parent/a["image"]
        with Image.open(path) as im:
            assert im.size==(780,540) and im.mode=="RGB"
            assert hashlib.sha256(im.tobytes()).hexdigest()==s["rgb_sha256"]
        pics[group]=sha(path)
        if s["kind"]=="choice":
            assert a["question"]==b["question"] and a["options"]!=b["options"] and set(a["options"])==set(b["options"])
        else:
            assert a["question"]!=b["question"] and a["options"]==b["options"]
        transitions.setdefault(s["kind"],Counter())[(a["gold"],b["gold"])]+=1
        for row in (a,b):
            assert len(set(row["options"]))==4
            value=independently_solve(s,row["question"])
            assert value==row["options"]["ABCD".index(row["gold"])]==row["reference"]["result"],row
            assert row["reference"]["required_values"]==[s["spec"]["data"][k] for k in row["reference"]["source"]]
    target={(a,b):1 for a in "ABCD" for b in "ABCD" if a!=b}
    assert len(transitions)==4 and all(c==target for c in transitions.values())
    assert all(x==3 for x in Counter((s["kind"],s["family"]) for s in scenes).values())
    for start in range(0,48,12):
        panel=Image.new("RGB",(1170,1144),"white");draw=ImageDraw.Draw(panel)
        for idx,s in enumerate(scenes[start:start+12]):
            with Image.open(HERE/"images"/(s["group"]+".png")) as im:
                im.thumbnail((390,270));x=390*(idx%3);y=286*(idx//3);panel.paste(im,(x,y))
            draw.text((x+4,y+270),s["group"]+" "+s["family"],fill="black")
        panel.save(HERE/f"sheet_{start//12}.jpg",quality=88)
    receipt=dict(n_items=96,n_groups=48,n_per_kind=12,families_per_kind={f:3 for f in ("ledger","trace","route","bars")},
                 verified_questions_with_second_oracle=96,per_kind_transition_counts={a+b:n for (a,b),n in sorted(target.items())},
                 items_sha256=sha(HERE/"items.jsonl"),sources_sha256=sha(HERE/"sources.jsonl"),image_sha256=pics,
                 generator_sha256=sha(HERE/"make_pairs.py"),verification_sha256=sha(HERE/"verify.py"))
    (HERE/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    print(receipt["items_sha256"],receipt["sources_sha256"])


if __name__=="__main__":main()
