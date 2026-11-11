"""Answer each published question against its source independently of generator formulas."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent


def read(p):return list(map(json.loads,p.read_text(encoding="utf8").splitlines()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def independently_solve(scene,q):
    s=scene["spec"];names=s["labels"];kind=scene["kind"]
    if kind=="change":
        v=s["values"]
        if "higher" in q or "signed change" in q:
            return str(v[-1]-v[0])
        assert "wheel counter" in q or "mechanism withstand" in q
        up=down=0
        for k in range(1,len(v)):
            amt=v[k]-v[k-1]
            if amt>0:up+=amt
            else:down-=amt
        return str(up+down)
    if kind=="rate":
        if "same number" not in q:return names[s["numerators"].index(max(s["numerators"]))]
        pairs=list(zip(s["numerators"],s["denominators"]))
        leaders=[k for k,(a,b) in enumerate(pairs) if all(a*y>=b*x for x,y in pairs)]
        assert len(leaders)==1
        return names[leaders[0]]
    if kind=="route":
        t=dict(zip(s["edges"],s["times"],strict=True));u,a,b,v=names
        if "without an intermediate" in q or "avoid both" in q:return str(t[u+"/"+v])
        assert "change trains exactly once" in q or "visit one middle checkpoint" in q
        times=[]
        for mid,layover in zip((a,b),s["layovers"],strict=True):
            times.append(t[u+"/"+mid]+layover+t[mid+"/"+v])
        assert times[0]!=times[1]
        return str(min(times))
    if "takes one litre" in q or "equal influence" in q:return str(Fraction(sum(s["percent"]),len(names)))
    assert "full batches" in q or "single inspection lot" in q
    passed=[Fraction(q*p,100) for q,p in zip(s["weights"],s["percent"])]
    return str(100*sum(passed)/sum(s["weights"]))


def main():
    cards=read(HERE/"items.jsonl");scenes=read(HERE/"sources.jsonl")
    assert len(cards)==len({c["id"] for c in cards})==len({(c["image"],c["question"]) for c in cards})
    assert len(scenes)==len({s["rgb_sha256"] for s in scenes})==48 and len(cards)==96
    transitions={};photos={}
    for scene in scenes:
        kind=scene["kind"];g=scene["group"]
        x,y=[c for c in cards if c["group"]==g]
        assert x["options"]==y["options"] and len(set(x["options"]))==4 and x["gold"]!=y["gold"]
        assert x["image"]==y["image"] and x["reference"]["operation"]!=y["reference"]["operation"]
        path=HERE.parent/x["image"]
        with Image.open(path) as im:
            assert im.size==(860,540) and im.mode=="RGB" and hashlib.sha256(im.tobytes()).hexdigest()==scene["rgb_sha256"]
        photos[g]=sha(path)
        transitions.setdefault(kind,Counter())[x["gold"]+y["gold"]]+=1
        for r in (x,y):
            assert independently_solve(scene,r["question"])==r["options"]["ABCD".index(r["gold"])]==r["reference"]["result"]
            ref=r["reference"]
            assert len(ref["required_values"])==len(ref["source"]) and all(k in ref["e"] for k in ref["source"])
    expected={a+b:1 for a in "ABCD" for b in "ABCD" if a!=b}
    assert all(t==expected for t in transitions.values())
    for n in range(0,48,6):
        page=Image.new("RGB",(1290,572),"white");d=ImageDraw.Draw(page)
        for j,scene in enumerate(scenes[n:n+6]):
            x=430*(j%3);y=286*(j//3)
            im=Image.open(HERE/"images"/(scene["group"]+".png"));im.thumbnail((430,270));page.paste(im,(x,y))
            d.text((x+2,y+271),scene["group"]+" "+scene["family"],fill="black")
        page.save(HERE/f"sheet_{n//6}.jpg",quality=90)
    out=dict(n_items=96,n_groups=48,n_per_kind=12,families=dict(Counter(s["family"] for s in scenes)),
             independently_recalculated=96,unique_question_texts=len({c["question"] for c in cards}),transitions_per_kind=expected,items_sha256=sha(HERE/"items.jsonl"),
             sources_sha256=sha(HERE/"sources.jsonl"),image_sha256=photos,generator_sha256=sha(HERE/"make.py"),
             oracle_sha256=sha(Path(__file__)))
    (HERE/"receipt.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
    print(out["items_sha256"],out["sources_sha256"],sha(HERE/"receipt.json"))


if __name__=="__main__":main()
