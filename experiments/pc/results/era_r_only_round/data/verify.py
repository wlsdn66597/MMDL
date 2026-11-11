"""Check fixed targets and independently solve the synthetic training pairs."""
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image,ImageOps,ImageDraw

HERE=Path(__file__).resolve().parent
ROUND=HERE.parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/".runtime/tokenizers"))
from tokenizers import Tokenizer


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(map(json.loads,p.read_text(encoding="utf8").splitlines()))


def oracle(row,source,ordinal):
    data=source["source"]
    f=source["family"]
    if f=="signal":
        vs=data["values"]
        return str(vs[-1]-vs[0] if ordinal==0 else sum(abs(vs[i+1]-v) for i,v in enumerate(vs[:-1])))
    if f=="register":
        first,second=data["data"][:2]
        calc=lambda z: z[1] if ordinal==0 else Fraction(z[1],z[2])
        return max([first,second],key=calc)[0]
    if f=="circuit":
        names=data["nodes"];links={(names[a],names[b]):v for a,b,v in data["edges"]}
        src,dst=names[0],names[-1]
        if ordinal==0:return str(links[src,dst])
        mid=[n for n in names[1:3] if n in row["question"]]
        assert len(mid)==1
        return str(links[src,mid[0]]+links[mid[0],dst])
    if f=="locator":
        target=[i for i,n in enumerate(data["objects"]) if n.lower() in row["question"].lower()]
        assert len(target)==1
        return data["marks"][target[0]]
    assert f=="photo"
    return source["oracle"][ordinal]["answer"]


def main():
    src={r["group"]:r for r in json.loads((HERE/"sources.json").read_text(encoding="utf8"))}
    allrows=rows(HERE/"rows.jsonl")
    uni=defaultdict(list);changes=[];positions=Counter()
    tokenizer=Tokenizer.from_file(str(ROOT/".runtime/qwen_tokenizer/tokenizer.json"))
    tok=defaultdict(lambda:[0,0])
    for r in allrows:
        s=src[r["group"]]
        assert sha(ROUND/r["image"])==s["image_sha256"]==r["image_sha256"]
        assert r["kind"]==s["kind"] and r["family"]==s["family"]
        ord_=len(uni[r["group"]]);uni[r["group"]].append(r)
        assert ord_<2
        truth=oracle(r,s,ord_);assert r["options"][ord(r["gold"])-65]==truth==(s["oracle"][ord_]["answer"])
        x,y=r["ordinary"],r["enhanced"]
        assert x["e"]==y["e"] and x["a"]==y["a"]
        assert (x["r"]!=y["r"])==(r["kind"]=="R")
        if x["r"]!=y["r"]:
            changes.append(r["id"])
            for i,t in enumerate([x,y]):
                text=f"Evidence: {t['e']}\nRule: {t['r']}\nApplication: {t['a']}\nFinal answer: {r['gold']}"
                n=len(tokenizer.encode(text,add_special_tokens=False).ids)+1
                tok["changed"][i]+=n;tok[r["family"]][i]+=n
        positions[r["gold"]]+=1
    for k,g in uni.items():
        assert len(g)==2 and g[0]["image"]==g[1]["image"]
        if g[0]["kind"]=="R":
            assert g[0]["options"]==g[1]["options"] and g[0]["gold"]!=g[1]["gold"]
        if g[0]["kind"]=="SYMBOL":
            assert g[0]["gold"]!=g[1]["gold"] and g[0]["options"][ord(g[0]["gold"])-65]==g[1]["options"][ord(g[1]["gold"])-65]
    assert len(src)==len(uni)==24 and len(allrows)==len({r["id"] for r in allrows})==48 and len(changes)==24
    assert sorted(positions.values())==[12]*4
    print("changed supervision estimates",dict(tok))
    images=[ROUND/gr[0]["image"] for gr in uni.values()]
    assert len(set(sha(p) for p in images))==24
    for offset in range(0,24,6):
        canvas=Image.new("RGB",(1230,3*265),"white");d=ImageDraw.Draw(canvas)
        for i,path in enumerate(images[offset:offset+6]):
            x=(i%2)*615;y=(i//2)*265
            im=ImageOps.contain(Image.open(path).convert("RGB"),(599,235));canvas.paste(im,(x+(615-im.width)//2,y+23));d.text((x+8,y+4),path.stem,fill="black")
        canvas.save(HERE/f"sheet_{offset//6}.jpg",quality=88)
    out=dict(n=48,groups=24,kinds=dict(Counter(s["kind"] for s in src.values())),changed=changes,positions=dict(positions),
             application_and_evidence_fixed=True,solver_answer_fixed=True,options_identical_within_R_group=True,
             rows_sha256=sha(HERE/"rows.jsonl"),sources_sha256=sha(HERE/"sources.json"),token_estimates=dict(tok),
             scienceqa_train_sha256=sha(ROOT/"tmp/era_prereview_20261021/scienceqa_train.parquet"),
             verified_math_and_structure=36,verified_image_observations=12)
    (HERE/"receipt.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(out["rows_sha256"],out["kinds"])


if __name__=="__main__":main()
