"""Build solver-safe input records independently of answer-bearing judge records."""
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE=Path(__file__).resolve().parent
ROUND=HERE.parent
OLD=ROUND.parent/"era_midstep_round/eval"


def sha(path):
    return hashlib.file_digest(path.open("rb"),"sha256").hexdigest()


def lines(path):
    return list(map(json.loads,path.read_text(encoding="utf8").splitlines()))


def rows(path,records):
    content="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records).encode()
    if path.exists():assert path.read_bytes()==content,path
    else:path.write_bytes(content)


def prompt(q,opts,hint=""):
    hint=("Hint: "+hint+"\n") if hint else ""
    return (hint+"Question: "+q+"\nOptions:\n"+"".join(f"{chr(65+i)}. {v}\n" for i,v in enumerate(opts))+
        'Please select the correct answer from the options above.\nOn the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.')


def main():
    dest=HERE/"spec";dest.mkdir(exist_ok=True)
    ins=[];gold=[];cached=[]
    charts={r["group"]:r["spec"] for r in lines(ROUND/"paired/sources.jsonl")}
    for ds in ("paired","general"):
        for r in lines(ROUND/ds/"items.jsonl"):
            rel=r["image"]
            p=(ROUND/rel).resolve()
            assert p.is_relative_to(ROUND.resolve())
            text=prompt(r["question"],r["options"],r.get("hint") or "")
            ins.append(dict(id=r["id"],dataset=ds,prompt=text,image=rel,image_sha256=sha(p)))
            ref=r["reference"]
            if r.get("witness"):ref=ref|dict(witness=r["witness"])
            if ds=="paired":ref=ref|dict(source_chart=charts[r["group"]])
            gold.append(dict(id=r["id"],dataset=ds,group=r.get("group",r["id"]),kind=r["kind"],
                             question=r["question"],options=r["options"],gold=r["gold"],reference=ref))
    for ds,expected in (("charts","9f236b80e13641f7fa5b4a80eab98241a950591b84f790f36e4254f7bd3e91a6"),
                        ("science","664eeb6ba49e0ad753767bb566c2babee954849c456cfa71721d0013490fdb22")):
        folder=OLD/"spec"/ds;stored=OLD/"runs"/(ds+"-base")/"predictions.jsonl"
        assert sha(stored)==expected
        refs=json.loads((folder/"gold.json").read_text(encoding="utf8"))
        inputs=json.loads((folder/"inputs.json").read_text(encoding="utf8"))
        for r,g,past in zip(inputs,refs,lines(stored),strict=True):
            rid=r["id"];assert rid==g["id"]==past["id"]
            rel="eval/legacy/"+ds+"/"+Path(r["image"]).name
            path=ROUND/rel;path.parent.mkdir(exist_ok=True,parents=True)
            shutil.copyfile(folder/r["image"],path)
            assert sha(path)==r["image_sha256"]==past["image_sha256"]
            old_prompt=past["prompt"]
            opts=r["options"] if ds=="charts" else re.findall(r"(?m)^[A-H]\. (.+)$",old_prompt.split("Options:\n")[-1])
            q=r["question"] if ds=="charts" else old_prompt.split("Question: ",1)[1].split("\nOptions:",1)[0]
            ins.append(dict(id=rid,dataset=ds,prompt=old_prompt,image=rel,image_sha256=sha(path)))
            gold.append(dict(id=rid,dataset=ds,group=g.get("scene",rid),kind=g.get("domain",g.get("skill")),
                             question=q,options=opts,gold=g.get("answer",g.get("option_letter")),reference=g))
            cached.append(past)
    assert len(ins)==len({r["id"] for r in ins})==307 and len(cached)==115
    assert all(chr(65+len(r["options"]))>r["gold"]>="A" for r in gold)
    for name,data in [("inputs",ins),("gold",gold),("cached_base",cached)]:rows(dest/(name+".jsonl"),data)
    receipt={n:sha(dest/(n+".jsonl")) for n in ("inputs","gold","cached_base")}
    (dest/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    print(receipt)


if __name__=="__main__":main()
