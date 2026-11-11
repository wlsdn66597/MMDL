"""Freeze a strict solver view and a separate answer-bearing reviewer view."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def lines(p):
    return [json.loads(line) for line in p.read_text(encoding="utf8").splitlines() if line]


def sealed(p,raw):
    if p.exists():assert p.read_bytes()==raw,p
    else:p.write_bytes(raw)


def prompt(q,opts,hint):
    lead=f"Hint: {hint.strip()}\n" if hint else ""
    choices="".join(f"{chr(65+i)}. {o}\n" for i,o in enumerate(opts))
    return lead+f'Question: {q}\nOptions:\n'+choices+'Please select the correct answer from the options above.\nOn the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'


def main():
    dst=HERE/"spec";dst.mkdir(parents=True,exist_ok=True)
    x=[];y=[]
    tables={r["group"]:r for r in lines(ROOT/"paired/sources.jsonl")}
    for ds in ("paired","general"):
        rs=lines(ROOT/ds/"items.jsonl")
        assert len(rs)==96
        for r in rs:
            f=(ROOT/r["image"]).resolve()
            assert f.is_file() and f.is_relative_to(ROOT.resolve()) and not Path(r["image"]).is_absolute()
            opts=r["options"];assert 2<=len(opts)<=8 and len(set(opts))==len(opts)
            assert r["gold"] in [chr(65+i) for i in range(len(opts))]
            ref=r["reference"];assert isinstance(ref,dict)
            if ds=="paired":
                entry=tables[r["group"]]
                ref=ref|dict(witness=r.get("witness"),chart=entry.get("spec",entry))
            x.append(dict(id=r["id"],dataset=ds,prompt=prompt(r["question"],opts,r.get("hint")),image=r["image"],image_sha256=sha(f)))
            y.append(dict(id=r["id"],dataset=ds,group=r.get("group",r["id"]),kind=r["kind"],sentinel=bool(r.get("sentinel")),
                          sentinel_type=r.get("sentinel_type"),question=r["question"],options=opts,gold=r["gold"],reference=ref))
    assert len({r["id"] for r in x})==192
    assert len({r["image_sha256"] for r in x if r["dataset"]=="paired"})==48
    assert sum(r["sentinel"] for r in y)==16
    for name,rows in (("inputs",x),("gold",y)):
        sealed(dst/(name+".jsonl"),"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode())
    receipt={n:sha(dst/(n+".jsonl")) for n in ("inputs","gold")}
    sealed(dst/"receipt.json",(json.dumps(receipt,indent=2)+"\n").encode())
    print(receipt)


if __name__=="__main__":main()
