"""Independent data, training identity, paired scoring, report links and ZIP verification."""
import csv
from decimal import Decimal
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding="utf8"))
def rows(p):return list(map(json.loads,p.read_text(encoding="utf8").splitlines()))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def dataset():
    data=ROOT/"data"; src=ROOT.parent/"d_era_extension_20261105"
    original={r["id"]:r for r in rows(src/"data/d_era.jsonl")}
    normal,checked=[rows(data/f) for f in ("standard.jsonl","checked.jsonl")]
    assert len(normal)==len(checked)==len(original)==226
    changed=set()
    for a,b in zip(normal,checked):
        assert all(a[k]==b[k]==original[a["id"]][k] for k in ("id","images","input_prompt","answer"))
        assert a["rule"]==b["rule"] and a["application"]==b["application"] and a["target"].splitlines()[1:]==b["target"].splitlines()[1:]
        assert "?" not in a["target"]
        if a["target"]!=b["target"]:
            assert b["evidence"].count("?")==1 and b["evidence"].split("?",1)[1].strip() in a["evidence"]
            changed.add(a["id"])
        else:assert a==b
        for f in a["images"]:assert sha(data/f)==sha(src/f)
    assert changed=={j["id"] for j in rows(data/"selection.jsonl")} and len(changed)==47
    path=ROOT.parent/"d226_echeck_training_data_20261105.zip"
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(data.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:z.write(f,f.relative_to(data))
        z.write(src/"LICENSES.md","LICENSES.md")
        for f in (src/"credits").glob("*"):
            if f.is_file():z.write(f,"credits/"+f.name)
    with zipfile.ZipFile(path) as z:assert z.testzip() is None and z.read("checked.jsonl")== (data/"checked.jsonl").read_bytes()
    return changed

def results():
    got=read(ROOT/"summary.json")
    manifests=[read(ROOT/"train"/a/"manifest.json") for a in ("standard","checked")]
    m,n=manifests
    assert m["initial_adapter"]==n["initial_adapter"] and m["sequence_sha256"]==n["sequence_sha256"] and m["identity"]==n["identity"]
    assert m["final_adapter"]!=n["final_adapter"] and m["actual_steps"]==n["actual_steps"]==45
    sr=read(ROOT/"server_receipt.json")
    for rel,h in sr["files"].items():assert sha(ROOT/rel)==h,rel
    items=read(ROOT/"fresh/items.json")
    base=None
    for a in ("baseline","standard","checked"):
        outputs={r["id"]:r for r in rows(ROOT/"eval"/a/"predictions.jsonl")};true=set()
        for it in items:
            for c in it["cells"].values():
                rid=c["id"];r=outputs[rid]
                if r["finish_reason"]!="stop":continue
                nums=sorted(map(Decimal,c["rows"].values()));ans=sum(nums[:3])
                parts=re.findall(r"(?im)\bfinal\s+answer\s*:[ \t]*([^\r\n]*)",r["raw"])
                field=parts[-1].strip().strip("*").strip() if parts else r["raw"].splitlines()[-1].strip()
                letter=re.match(r"(?i)^(?:option\s+)?[\[(]?([A-D])\b",field)
                if not parts and letter is None:
                    letter=re.search(r"(?i)(?:the answer is|select(?:ing)?|match(?:es|ing) option)\s+(?:option\s+)?([A-D])\b[ .!*)]*$",field)
                if letter and Decimal(it["options"]["ABCD".index(letter[1].upper())])==ans:true.add(rid)
                elif re.fullmatch(r"[-+]?\d+(?:\.\d+)?",field) and Decimal(field)==ans:true.add(rid)
        assert len(true)==got[a]["correct"]
        if a=="baseline":base=true
        else:
            assert true-base==set(got[a]["vs_baseline"]["recovered"])
            assert base-true==set(got[a]["vs_baseline"]["lost"])
            assert base & true==set(got[a]["vs_baseline"]["kept"])
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            if tag=="img":
                assert (ROOT/dict(attrs)["src"]).is_file()
    Links().feed((ROOT/"review.html").read_text(encoding="utf8"))
    assert len(list(csv.DictReader((ROOT/"comparisons.csv").open(encoding="utf-8-sig"))))==64
    print("Original prompts/images, 226 matched examples, identical initialization and sample order, server artifacts, all three score sets and paired transitions verified.")
if __name__=="__main__":
    import sys
    dataset()
    if "--dataset-only" not in sys.argv:results()
