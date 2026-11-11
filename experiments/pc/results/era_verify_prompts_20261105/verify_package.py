"""Cross-check the paired scores and package the local viewer with its original 48 charts."""
import csv
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
import json
import re
import zipfile

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/"era_row_position_20261105"

def main():
    records=json.loads((OLD/"items.json").read_text(encoding="utf8"))
    comparisons=json.loads((ROOT/"comparisons.json").read_text(encoding="utf8"))
    checks={r["id"]:r for r in comparisons}
    # Re-derive exact chart answers from row values, independently of stored gold.
    for item in records:
        for cell in item["cells"].values():
            vals=[Decimal(x) for x in cell["rows"].values()]
            gold=sum(sorted(vals)[:3]);r=checks[cell["id"]]
            assert Decimal(r["truth"]["total"])==gold and Decimal(r["bottom"]["total"])==sum(vals[-3:])
    for arm,path in (("baseline",OLD/"run"),("generic",ROOT/"generic"),("boundary",ROOT/"boundary")):
        pred={r["id"]:r for r in map(json.loads,(path/"predictions.jsonl").read_text(encoding="utf8").splitlines())}
        correct=set(); assert len(pred)==48
        for chart in records:
            for cell in chart["cells"].values():
                rid=cell["id"];raw=pred[rid]["raw"]
                fields=re.findall(r"(?im)Final answer:[ \t]*([^\r\n]*)",raw)
                choice=re.match(r"\s*\**\s*([A-D])\b",fields[-1],re.I) if fields else None
                if choice and pred[rid]["finish_reason"]=="stop" and Decimal(chart["options"]["ABCD".index(choice[1].upper())])==sum(sorted(map(Decimal,cell["rows"].values()))[:3]):
                    correct.add(rid)
        assert correct=={rid for rid,r in checks.items() if r[arm]["correct"]}
        if arm=="baseline": base_correct=correct
        else:
            assert correct-base_correct=={rid for rid,r in checks.items() if r[arm]["transition"]=="recovered"}
            assert base_correct-correct=={rid for rid,r in checks.items() if r[arm]["transition"]=="lost"}
    class Media(HTMLParser):
        def handle_starttag(self,tag,attrs):
            if tag=="img":
                assert (ROOT/dict(attrs)["src"]).is_file()
    Media().feed((ROOT/"review.html").read_text(encoding="utf8"))
    assert len(list(csv.DictReader((ROOT/"comparisons.csv").open(encoding="utf-8-sig"))))==48
    archive=ROOT.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted(ROOT.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:z.write(f,f.relative_to(ROOT.parent))
        old_files=[*sorted((OLD/"images").glob("*.png")),*[OLD/name for name in (
            "inputs.json","items.json","reviews.json","eval_config.json","run_eval.py","run/manifest.json","run/predictions.jsonl")]]
        for f in old_files:z.write(f,f.relative_to(ROOT.parent))
    with zipfile.ZipFile(archive) as z: assert z.testzip() is None
    print("Independent paired transition, chart arithmetic, HTML media and archive checks passed.")

if __name__=="__main__": main()
