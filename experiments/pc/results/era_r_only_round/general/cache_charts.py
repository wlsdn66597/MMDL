"""Cache a reproducible subset of the existing native ChartQA validation revision."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request,urlopen
import hashlib,json,re

ROOT=Path(__file__).resolve().parents[3]
CACHE=ROOT/"tmp/er_r8_chartqa"
REV="044eabfc306abfe9340c5741f0093aefc5973d06"
BASE=f"https://raw.githubusercontent.com/vis-nlp/ChartQA/{REV}/ChartQA%20Dataset/val/"
def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    raw=json.loads((ROOT/"tmp/er_r7_sources/chart-val_human.json").read_text(encoding="utf8"))
    first={}
    for r in raw:first.setdefault(r["imgname"],r)
    eligible=[fn for fn,r in first.items() if re.fullmatch(r"\d+(?:\.\d+)?",str(r["label"])) and
              not (ROOT/"tmp/er_r7_sources"/("chart-png_"+fn)).exists()]
    chosen=sorted(eligible,key=lambda fn:hashlib.sha256(("r8-extra-chart-native/"+fn).encode()).digest())[:55]
    def get(fn):
        for rel in ("png/"+fn,"tables/"+Path(fn).stem+".csv"):
            target=CACHE/("chart-"+rel.replace("/","_"))
            if not target.is_file():
                data=urlopen(Request(BASE+rel,headers={"User-Agent":"research-evaluation"}),timeout=18).read()
                target.write_bytes(data)
    with ThreadPoolExecutor(max_workers=8) as exe:list(exe.map(get,chosen))
    print("Verified fixed candidate URLs and cached",len(chosen),"image/table pairs.")
if __name__=="__main__":main()
