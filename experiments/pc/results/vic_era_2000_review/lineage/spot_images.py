"""Check fixed hash-sampled archive pictures against the public viewer for the same pinned row."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
from urllib.parse import urlencode, urlsplit
from urllib.request import Request,urlopen
from zipfile import ZipFile

from PIL import Image
from PIL import ImageOps
import numpy as np

from check_source import HERE, MM_REPO, MM_REV, OM_REPO, OM_REV, norm


def fetch(url):
    with urlopen(Request(url,headers={"User-Agent":"project-dataset-provenance/1"}),timeout=14) as r:
        return r.read()


def compare(args):
    row,m,im=args
    repo,rev=(MM_REPO,MM_REV) if m["split"]=="sft" else (OM_REPO,OM_REV)
    offs=m.get("global_row",m["row"])
    target="https://datasets-server.huggingface.co/rows?"+urlencode(dict(dataset=repo,config="default",split=m["split"],offset=offs,length=1))
    out=dict(id=row["id"],vic_type=row["vic_type"],split=m["split"],row=offs)
    try:
        ref=json.loads(fetch(target))["rows"][0]
        assert ref["row_idx"]==offs and norm(ref["row"]["question"])==norm(row["question"])
        remote=ref["row"]["image"]["src"]
        assert urlsplit(remote).hostname=="datasets-server.huggingface.co" and "/"+rev+"/" in remote
        ro=Image.open(BytesIO(fetch(remote))).convert("RGB")
        lo=Image.open(BytesIO(im)).convert("RGB")
        out.update(original_wh=ro.size,release_wh=lo.size,same_dimensions=lo.size==ro.size)
        rs=ImageOps.contain(ro.copy(),(160,160));ls=ImageOps.contain(lo.copy(),(160,160))
        if rs.size==ls.size:
            err=float(np.abs(np.asarray(rs,dtype=float)-np.asarray(ls,dtype=float)).mean())
            out.update(rgb_equal=ro.size==lo.size and ro.tobytes()==lo.tobytes(),thumbnail_mae_on_255=round(err,3))
    except Exception as err:
        out["error"]=str(err) or type(err).__name__
    return out


def main():
    with ZipFile(Path(sys.argv[1])) as z:
        rows={r["id"]:r for r in json.loads(z.read("data.json"))}
        matches={r["zip_id"]:r for r in json.loads((HERE/"upstream.json").read_text(encoding="utf8"))["matches"]}
        bins={}
        for rid,r in rows.items():
            bins.setdefault((matches[rid]["split"],r["vic_type"]),[]).append(rid)
        sample=[]
        for group in bins.values():
            group.sort(key=lambda rid:hashlib.sha256(("vic-image-link-audit/"+rid).encode()).digest())
            sample.extend((rows[rid],matches[rid],z.read(rows[rid]["image_1"])) for rid in group[:4])
    with ThreadPoolExecutor(max_workers=4) as jobs:
        verdicts=list(jobs.map(compare,sample))
    (HERE/"upstream_images.json").write_text(json.dumps(dict(n=len(verdicts),verdicts=verdicts),ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print([(r["id"],r.get("thumbnail_mae_on_255",r.get("error","dimensions differ"))) for r in verdicts])


if __name__=="__main__":
    main()
