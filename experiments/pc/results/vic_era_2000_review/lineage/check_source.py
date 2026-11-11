"""Match archive text locally to pinned public Parquet columns using read-only HTTPS ranges.

Only upstream shard paths and byte ranges are transmitted. Never send local text or imagery.
"""
import argparse
import bisect
import collections
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
MM_REPO = "OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking"
MM_REV = "7c93f724791d01246f5425e1182b14ec8d013847"
OM_REPO = "researcher2026/OpenMedReason"
OM_REV = "d1a5dbaad08758c7275aebd6edf4f805c4773288"


def norm(text):
    return " ".join(text.split())


def answer_text(x):
    return norm(str(x)).strip()


def sha(value):
    return hashlib.sha256(value).hexdigest()


class Subset(io.RawIOBase):
    def __init__(self, size, parts):
        self.size = size
        self.keys = sorted(parts)
        self.parts = parts
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.pos = offset if whence==0 else (self.pos if whence==1 else self.size)+offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, n=-1):
        start = self.pos
        end = self.size if n<0 else min(self.size,start+n)
        self.pos = end
        data=bytearray(max(0,end-start))
        for k in self.keys[max(0,bisect.bisect_right(self.keys,start)-1):]:
            if k>=end:
                break
            block=self.parts[k]
            a,b=max(k,start),min(k+len(block),end)
            if b>a:
                data[a-start:b-start]=block[a-k:b-k]
        return bytes(data)


def get(url, start=None, stop=None):
    rg = "-8" if start is None else f"{start}-{stop-1}"
    request=Request(url+"?download=true&bytes="+rg,headers={"Range":"bytes="+rg})
    with urlopen(request,timeout=25) as res:
        if res.status!=206:
            raise RuntimeError(f"Source did not honor range {rg}")
        a,b,size=map(int,re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",res.headers["Content-Range"]).groups())
        data=res.read()
    assert len(data)==b-a+1 and (start is None or (a,b+1)==(start,stop))
    return size,data


def table(repo,rev,shard,cols):
    url=f"https://huggingface.co/datasets/{repo}/resolve/{rev}/data/{shard}.parquet"
    total,end=get(url)
    assert end[-4:]==b"PAR1"
    footer_length=int.from_bytes(end[:4],"little")+8
    footer_pos=total-footer_length
    _,footer=get(url,footer_pos,total)
    meta=pq.read_metadata(pa.BufferReader(b"PAR1"+footer))
    spans=[]
    for gi in range(meta.num_row_groups):
        group=meta.row_group(gi)
        for ci in range(group.num_columns):
            col=group.column(ci)
            if col.path_in_schema in cols:
                p=col.dictionary_page_offset if col.has_dictionary_page else col.data_page_offset
                spans.append((p,p+col.total_compressed_size))
    spans.sort()
    merged=[]
    for a,b in spans:
        if merged and a-merged[-1][1]<100000:
            merged[-1]=(merged[-1][0],b)
        else:
            merged.append((a,b))
    parts={0:b"PAR1",footer_pos:footer}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        blocks=pool.map(lambda s:get(url,*s)[1],merged)
        for (a,b),block in zip(merged,blocks):
            parts[a]=block
    vals=pq.ParquetFile(Subset(total,parts)).read(columns=cols,use_threads=False)
    return vals.to_pylist(), dict(source=shard,upstream_rows=meta.num_rows,source_file_size=total,
                                   fetched_bytes=sum(map(len,parts.values())),columns=cols,footer_sha256=sha(footer))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("zip",type=Path)
    ap.add_argument("--medical-shards",type=int,default=1)
    args=ap.parse_args()
    with ZipFile(args.zip) as z:
        release=json.loads(z.read("data.json"))
    mm,med={},{}
    for r in release:
        if r["mmfinereason_source"]:
            mm[r["mmfinereason_source"]["row_index"]]=r
        else:
            med[r["id"]]=r
    tables=[]
    matches=[]
    # All declared MMFineReason row offsets are within first shard; insist, never wrap.
    rows,meta=table(MM_REPO,MM_REV,"sft-00000-of-00193",["id","source","question","answer","original_answer"])
    tables.append(meta)
    assert max(mm)<len(rows)
    for i,r in sorted(mm.items()):
        m=rows[i];s=r["mmfinereason_source"]
        extracted=answer_text(m["answer"])
        letter=answer_text(r["answer"])
        selected=answer_text(r["options"][ord(letter)-65]) if r["question_type"]=="multiple-choice" else letter
        matches.append(dict(zip_id=r["id"],split="sft",shard=meta["source"],row=i,
                source_identity=bool(m["id"]==s["id"] and m["source"]==s["source"]),
                question=norm(r["question"])==norm(m["question"]),explanation=norm(r["explanation"])==norm(m["original_answer"]),
                answer=letter==extracted, selected_answer_matches=selected==extracted,
                different_answer_example=({"upstream":extracted,"release":letter,"selected_text":selected}
                                          if extracted not in (letter,selected) else None)))
    med_hash=collections.defaultdict(list)
    for k,r in med.items():
        med_hash[(norm(r["explanation"]),r["answer"])].append(k)
    found=collections.Counter()
    offset=0
    for i in range(args.medical_shards):
        rows,meta=table(OM_REPO,OM_REV,f"train-{i:05}-of-00039",["question","reasoning","answer"])
        tables.append(meta)
        for j,m in enumerate(rows):
            for k in med_hash[(norm(m["reasoning"]),m["answer"])]:
                if norm(med[k]["question"])==norm(m["question"]):
                    found[k]+=1
                    matches.append(dict(zip_id=k,split="train",shard=meta["source"],global_row=offset+j,row=j,question=True,
                                        explanation=True,answer=True))
        offset+=len(rows)
    out=dict(zip_sha256=sha(args.zip.read_bytes()),revisions={MM_REPO:MM_REV,OM_REPO:OM_REV},tables=tables,
             mm_counts={f:sum(r.get(f,False) for r in matches if r["split"]=="sft") for f in ("source_identity","question","explanation","answer","selected_answer_matches")},
             med_exact_unique_in_scanned_shards=sum(n==1 for n in found.values()),
             med_ambiguous_in_scanned_shards=sum(n>1 for n in found.values()),med_missing=sorted(set(med)-found.keys()),
             matches=matches,note="Text-only comparison; MC gold can differ in text vs letter representations. Upstream image bytes and original pre-aggregation splits not independently verified.")
    (HERE/"upstream.json").write_text(json.dumps(out,ensure_ascii=False)+"\n",encoding="utf8")
    print(out["mm_counts"],"med",len(found),"missing",len(out["med_missing"]),"MiB",round(sum(x["fetched_bytes"] for x in tables)/1048576,1))


if __name__=="__main__":
    main()
