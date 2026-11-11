"""Read VIC-ERA directly from ZIP and compare it with pinned local training/eval records."""
from collections import Counter, defaultdict
import difflib
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import tempfile
import unicodedata
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/"scripts/visual_groups"))
from visual_groups import fingerprint

DEFAULT_ZIP = Path("C:/김선재/대학교/3학년/2학기/멀티모달딥러닝/프로젝트/데이터셋/vic_era_2000.zip")
CAP = {"dhash":4, "phash":6}
def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def dump(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")

def hash_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def qbody(q):
    q=q.split("Question: ",1)[-1]
    q=re.split(r"\n(?:Options:|On the last line|Please select |Show the image readings)",q,1)[0]
    q=re.sub(r"<\s*image(?:[_ ]?\d+)?\s*>"," ",q,flags=re.I)
    return " ".join(re.findall(r"\w+|[%+−-]",unicodedata.normalize("NFKC",q).casefold()))

def signature(fp):
    return dict(file=fp["file_sha256"],pixel=fp["pixel_sha256"],
                dh=int(fp["dhash"],16), ph=int(fp["phash"],16),w=fp["width"],h=fp["height"])

def load_refs():
    refs={}; manifests={}; cache={}
    def register(label,path,root):
        rows = ([json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s]
                if path.suffix==".jsonl" else read(path))
        rec=[]
        for r in rows:
            rid=r.get("sample_id",r.get("id"))
            q=r.get("question") or r.get("input_prompt") or r["prompt"]
            files=r.get("images") or [r.get("image") or r.get("image_path")]
            fps=[]
            for rel in files:
                file=(root/rel).resolve()
                if file not in cache:
                    cache[file]=signature(fingerprint(file))
                fps.append((str(rel),cache[file]))
            rec.append((rid,qbody(q),fps))
        assert len({r[0] for r in rec})==len(rec),label
        refs[label]=rec;manifests[label]={str(path.relative_to(ROOT)):hash_file(path)}
    configs={
        "donor_train200": ("code/training/era_decoder_lora/data/pilot200/exports/train_era.jsonl","code/training/era_decoder_lora/data/pilot200"),
        "D226": ("results/d226_echeck_20261105/data/standard.jsonl","results/d226_echeck_20261105/data"),
        "old_train449": ("results/era_single_call_campaign/data/train.jsonl","results/era_single_call_campaign/data"),
        "old_dev76": ("results/era_single_call_campaign/data/dev.jsonl","results/era_single_call_campaign/data"),
        "round4_train164": ("results/era_a_pair_round/data_work/train.jsonl","results/era_a_pair_round/data_work"),
        "round4_dev48": ("results/era_a_pair_round/connection_work/spec/inputs.json","results/era_a_pair_round/connection_work/spec"),
        "round5_teacher192": ("results/era_midstep_round/data/teacher_inputs.json","results/era_midstep_round/data"),
        "round5_train96": ("results/era_midstep_round/data/train_pairs.jsonl","results/era_midstep_round/data"),
        "round5_dev_charts96": ("results/era_next_direction/diagnostic/inputs.json","results/era_next_direction/diagnostic"),
        "round5_dev_ScienceQA19": ("results/era_next_direction/retention/native_validation/inputs.json","results/era_next_direction/retention/native_validation"),
    }
    for label,(path,base) in configs.items():
        register(label,ROOT/path,ROOT/base)
    imgs=ROOT/"reports/era_abcd_review_20261020/mmmu_reference_fingerprints.json"
    base=ROOT/"results/mmmu_first_baseline_20261009/baseline_seed3407/results.json"
    photo=defaultdict(list)
    for x in read(imgs):
        photo[x["id"]].append((x["field"],signature(x)))
    orig=read(base)["records"]
    refs["MMMU900"]=[(r["id"],qbody(r["question"]),photo[r["id"]]) for r in orig]
    assert len(orig)==900 and sum(map(len,photo.values()))==982
    manifests["MMMU900"]={str(p.relative_to(ROOT)):hash_file(p) for p in (base,imgs)}
    return refs,manifests

def match(a,b):
    dp=(a["ph"]^b["ph"]).bit_count()
    dd=(a["dh"]^b["dh"]).bit_count()
    if a["file"]==b["file"]:kind="file"
    elif a["pixel"]==b["pixel"]:kind="pixel"
    elif dp<=CAP["phash"] or dd<=CAP["dhash"]:kind="near"
    else:return None
    return {"kind":kind,"dhash":dd,"phash":dp}

def duplicates(mapping):
    return [sorted(ids) for ids in mapping.values() if len(ids)>1]

def groups(rids,edges):
    parent={rid:rid for rid in rids}
    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]]
            k=parent[k]
        return k
    for i,j in edges:
        i,j=find(i),find(j)
        if i!=j:parent[max(i,j)]=min(i,j)
    out=defaultdict(list)
    for k in sorted(rids):out[find(k)].append(k)
    return list(out.values())

def run(path):
    rows=[]; imgs={}; fps={}; qidx=defaultdict(list); fidx=defaultdict(list); pidx=defaultdict(list); qpidx=defaultdict(list)
    srcidx=defaultdict(list)
    with ZipFile(path) as z:
        temp_img=HERE/".fingerprint_current.tmp"
        records=json.loads(z.read("data.json"))
        for r in records:
            rid=r["id"]; im=r["image_1"]
            if im not in imgs:
                temp_img.write_bytes(z.read(im))
                imgs[im]=signature(fingerprint(temp_img))
            f=imgs[im]; q=qbody(r["question"])
            rows.append((rid,q,im))
            fps[rid]=f
            qidx[q].append(rid);pidx[f["pixel"]].append(rid);fidx[f["file"]].append(rid)
            qpidx[(q,f["pixel"])].append(rid)
            ext=r.get("mmfinereason_source") or {}
            orig=r.get("original_source") or {}
            if ext.get("id") is not None:srcidx[("mmfinereason",ext["source"],ext["split"],str(ext["id"]))].append(rid)
            if orig.get("id") is not None:srcidx[("original",orig["dataset"],orig["split"],str(orig["id"]))].append(rid)
        temp_img.unlink()
    assert len(records)==len(fps)==len(rows)==2000
    refs, manifests=load_refs()
    cross=[]
    for setname,data in refs.items():
        for rid,q,im in rows:
            a=imgs[im]
            for other,oq,pics in data:
                ih=[{**hit,"image":name} for name,b in pics if (hit:=match(a,b))]
                qm=bool(q) and q==oq
                nearq=None
                if not qm and min(len(q.split()),len(oq.split()))>=12:
                    # Only closely related, similarly sized texts; keep candidate matches for inspectability.
                    if 0.8<len(q)/len(oq)<1.25 and len(set(q.split())&set(oq.split()))>=0.8*max(len(set(q.split())),len(set(oq.split()))):
                        sim=difflib.SequenceMatcher(None,q,oq,autojunk=False).ratio()
                        if sim>=.9:nearq=round(sim,5)
                if ih or qm or rid==other or nearq:
                    cross.append({"set":setname,"id":rid,"other":other,"image":ih,"question_exact":qm,"id_exact":rid==other,"text_candidate":nearq})
    internal=[]
    for (i,_,a),(j,_,b) in itertools.combinations(rows,2):
        hit=match(imgs[a],imgs[b])
        if hit:internal.append({"id":i,"other":j,**hit})
    allids=list(fps)
    edge_near=[(r["id"],r["other"]) for r in internal]
    edge_exact=[(r["id"],r["other"]) for r in internal if r["kind"] in ("file","pixel")]
    qedges=[(g[0],j) for g in duplicates(qidx) for j in g[1:]]
    rawjoins=groups(allids,edge_near)
    joins=groups(allids,edge_near+qedges)
    subset_sizes={k:{"rows":len(v),"images_listed":sum(len(x[2]) for x in v),
                     "distinct_pixel_images":len({sig["pixel"] for _,_,pics in v for _,sig in pics})} for k,v in refs.items()}
    counts={}
    for k in refs:
        sub=[e for e in cross if e["set"]==k]
        def n(test):return len({a["id"] for a in sub if test(a)})
        counts[k]=dict(id=n(lambda a:a["id_exact"]),question=n(lambda a:a["question_exact"]),
                       text_candidates=n(lambda a:a["text_candidate"]),file=n(lambda a:any(i["kind"]=="file" for i in a["image"])),
                       exact_image=n(lambda a:any(i["kind"] in ("pixel","file") for i in a["image"])),
                       near_image=n(lambda a:any(i["kind"]=="near" for i in a["image"])),any=n(lambda a:True))
    def info(dic):
        dd=duplicates(dic)
        return {"unique":len(dic),"repeat_groups":len(dd),"rows_in_repeat_groups":sum(map(len,dd)),"max_size":max(map(len,dic.values()))}
    summary=dict(zip_size=path.stat().st_size,zip_sha256=hash_file(path),rows=len(rows),archive_images=len(imgs),
        distinct_id=len(allids),source_id_available_vic=len(set(itertools.chain.from_iterable(srcidx.values()))),
        duplicate_source_id_keys=duplicates(srcidx),question=info(qidx),file=info(fidx),pixel=info(pidx),question_and_pixel=info(qpidx),
        internal_image_pair_types=dict(Counter(e["kind"] for e in internal)),
        perceptual_image_components=len(rawjoins),perceptual_components_over1=sum(len(g)>1 for g in rawjoins),
        max_perceptual_group=max(map(len,rawjoins)),perceptual_or_exact_question_components=len(joins),
        max_combined_group=max(map(len,joins)), cross_rows=counts,reference_sizes=subset_sizes,references=manifests,hash_limits=CAP,
        vic_image_threshold="EXIF-transposed RGBA exact; 64-bit dHash ≤4 OR pHash ≤6 flags for review, per project's visual_groups.fingerprint. Near-only matches are candidates, not confirmed duplicates.")
    dump("overlap_summary.json",summary)
    dump("cross_candidates.json",cross)
    dump("internal_image_pairs.json",internal)
    dump("internal_groups.json",{"same_question":duplicates(qidx),"same_pixels":duplicates(pidx),"same_question_and_pixels":duplicates(qpidx),"perceptual_image":rawjoins,"perceptual_image_or_question":joins})
    (HERE/"vic_image_hashes.jsonl").write_text("".join(json.dumps({"id":i,"image":im,**fps[i],"normalized_question":q},ensure_ascii=False)+"\n" for i,q,im in rows),encoding="utf8")
    print(json.dumps({k:v for k,v in summary.items() if k!="references"},indent=2))

if __name__=="__main__":
    run(Path(sys.argv[1]) if len(sys.argv)>1 else DEFAULT_ZIP)
