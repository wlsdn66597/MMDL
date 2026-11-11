"""Create immutable chart shards, mine first-E model drafts, and assemble matched A/B SFT rows."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal as D
import hashlib
import io
from itertools import combinations_with_replacement
import json
import math
import os
from pathlib import Path
import random
import re
import shutil

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FONT = ROOT / "assets/aileron_regular.ttf"
SHARD_SIZE = 500
DEV_FAMILIES = 60
SEEDS = {"probe":528136, "anchors":118735, "holdout":995143, "sealed":771936}
NAMES = {
    "probe":"Alder Birch Cedar Elm Hazel Juniper Laurel Maple Oak Pine Rowan Willow".split(),
    "anchors":"Acorn Brook Canal Dune Glade Knoll Marsh Meadow Plain Ridge Slope Valley".split(),
    "holdout":"Arbor Beacon Crest Dusk Flint Grove Heath Isle Lotus Peak Stone Wave".split(),
    "sealed":"Azure Beryl Cobalt Drift Ember Fawn Gold Iris Larch Mist Plum Slate".split(),
}
QUERY = "What is the sum of the three smallest values in the chart?"
FORM = 'Please select the correct answer from the options above.\nOn the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'


def fmt(v):
    return format(D(v).normalize(),"f")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf8").splitlines() if l.strip()]


def write(path,content):
    """Atomic, safe reruns: only replace an existing result when its content changes."""
    data=content.encode("utf8")
    if path.is_file() and path.read_bytes()==data:return
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+f".{os.getpid()}.part")
    tmp.write_bytes(data);tmp.replace(path)


def save(path,obj):
    write(path,json.dumps(obj,ensure_ascii=False,indent=2)+"\n")


def save_rows(path,rows):
    write(path,"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))


def prompt(question,options):
    return "Question: "+question+"\nOptions:\n"+"".join(f"{chr(65+i)}. {v}\n" for i,v in enumerate(options))+FORM


def font(size):
    if not FONT.is_file():
        FONT.parent.mkdir(exist_ok=True)
        FONT.write_bytes(ImageFont.load_default(size=20).font_bytes)
    return ImageFont.truetype(str(FONT),size)


FONTS = {}


def draw(path,title,rows,limit,style):
    if not FONTS:FONTS.update({n:font(n) for n in (16,18,19,24)})
    im=Image.new("RGB",(1160,640),"white");d=ImageDraw.Draw(im)
    d.text((30,20),title,font=FONTS[24],fill="#142837")
    d.text((30,90),"Measured value",font=FONTS[18],fill="#364858")
    left,right,top,bottom=335,975,130,578
    for i in range(6):
        x=left+(right-left)*i/5
        d.line((x,top,x,bottom),fill="#e0e5e8",width=2)
        d.text((x,bottom+10),fmt(limit*D(i)/5),font=FONTS[16],anchor="mt",fill="#37414b")
    gap=(bottom-top)/len(rows); half=min(17,int(gap*.3))
    for i,(n,v) in enumerate(rows.items()):
        y=round(top+(i+.5)*gap);end=left+round(float(v/limit)*(right-left))
        d.text((left-12,y),n,font=FONTS[19],anchor="rm",fill="#182934")
        d.rectangle((left,y-half,max(left,end),y+half),fill=["#307285","#4d648a","#568065"][style%3])
        d.text((end+9,y),fmt(v),font=FONTS[19],anchor="lm",fill="#172636")
    path.parent.mkdir(exist_ok=True,parents=True)
    buf=io.BytesIO();im.save(buf,format="PNG",compress_level=4)
    path.write_bytes(buf.getvalue())


def family(split,index):
    rng=random.Random(SEEDS[split]+index*4523)
    count=rng.randint(5,9); names=rng.sample(NAMES[split],count)
    scale=D(rng.choice([1,10,10]))
    numbers=sorted(rng.sample(range(5,1170),count))
    while numbers[2]-numbers[1]<22 or numbers[4]-numbers[2]<30:
        numbers=sorted(rng.sample(range(5,1170),count))
    distance=rng.randint(12,20) if index%8==0 else rng.randint(1,6)
    numbers[3]=numbers[2]+distance
    base=[D(i)/scale for i in numbers]
    down=(D(numbers[2])-rng.randint(1,8))/scale
    order=list(range(count))
    if index%6<4:
        order.reverse()
        if index%6!=0:order[-3],order[-4]=order[-4],order[-3]
    else:
        rng.shuffle(order)
    return rng,names,base,down,order


def chart(split,index,fid,suffix,out,case,order,*,mode="rank",draft=None,options=None,direct=None):
    rng,names,values,down,_=family(split,index)
    if case=="replace":values[3]=down
    rows={names[i]:values[i] for i in order}
    best=sorted(rows.values())[:3];truth=sum(best)
    question=QUERY
    if mode=="draft":
        assert draft is not None
        question+=" A provisional selection is "+", ".join(draft)+"."
    if mode=="direct":
        direct=direct or rng.choice(names);truth=rows[direct]
        question="What value is printed beside "+direct+"?"
        nums=[v for n,v in rows.items() if n!=direct]
        options=[fmt(truth)]+[fmt(x) for x in rng.sample(nums,3)]
        rng.shuffle(options)
    if options is None:
        ordered=sorted(rows.values())
        options=list(map(fmt,[truth,truth-ordered[2]+ordered[3],sum(ordered[:2]),sum(ordered[:4])]))
        rng.shuffle(options)
    assert len(options)==len(set(options))==4 and fmt(truth) in options
    rel=f"images/{fid}_{suffix}.png";path=out/rel
    if not path.is_file():
        limit=(max(rows.values())/10).to_integral_value(rounding="ROUND_CEILING")*10
        title=("Survey" if split!="holdout" else "Regional study")+f" — panel {index+1}"
        draw(path,title,rows,limit,index)
    return dict(id=fid+"_"+suffix,chart=fid,layout=case,image=rel,image_sha256=digest(path),
                rows={n:fmt(v) for n,v in rows.items()},truth_vals=list(map(fmt,best)),truth=fmt(truth),letter="ABCD"[options.index(fmt(truth))],
                mode=mode,question=question,options=options,**({"draft":draft} if draft else {}),**({"direct_name":direct} if direct else {}))


def variants(split,i,out):
    rng,names,vals,down,order=family(split,i)
    fid=("F" if split=="anchors" else "X")+f"{i+1:03}"
    draft=list(map(fmt,vals[:3]));rng.shuffle(draft)
    options=list(map(fmt,[sum(vals[:3]),sum(vals[:2])+down,sum(vals[:2]),sum(vals[:4])]))
    rng.shuffle(options)
    assert len(set(options))==4
    others=order.copy()
    while others==order or set(others[-3:])=={0,1,2}:rng.shuffle(others)
    output=[]
    for case,these in [("keep",order),("replace",order),("mix",others)]:
        r=chart(split,i,fid,case,out,case,these,options=options)
        r["id"]=fid+"_r"+case
        output.append(r)
        if case!="mix":
            d=dict(r,id=fid+"_d"+case,mode="draft",draft=draft,
                   question=QUERY+" A provisional selection is "+", ".join(draft)+".")
            output.append(d)
        if case=="keep":
            direct=chart(split,i,fid,case,out,case,these,mode="direct",direct=names[i%len(names)])
            direct["id"]=fid+"_read";output.append(direct)
    k,v=[next(r for r in output if r["id"].endswith("_d"+x)) for x in ("keep","replace")]
    assert k["question"]==v["question"] and k["options"]==v["options"] and k["letter"]!=v["letter"]
    assert list(k["rows"])==list(v["rows"]) and sum(k["rows"][n]!=v["rows"][n] for n in k["rows"])==1
    return output


def freeze_split(out,rows,seed):
    inputs=[{k:r[k] for k in ("id","image","image_sha256","question","options")} for r in rows]
    random.Random(seed).shuffle(inputs)
    assert len(rows)==len({r["id"] for r in rows})
    save(out/"inputs.json",inputs);save(out/"gold.json",rows)
    save(out/"receipt.json",dict(n=len(rows),gold_sha256=digest(out/"gold.json"),inputs_sha256=digest(out/"inputs.json"),
        font_sha256=digest(FONT),generator_seed=seed))


def probe(shard):
    assert shard>=0
    out=ROOT/"probe"/f"{shard:03}"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index in range(shard*SHARD_SIZE,(shard+1)*SHARD_SIZE):
        fid=f"P{index+1:06}";_,_,_,_,order=family("probe",index)
        r=chart("probe",index,fid,"rank",out,"keep",order)
        rows.append(r)
    freeze_split(out,rows,SEEDS["probe"]+shard)
    return read(out/"receipt.json")


def teacher(it,draft,mode):
    rows={n:D(v) for n,v in it["rows"].items()}
    names={v:n for n,v in rows.items()};allvals=set(names)
    if len(names)!=len(rows):raise ValueError("This pilot admits unique displayed row values only.")
    if it["mode"]=="direct":
        name=it["direct_name"];expected=rows[name]
        assert fmt(expected)==it["truth"]
        e=f"Direct. The {name} bar is labeled {fmt(expected)}."
        r="Read the value printed beside the named bar."
        a=f"The requested value is {it['truth']}, option {it['letter']}."
        masked=""
    else:
        cand=list(map(D,draft))
        assert len(cand)==3 and set(cand)<=allvals
        truth=sorted(allvals)[:3]
        assert truth==list(map(D,it["truth_vals"])) and sum(truth)==D(it["truth"])
        sentence="My initial candidates are "+", ".join(map(fmt,cand))+"."
        masked="Evidence: "+sentence if mode=="natural" else ""
        active=set(cand);duplicate=len(active)!=3
        # A visibly well-separated cutoff may stop directly. A boundary violation and/or duplicate is observable with the picture.
        ceiling=max(active);excluded=allvals-active
        proximity=(min(excluded)-ceiling)/(max(allvals)-min(allvals))
        direct=mode!="fixed" and not duplicate and proximity>D(".015")
        gate="Direct." if direct else "Check."
        words=[sentence,gate] if mode=="natural" else [gate,sentence]
        if direct:
            assert sorted(active)==truth
            words.append("The selected bars are distinct and the remaining bars are visibly longer.")
        else:
            if duplicate:
                repeated=next(v for v in cand if cand.count(v)>1)
                while len(active)<3:
                    entrant=min(allvals-active)
                    words.append(f"{names[repeated]} at {fmt(repeated)} was counted more than once. Count that row once and fill the vacant place with {names[entrant]} at {fmt(entrant)}, the lowest unselected row.")
                    active.add(entrant)
            for _ in range(4):
                ceiling=max(active);challenger=min(allvals-active)
                words.append(f"Is any excluded bar smaller than {fmt(ceiling)}?")
                direction="below" if challenger<ceiling else "above"
                words.append(f"The lowest excluded bar is {names[challenger]} at {fmt(challenger)}, {direction} {names[ceiling]} at {fmt(ceiling)}.")
                if challenger>ceiling:
                    words.append("Keep the selection.");break
                words.append(f"Replace {fmt(ceiling)} with {fmt(challenger)}.")
                active.remove(ceiling);active.add(challenger)
        assert sorted(active)==truth
        words.append("Verified values: "+", ".join(map(fmt,truth))+".")
        e=" ".join(words)
        r="Add the three smallest chart values after verifying the selection."
        a=" + ".join(map(fmt,truth))+f" = {it['truth']}, option {it['letter']}."
    out=f"Evidence: {e}\nRule: {r}\nApplication: {a}\nFinal answer: {it['letter']}"
    assert out.startswith(masked)
    return out,masked


def shared(rid,image,sha,question,target,kind,family):
    return dict(sample_id=rid,image=image,image_sha256=sha,kind=kind,family=family,
                control_prompt=question,control_target=target,control_mask_prefix="",
                self_prompt=question,self_target=target,self_mask_prefix="")


def fixed():
    hold=ROOT/"holdout"; out=ROOT/"anchors"
    holdrows=[r for i in range(DEV_FAMILIES) for r in variants("holdout",i,hold)]
    freeze_split(hold,holdrows,SEEDS["holdout"])
    rows=[];refs=[]
    for i in range(100):
        for it in variants("anchors",i,out):
            if it["mode"]!="draft":continue
            tgt,_=teacher(it,it["draft"],"fixed")
            rows.append(shared(it["id"],"anchors/"+it["image"],it["image_sha256"],
                               prompt(it["question"],it["options"]),tgt,"fixed_"+it["layout"],it["chart"]))
            refs.append(it)
    for i in range(100,300):
        fid=f"D{i:04}";_,_,_,_,order=family("anchors",i)
        it=chart("anchors",i,fid,"read",out,"keep",order,mode="direct")
        tgt,_=teacher(it,None,"natural")
        rows.append(shared(it["id"],"anchors/"+it["image"],it["image_sha256"],
                           prompt(it["question"],it["options"]),tgt,"direct",it["chart"]))
        refs.append(it)
    source=ROOT.parent/"d226_echeck_20261105/data"
    base=[r for r in jsonl(source/"standard.jsonl") if len(r["images"])==1]
    random.Random(44911).shuffle(base)
    levels=defaultdict(list)
    for row in base:levels[row["source_group"]].append(row)
    queue=[]
    while any(levels.values()):
        for group in sorted(levels):
            if levels[group]:queue.append(levels[group].pop())
    for it in queue[:100]:
        img=source/it["images"][0];h=digest(img);rel="anchors/images/d_"+h+img.suffix
        dst=ROOT/rel
        if not dst.exists():shutil.copyfile(img,dst)
        assert digest(dst)==h
        rows.append(shared(it["id"],rel,h,it["input_prompt"],it["target"],"D",it["source_group"]))
    assert Counter(r["kind"] for r in rows)==dict(fixed_keep=100,fixed_replace=100,direct=200,D=100)
    assert len({r["sample_id"] for r in rows})==500
    save_rows(ROOT/"anchors.jsonl",rows);save(out/"reference.json",refs)
    save(out/"receipt.json",dict(counts=dict(Counter(r["kind"] for r in rows)),D_groups=len({r["family"] for r in rows if r["kind"]=="D"}),
        manifest_sha256=digest(ROOT/"anchors.jsonl"),reference_sha256=digest(out/"reference.json"),font_sha256=digest(FONT)))
    save(ROOT/"sealed_spec.json",dict(seed=SEEDS["sealed"],split="sealed",families=DEV_FAMILIES,
         benchmark="same cases and questions as holdout",labels=NAMES["sealed"],status="specified, not rendered or scored",
         data_rule="family(split, index), same keep/replace/mix/direct cases, frozen seeds and family split; no source rows in this file"))
    return dict(holdout=read(hold/"receipt.json"),anchors=read(out/"receipt.json"))


NUM = r"(?<![\w.])-?\d+(?:\.\d+)?"
EVIDENCE = re.compile(r"(?is)^\s*(?:\*\*)?Evidence(?:\*\*)?\s*:\s*(.*?)(?=(?:\s+)(?:Rule|Application|Final answer)\s*:|\Z)")


def extract_initial(raw,rows):
    """Admit a concrete first triple in Evidence only; never use gold ranks, later corrections or options."""
    match=EVIDENCE.match(raw)
    if not match:return None
    e=match[1].strip()
    before=re.split(r"\b(?:recalculat\w*|replac\w*|corrected|revised|instead|verified (?:values|selection|set)|sorted values|verify the selection)\b",e,maxsplit=1,flags=re.I)[0]
    before=re.sub(r"^\s*(?:Check|Direct(?: read)?|Verify)\s*[:.]\s*","",before,flags=re.I)
    allowed={D(v) for v in rows.values()}
    def accept(vs):
        vs=list(map(D,vs))
        return list(map(fmt,vs)) if len(vs)==3 and set(vs)<=allowed else None
    # Explicit equations and candidate lists come before relational comparison operands.
    first=re.split(r"(?<=\D)\.\s+|(?<=\d)\.(?!\d)\s+|\?|\n|;",before,maxsplit=1)[0]
    eq=re.search(rf"({NUM})\s*\+\s*({NUM})\s*\+\s*({NUM})",first)
    if eq:return accept(eq.groups())
    if re.search(r"(?:candidates|selection|three (?:smallest|lowest|shortest)|(?:smallest|shortest|lowest) (?:three|bars)|^\s*Is\s+[-\d])",first,re.I):
        vals=re.findall(NUM,first)
        if len(vals)==3 and not re.search(r"\b(?:below|above|between|lower than|higher than|instead)\b",first,re.I):
            # Conflicting claims of a named bar's numeric value need adjudication.
            for name,val in rows.items():
                hit=re.search(rf"\b{re.escape(name)}\s*(?:is|at|=|:)\s*({NUM})",first,re.I)
                if hit and D(hit[1])!=D(val):return None
            return accept(vals)
    # One candidate per rank, e.g. "Is 12 first? Yes ... Is 15 second? Yes ..."
    selected=[]
    for slot in range(3):
        if slot==0:rank=r"(?:the\s+)?(?:smallest|lowest|shortest|first)"
        elif slot==1:rank=r"(?:the\s+)?(?:second(?:[- ](?:smallest|lowest|shortest))?|next(?:[- ](?:smallest|lowest|shortest))?)"
        else:rank=r"(?:the\s+)?(?:third(?:[- ](?:smallest|lowest|shortest))?)"
        question=re.search(rf"\b(?:is|next\s+is)\s+({NUM})\s+{rank}\b[^?\n]*\?\s*(?:yes|correct|confirmed)\b",before,re.I)
        statement=re.search(rf"\b{rank}\s+(?:(?:bar|value)\s+)?(?:is|:)\s+([A-Za-z]+)\s+(?:at|=)\s+({NUM})",before,re.I)
        choices=[]
        if question:choices.append((question.start(),question[1],question.end()))
        if statement and statement[1] in rows and D(statement[2])==D(rows[statement[1]]):
            choices.append((statement.start(),statement[2],statement.end()))
        if not choices:
            # A combined question can affirm both remaining slots.
            if slot==1:
                two=re.search(rf"\bare\s+({NUM})\s+and\s+({NUM})\s+(?:the\s+)?next\s+two\s+(?:lowest|smallest|shortest)\b[^?]*\?\s*yes\b",before,re.I)
                if two:return accept([*selected,two[1],two[2]])
            return None
        _,value,end=min(choices)
        selected.append(value);before=before[end:]
    return accept(selected)


def tally():
    collected=[]; audit=[]; identity=None
    shard_info=[]
    for folder in sorted((ROOT/"probe").glob("[0-9][0-9][0-9]")):
        if not (folder/"done.json").is_file():continue
        receipt=read(folder/"receipt.json");manifest=read(folder/"manifest.json")
        if manifest["mode"]!="probe" or not manifest["model"].endswith("/era_link_repair_pilot/models/question/merged"):
            raise ValueError("Probe must originate in the frozen starting checkpoint")
        core={k:manifest[k] for k in ("weights","model_checkpoint","config_sha256","seed_base","script_sha256","batch")}
        if identity is not None and identity!=core:raise ValueError("Mixed rollout identities")
        identity=core
        assert receipt["inputs_sha256"]==manifest["inputs_sha256"]==digest(folder/"inputs.json")
        assert receipt["gold_sha256"]==digest(folder/"gold.json")
        done=read(folder/"done.json")
        assert done["output_sha256"]==digest(folder/"predictions.jsonl") and done["n"]==receipt["n"]==SHARD_SIZE
        gold={r["id"]:r for r in read(folder/"gold.json")};pred=jsonl(folder/"predictions.jsonl")
        assert [x["id"] for x in pred]==[x["id"] for x in read(folder/"inputs.json")]
        counts=Counter()
        for r in pred:
            it=gold[r["id"]];assert r["image_sha256"]==it["image_sha256"]
            original=EVIDENCE.match(r["raw"]);orig=original[1].strip() if original else None
            vals=extract_initial(r["raw"],it["rows"]) if r["finish_reason"]=="stop" else None
            status="unclear" if vals is None else "keep" if sorted(map(D,vals))==sorted(map(D,it["truth_vals"])) else "replace"
            counts[status]+=1
            audit.append(dict(id=r["id"],shard=folder.name,seed=r["seed"],status=status,initial=vals,original_E=orig,
                              finish=r["finish_reason"],raw_sha256=hashlib.sha256(r["raw"].encode()).hexdigest(),image_sha256=it["image_sha256"]))
            if vals is not None: collected.append(dict(item=it,initial=vals,shard=folder.name,status=status))
        shard_info.append(dict(shard=folder.name,counts=dict(counts),gold_sha256=receipt["gold_sha256"],
            inputs_sha256=receipt["inputs_sha256"],manifest_sha256=digest(folder/"manifest.json"),output_sha256=done["output_sha256"]))
    ids=[r["id"] for r in audit]
    assert len(ids)==len(set(ids))
    counts=Counter(r["status"] for r in audit)
    summary=dict(ready=counts["keep"]>=250 and counts["replace"]>=250,counts={k:counts[k] for k in ("keep","replace","unclear")},
        complete_shards=len(shard_info),n=len(audit),shards=shard_info,rollout_identity=identity)
    save(ROOT/"tally.json",summary);save_rows(ROOT/"draft_audit.jsonl",audit)
    return summary,collected


def assemble():
    if (ROOT/"freeze.json").is_file():
        freeze=read(ROOT/"freeze.json")
        assert all(digest(ROOT/k)==v for k,v in freeze.items())
        return read(ROOT/"dataset_receipt.json")
    summary,mined=tally()
    if not summary["ready"]:
        raise RuntimeError(f"Insufficient genuine rollouts, mine another shard: {summary['counts']}")
    anchor_ref=read(ROOT/"anchors/receipt.json");assert anchor_ref["manifest_sha256"]==digest(ROOT/"anchors.jsonl")
    shared_rows=jsonl(ROOT/"anchors.jsonl");assert len(shared_rows)==500
    rng=random.Random(42137);groups={}
    for status in ("keep","replace"):
        pool=[x for x in mined if x["status"]==status]
        rng.shuffle(pool);groups[status]=pool[:250]
    selected=groups["keep"]+groups["replace"]
    # Match the empirical distribution of erroneous rank patterns on the SAME chart length.
    # Assignment uses another item's pattern when available. This yields an engineered A draft from gold rows.
    buckets=defaultdict(list)
    for row in selected:
        vals=sorted(map(D,row["item"]["rows"].values()))
        pattern=tuple(vals.index(D(v)) for v in row["initial"]);row["pattern"]=pattern
        buckets[(row["status"],len(vals),len(set(pattern)))].append(row)
    for items in buckets.values():
        rng.shuffle(items)
        items.sort(key=lambda r:tuple(sorted(r["pattern"])))
        rotation=max(Counter(tuple(sorted(r["pattern"])) for r in items).values())
        for i,row in enumerate(items):
            pattern=items[(i+rotation)%len(items)]["pattern"]
            row["control_source"]="matched_rank_pattern"
            orig=row["pattern"]
            if row["status"]=="replace" and Counter(pattern)==Counter(orig):
                # A is a constructed draft. When all observed mistakes have the same rank pattern,
                # choose the nearest different valid error, preserving the number of missing true
                # rows, of distinct selected rows, and interventions needed. B is never edited.
                values=sorted(map(D,row["item"]["rows"].values()))
                def effect(p):
                    return sum(values[i] for i in p)-sum(values[:3])
                alternatives=[p for p in combinations_with_replacement(range(len(values)),3)
                    if Counter(p)!=Counter(orig) and len(set(p))==len(set(orig))
                    and len(set(p)&{0,1,2})==len(set(orig)&{0,1,2}) and set(p)!={0,1,2}]
                if alternatives:
                    def distance(p):
                        return (abs(effect(p)-effect(orig)),sum(abs(a-b) for a,b in zip(p,sorted(orig))),p)
                    best=min(alternatives,key=distance)
                    ranks=sorted(range(3),key=lambda ix:(orig[ix],ix));result=[0,0,0]
                    for ix,rk in zip(ranks,best):result[ix]=rk
                    pattern=tuple(result)
                    row["control_source"]="nearest_distinct_matched_error"
                else:
                    row["control_source"]="no_distinct_matched_error_available"
            row["control_pattern"]=pattern
    results=list(shared_rows);modes=Counter();swaps=Counter();provenance=[]
    for status,items in groups.items():
        for i,row in enumerate(items):
            it=row["item"];values=sorted(map(D,it["rows"].values()))
            mode="natural" if i<125 else "supplied";kind=f"mined_{status}_{mode}"
            path=Path("probe")/row["shard"]/it["image"]
            dest=dict(sample_id=it["id"],kind=kind,image=path.as_posix(),image_sha256=it["image_sha256"],family=it["chart"])
            source={}
            for arm,sets in (("self",row["initial"]),("control",[fmt(values[j]) for j in row["control_pattern"]])):
                assert ("keep" if sorted(map(D,sets))==sorted(map(D,it["truth_vals"])) else "replace")==status
                question=it["question"]
                if mode=="supplied":question+=" A provisional selection is "+", ".join(sets)+"."
                target,masked=teacher(it,sets,mode)
                dest.update({arm+"_prompt":prompt(question,it["options"]),arm+"_target":target,arm+"_mask_prefix":masked})
                source[arm+"_draft"]=sets
            assert dest["control_target"].splitlines()[1:]==dest["self_target"].splitlines()[1:]
            assert (dest["self_mask_prefix"]!="")== (mode=="natural")
            modes[kind]+=1;swaps[status]+=Counter(source["control_draft"])!=Counter(source["self_draft"])
            results.append(dest)
            provenance.append(dict(sample_id=it["id"],shard=row["shard"],kind=kind,control_source=row["control_source"],
                boundary_effect_control=fmt(sum(map(D,source["control_draft"]))-D(it["truth"])),
                boundary_effect_self=fmt(sum(map(D,source["self_draft"]))-D(it["truth"])),**source))
    assert set(modes.values())=={125} and len(modes)==4
    assert len(results)==len({r["sample_id"] for r in results})==1000
    assert len(selected)==len({r["item"]["chart"] for r in selected})==500
    hold=read(ROOT/"holdout/gold.json");known={i["image_sha256"] for i in hold}
    assert all(digest(ROOT/r["image"])==r["image_sha256"] and r["image_sha256"] not in known for r in results)
    def table_signature(r):return tuple(sorted(map(D,r["rows"].values())))
    train_tables={table_signature(x["item"]) for x in selected}|{table_signature(r) for r in read(ROOT/"anchors/reference.json")}
    assert train_tables.isdisjoint(table_signature(r) for r in hold)
    rng.shuffle(results);save_rows(ROOT/"train.jsonl",results);save_rows(ROOT/"mined_sources.jsonl",provenance)
    comparisons={}
    for group,group_rows in [("all",results)]+[(k,[r for r in results if r["kind"]==k]) for k in sorted({r["kind"] for r in results})]:
        comparisons[group]=dict(n=len(group_rows))
        for name in ("prompt","target"):
            changed=sum(r["control_"+name]!=r["self_"+name] for r in group_rows)
            comparisons[group][name+"_different"]=changed
            comparisons[group][name+"_identical"]=len(group_rows)-changed
        comparisons[group]["either_different"]=sum(any(r["control_"+name]!=r["self_"+name] for name in ("prompt","target")) for r in group_rows)
    report=dict(included=len(results),counts=dict(Counter(r["kind"] for r in results)),n_mined=500,
        natural_rates_before_selection=summary["counts"],shards=summary["shards"],rollout_identity=summary["rollout_identity"],
        differing_control_sets=dict(swaps),text_comparison=comparisons,engineered_source_counts=dict(Counter(r["control_source"] for r in provenance)),
        D_groups=anchor_ref["D_groups"],paired_train_sha256=digest(ROOT/"train.jsonl"),
        extractor_sha256=digest(Path(__file__)),notes="Synthetic family split; explicit first E only, no final-answer oracle; unique row values, duplicate claims allowed.")
    save(ROOT/"dataset_receipt.json",report)
    files=("train.jsonl","holdout/gold.json","holdout/inputs.json","dataset_receipt.json","anchors.jsonl","mined_sources.jsonl","draft_audit.jsonl","sealed_spec.json")
    save(ROOT/"freeze.json",{f:digest(ROOT/f) for f in files})
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("probe","fixed","inspect","assemble"))
    p.add_argument("--shard",type=int,default=0)
    a=p.parse_args()
    if a.action=="probe": result=probe(a.shard)
    elif a.action=="fixed": result=fixed()
    elif a.action=="inspect": result=tally()[0]
    else:result=assemble()
    print(json.dumps(result,ensure_ascii=False))


if __name__=="__main__":main()
