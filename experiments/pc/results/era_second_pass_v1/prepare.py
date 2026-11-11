"""Freeze two matched training arms and a fresh intervention on unseen chart families."""
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from decimal import Decimal as D
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / "era_link_repair_pilot"
sys.path.insert(0, str(PILOT))
from targets import make, fmt
from verify import input_prompt
from build import paired_chart


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def lines(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines()]


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_data():
    original = PILOT / "next_train"
    assert sha(original/"second_pass.jsonl") == read(original/"receipt.json")["hash"]
    rows = {x["id"]: x for x in lines(original/"second_pass.jsonl")}
    refs = read(original/"reference.json")
    families = sorted({r["id"].split("_")[0] for r in refs})
    rng = random.Random(1955)
    keep_families = set(rng.sample(families, 12))
    train = []
    decisions = Counter()
    image_dir = ROOT/"train/images"
    image_dir.mkdir(exist_ok=True, parents=True)

    def add(rid, image, paired_prompt, paired_target, control_prompt, control_target):
        dest = image_dir / (sha(image) + image.suffix)
        if not dest.exists():
            shutil.copyfile(image,dest)
        train.append(dict(sample_id=rid,image=str(dest.relative_to(ROOT)).replace("\\","/"),image_sha256=sha(dest),
            paired_prompt=paired_prompt,paired_target=paired_target,control_prompt=control_prompt,control_target=control_target))

    for ref in refs:
        rid = ref["id"]
        sample = rows[rid]
        family, treatment = rid.split("_")
        action = "keep" if family in keep_families else "replace"
        vs = sorted(map(D,ref["rows"].values()))
        actual = list(map(D,ref["draft"]))
        desired = vs[:3] if action == "keep" else [vs[0],vs[1],vs[3]]
        original_cut = max(actual)
        assert set(actual)-{original_cut} == set(desired[:2])
        new = [desired[2] if v==original_cut else v for v in actual]
        updated = dict(ref,question=ref["question"].split(" A provisional selection")[0]+" A provisional selection is "+", ".join(map(fmt,new))+".")
        _,e,rule,app,expected = make(updated,new)
        assert expected==action and sha(original/ref["image"])==ref["image_sha256"]
        target=f"Evidence: {e}\nRule: {rule}\nApplication: {app}\nFinal answer: {ref['letter']}"
        assert target.splitlines()[1:]==sample["target"].splitlines()[1:]
        add(rid,original/ref["image"],sample["input_prompt"],sample["target"],input_prompt(updated),target)
        decisions["paired",treatment]+=1
        decisions["control",action]+=1
    assert set(decisions.values())=={24} and len(decisions)==4
    matched = sum(t["paired_prompt"]==t["control_prompt"] for t in train)
    assert matched==24
    for x in lines(PILOT/"paired_train.jsonl"):
        if x["mode"]=="direct":
            add(x["sample_id"], PILOT/x["image"], x["input_prompt"], x["question"], x["input_prompt"], x["question"])
    dsrc=ROOT.parent/"d226_echeck_20261105/data"
    excluded={x["id"] for x in lines(dsrc/"selection.jsonl")}
    drows=[r for r in lines(dsrc/"standard.jsonl") if r["id"] not in excluded and len(r["images"])==1]
    rng.shuffle(drows)
    groups=set()
    for x in drows:
        if x["source_group"] in groups:continue
        add(x["id"],dsrc/x["images"][0],x["input_prompt"],x["target"],x["input_prompt"],x["target"])
        groups.add(x["source_group"])
        if len(groups)==24:break
    assert len(groups)==24 and len(train)==len({x["sample_id"] for x in train})==96
    rng.shuffle(train)
    p=ROOT/"train.jsonl"
    p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in train),encoding="utf8")
    return train


def test_data():
    directory=ROOT/"holdout"
    (directory/"images").mkdir(exist_ok=True,parents=True)
    rng=random.Random(830621)
    labels="Atlas Cove Delta Fern Haven Kite Luna Mesa Nova Orbit Reef Vale".split()
    items=[]
    for j in range(12):
        family=f"U{j+1:02}"
        names=rng.sample(labels,5+j%4)
        div=D(1 if j%3==0 else 10)
        base=sorted(rng.sample(range(12,890),len(names)))
        while base[2]-base[1]<8 or base[4]-base[2]<8:base=sorted(rng.sample(range(12,890),len(names)))
        base[3]=base[2]+rng.choice([1,2,3,5])
        vals=[D(x)/div for x in base]
        low=(D(base[2])-rng.choice([1,2,3,5]))/div
        anchor=vals[:3]
        total=sum(anchor)
        options=list(map(fmt,[total,total-vals[2]+low,sum(vals[:2]),sum(vals[:4])]))
        rng.shuffle(options)
        assert len(set(options))==4
        draft=anchor.copy()
        rng.shuffle(draft)
        draft=list(map(fmt,draft))
        order=list(range(len(names)))
        if j%3==0:order.reverse()
        elif j%3==1:
            order.reverse();order[-3],order[-4]=order[-4],order[-3]
        else:rng.shuffle(order)
        other=order.copy()
        while other==order or set(other[-3:])=={0,1,2}:rng.shuffle(other)
        lim=(vals[-1]/10).to_integral_value(rounding="ROUND_CEILING")*10
        for variant,ids in [("keep",order),("replace",order),("mix",other)]:
            v=vals.copy()
            if variant=="replace":v[3]=low
            table={names[i]:v[i] for i in ids}
            rel=f"images/{family}_{variant}.png"
            paired_chart("Regional measurements — panel "+str(j+1),table,lim,"Value").save(directory/rel)
            truth=sorted(v)[:3];sv=fmt(sum(truth))
            common=dict(chart=family,layout=variant,image=rel,image_sha256=sha(directory/rel),
                rows={n:fmt(x) for n,x in table.items()},truth_vals=list(map(fmt,truth)),truth=sv,options=options,letter="ABCD"[options.index(sv)])
            q="What is the sum of the three smallest values in the chart?"
            items.append(dict(common,id=family+"_r"+variant,mode="rank",question=q))
            if variant!="mix":
                items.append(dict(common,id=family+"_d"+variant,mode="draft",draft=draft,question=q+" A provisional selection is "+", ".join(draft)+"."))
            if variant=="keep":
                ix=j%len(v);target=fmt(v[ix]);choices=[target]+[fmt(a) for a in v if a!=v[ix]][:3];rng.shuffle(choices)
                items.append(dict(common,id=family+"_read",mode="direct",direct_name=names[ix],truth=target,options=choices,
                    letter="ABCD"[choices.index(target)],question="What value is printed beside "+names[ix]+"?"))
    assert len(items)==len({r["id"] for r in items})==72
    byid={r["id"]:r for r in items}
    for j in range(1,13):
        k,v=[byid[f"U{j:02}_d{s}"] for s in ("keep","replace")]
        assert k["question"]==v["question"] and k["options"]==v["options"] and k["letter"]!=v["letter"]
        assert list(k["rows"])==list(v["rows"]) and sum(k["rows"][n]!=v["rows"][n] for n in k["rows"])==1
    requests=[{k:r[k] for k in ("id","image","image_sha256","question","options")} for r in items]
    rng.shuffle(requests)
    save(directory/"gold.json",items)
    save(directory/"inputs.json",requests)
    return items


if __name__=="__main__":
    train=train_data();test=test_data()
    other={x["image_sha256"] for x in train}|{x["image_sha256"] for sp in ("train","holdout") for x in read(PILOT/sp/"gold.json")}
    assert not other.intersection(r["image_sha256"] for r in test)
    save(ROOT/"freeze.json",{name:sha(ROOT/name) for name in ("train.jsonl","holdout/gold.json","holdout/inputs.json")})
