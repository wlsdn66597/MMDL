"""Build the 48 paired training items and audit their source, values and constraints."""
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from spec import GROUPS

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
REG = PROJECT / "results/vic_era_2000_review/overlap"
PRIOR = PROJECT / "results/vic_direction_round/data/reviewed_groups.json"
SOURCE = Path("C:/김선재/대학교/3학년/2학기/멀티모달딥러닝/프로젝트/데이터셋/vic_era_2000.zip")
ZIP_DIGEST = "aa30d0794a4791261197ba732d514af6e30008df9196da4b17b533edd275edb5"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def freeze(path, data):
    if path.exists():
        assert path.read_bytes() == data, f"Already sealed with different bytes: {path}"
    else:
        path.write_bytes(data)


def json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf8")


def math_key(data, question):
    # Expressions are authored in spec.py, not read from archive documents.
    allowed=dict(abs=abs, largest=lambda **v:max(v,key=v.get), smallest=lambda **v:min(v,key=v.get))
    val=eval(question["expression"],{"__builtins__":{}}, data | allowed)
    answer=question["values"][0]
    if isinstance(val,str):
        assert val.casefold()==answer.casefold()
    else:
        assert abs(val-float(answer)) < 1e-7,(val,answer)


def schedules():
    rng=random.Random(718206)
    common=[0,1,2,3]*6
    for _ in range(100000):
        left,right=common.copy(),common.copy()
        rng.shuffle(left)
        rng.shuffle(right)
        if sum(a==b for a,b in zip(left,right))!=6:
            continue
        if any(left[i]==right[i] for i,g in enumerate(GROUPS) if g["kind"] in ("A_INV","FINAL")):
            continue
        if any(left[i]!=right[i] for i,g in enumerate(GROUPS)
               if g["sid"] in ("v_73","v_115","c_598")):
            continue
        return list(zip(left,right)),rng
    raise RuntimeError("Failed to allocate balanced option positions.")


def scramble(pool, target, rng):
    distractors=pool[1:]
    rng.shuffle(distractors)
    distractors.insert(target,pool[0])
    return distractors


def main():
    assert len(GROUPS)==len({g["sid"] for g in GROUPS})==24
    assert Counter(g["kind"] for g in GROUPS)=={"E":12,"R":6,"A":3,"A_INV":2,"FINAL":1}
    assert sha(SOURCE.read_bytes())==ZIP_DIGEST
    refs=read(REG/"overlap_summary.json")["reference_sizes"]
    blocked_ids={r["id"] for r in read(REG/"cross_candidates.json")} | {r["source_id"] for r in read(PRIOR)}
    components=read(REG/"internal_groups.json")["perceptual_image"]
    comp={i:g for g in components for i in g}
    ids=[s["sid"] for s in GROUPS]
    assert len({min(comp[i]) for i in ids})==24
    assert not any(blocked_ids.intersection(comp[i]) for i in ids)
    index={r["id"]:r for r in map(json.loads,(REG/"vic_image_hashes.jsonl").read_text(encoding="utf8").splitlines())}
    (HERE/"images").mkdir(exist_ok=True)
    rows=[];review=[]
    schedules_,rng=schedules()
    with ZipFile(SOURCE) as archive:
        source={r["id"]:r for r in json.loads(archive.read("data.json"))}
        for group,positions in zip(GROUPS,schedules_):
            sid=group["sid"];kind=group["kind"];src=source[sid]
            raw=archive.read(src["image_1"])
            assert sha(raw)==index[sid]["file"]
            with Image.open(io.BytesIO(raw)) as im:
                rgb=sha(im.convert("RGB").tobytes())
                size=im.size
            dest=HERE/"images"/Path(src["image_1"]).name
            freeze(dest,raw)
            questions=group["questions"]
            assert len(questions)==2
            q0,q1=questions
            if kind=="A_INV":
                pool=q0["values"]
                assert set(pool)==set(q1["values"])
                opt=[""]*4
                opt[positions[0]]=q0["values"][0]
                opt[positions[1]]=q1["values"][0]
                remainder=[v for v in pool if v not in opt]
                rng.shuffle(remainder)
                arrangements=[[v or remainder.pop() for v in opt]]
                arrangements*=2
            elif q0["values"][0]==q1["values"][0] and set(q0["values"])==set(q1["values"]) and kind!="FINAL":
                arrangements=[scramble(q0["values"],positions[0],rng)]*2
            else:
                arrangements=[scramble(q["values"],pos,rng) for q,pos in zip(questions,positions)]
            audit=[]
            for j,(q_,opts,pos) in enumerate(zip(questions,arrangements,positions),1):
                math_key(group["facts"],q_)
                assert opts[pos]==q_["values"][0] and len(opts)==len(set(opts))==4
                base,enh=q_["ordinary"],q_["enhanced"]
                assert base["a"]==enh["a"]
                assert all(bool(base[k]) and bool(enh[k]) for k in "era")
                assert (base["e"]!=enh["e"])==(base["r"]!=enh["r"])==(kind in {"E","R"})
                rows.append(dict(id=f"r7-{sid}-q{j}",group=sid,kind=kind,image="data/images/"+dest.name,
                                 question=q_["question"],options=opts,gold="ABCD"[pos],ordinary=base,enhanced=enh))
                audit.append(dict(question=q_["question"],expression=q_["expression"],answer=q_["values"][0]))
            review.append(dict(group=sid,kind=kind,archive_type=src["vic_type"],origin=src["mmfinereason_source"],
                               original_question=src["question"],image=rows[-1]["image"],raw_sha256=sha(raw),
                               rgb_sha256=rgb,size=list(size),component=min(comp[sid]),facts=group["facts"],questions=audit))
    assert Counter(r["gold"] for r in rows)==dict.fromkeys("ABCD",12)
    assert sum(r["ordinary"]!=r["enhanced"] for r in rows)==36
    assert len({r["id"] for r in rows})==48 and len({r["rgb_sha256"] for r in review})==24
    file=HERE/"rows.jsonl";groupfile=HERE/"reviewed_groups.json"
    freeze(file,"".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows).encode("utf8"))
    freeze(groupfile,json_bytes(review))
    receipt=dict(verified=True,n_images=24,n_questions=48,changed_targets=36,
                 kind_groups=dict(Counter(g["kind"] for g in GROUPS)),type_groups=dict(Counter(g["archive_type"] for g in review)),
                 score_letter_counts=dict(Counter(r["gold"] for r in rows)),
                 paired_same_letter=sum(rows[i]["gold"]==rows[i+1]["gold"] for i in range(0,48,2)),
                 same_answer_content=sum(g["questions"][0]["values"][0]==g["questions"][1]["values"][0] for g in GROUPS),
                 protected_upstream_sets=refs,excluded_previous_r6_images=24,excluded_cross_candidates=len(read(REG/"cross_candidates.json")),
                 eligible_source_components=24,archive_sha256=ZIP_DIGEST,rows_sha256=sha(file.read_bytes()),
                 group_review_sha256=sha(groupfile.read_bytes()))
    freeze(HERE/"receipt.json",json_bytes(receipt))
    print(json.dumps(receipt))


if __name__=="__main__":
    main()
