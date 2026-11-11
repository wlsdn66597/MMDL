"""Seal 24 reviewed archive image groups and their two matched 48-row views."""
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import random
from string import ascii_uppercase
import sys
from zipfile import ZipFile

from curated import GROUPS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REG=ROOT/"results/vic_era_2000_review/overlap"
ZIP=Path("C:/김선재/대학교/3학년/2학기/멀티모달딥러닝/프로젝트/데이터셋/vic_era_2000.zip")
ARCHIVE_SHA="aa30d0794a4791261197ba732d514af6e30008df9196da4b17b533edd275edb5"
SEED=648365
FINAL_HINT='On the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def seal(path,data):
    if path.exists() and path.read_bytes()!=data:
        raise ValueError(f"Refusing to mutate frozen bytes: {path}")
    path.write_bytes(data)


def store(path,data):
    seal(path,(json.dumps(data,ensure_ascii=False,indent=2)+"\n").encode())


def letters():
    # Each type uses all 12 distinct ordered pairs of alternate forward/reverse letters.
    # Both arms retain the original correct letter for 3/12 groups of each type.
    rng=random.Random(SEED)
    while True:
        seq=[]
        for _ in range(2):
            perms=list(itertools.permutations(range(4),2))
            rng.shuffle(perms)
            for attempt in range(10000):
                start=[0,1,2,3]*3
                rng.shuffle(start)
                if sum(x==y for x,(y,z) in zip(start,perms))==sum(x==z for x,(y,z) in zip(start,perms))==3:
                    break
            else:
                raise ValueError("Could not allocate balanced letters")
            seq.extend((i,j,k) for i,(j,k) in zip(start,perms))
        if all(len({(v[0],v[arm]) for v in seq})==16 for arm in (1,2)):
            return seq


def arrangements(values,indices,rng):
    lead,fwd,inv=indices
    initial=list(values)
    rng.shuffle(initial)
    i=initial.index(values[0])
    initial[lead],initial[i]=initial[i],initial[lead]
    alternate=[""]*4
    alternate[fwd],alternate[inv]=values[:2]
    distractors=list(values[2:]);rng.shuffle(distractors)
    for j in range(4):
        if not alternate[j]:
            alternate[j]=distractors.pop()
    return initial,alternate


def prompt(q,opts):
    return "Question: "+q+"\nOptions:\n"+"\n".join(f"{ascii_uppercase[j]}. {s}" for j,s in enumerate(opts))+"\nPlease select the correct answer from the options above.\n"+FINAL_HINT


def response(ev,rule,app,letter):
    return f"Evidence: {ev}\nRule: {rule}\nApplication: {app}\nFinal answer: {letter}"


def check_calculations():
    assert 395-368==27 and (5-8)*100/8==-37.5 and (8-5)*100/5==60
    assert Fraction(100-64-26,64)==Fraction(5,32)
    assert Fraction(8+10,14+6)==Fraction(9,10)
    assert 70-Fraction(70*30,35)==10
    assert (4-1,-1-2)==(3,-3)
    assert Fraction(6+16,38+32)==Fraction(11,35)
    assert 51+10-53==8 and Fraction(10*11//2,11**2)==Fraction(5,11)
    assert Fraction(35,20)==Fraction(7,4)
    assert -40-25==-65 and 68362-67200==1162 and -128-134==-262


def main():
    check_calculations()
    assert sha(ZIP.read_bytes())==ARCHIVE_SHA
    summary=read(REG/"overlap_summary.json")
    crosses=read(REG/"cross_candidates.json")
    source_index={r["id"]:r for r in (json.loads(x) for x in (REG/"vic_image_hashes.jsonl").read_text(encoding="utf8").splitlines())}
    visual_group={rid:min(ids) for ids in read(REG/"internal_groups.json")["perceptual_image"] for rid in ids}
    ids=[r[0] for r in GROUPS]
    assert len(ids)==len(set(ids))==len({visual_group[i] for i in ids})==24
    assert not set(ids)&{r["id"] for r in crosses}
    with ZipFile(ZIP) as z:
        source={r["id"]:r for r in json.loads(z.read("data.json"))}
        (HERE/"images").mkdir(exist_ok=True)
        schedules=letters()
        reviewed=[];rows=[]
        rng=random.Random(SEED+1)
        for spec,indices in zip(GROUPS,schedules):
            sid,skill,q,qpar,qrev,ev,rule,a0,a1,*values=spec
            assert len(values)==len(set(values))==4
            assert len({q,qpar,qrev})==3 and a0!=a1
            src=source[sid]
            image=z.read(src["image_1"]);rel="images/"+Path(src["image_1"]).name
            assert sha(image)==source_index[sid]["file"]
            seal(HERE/rel,image)
            options0,options1=arrangements(values,indices,rng)
            orig_l,para_l,reverse_l=[ascii_uppercase[i] for i in indices]
            orig_p=prompt(q,options0);resp=response(ev,rule,a0,orig_l)
            meta=dict(group_id=sid,source_id=sid,source_type=src["vic_type"],image=rel,image_sha256=sha(image))
            rows.append(dict(**meta,id="r6-"+sid+"-original",role="original",control_prompt=orig_p,reverse_prompt=orig_p,control_response=resp,reverse_response=resp))
            rows.append(dict(**meta,id="r6-"+sid+"-alternate",role="alternate",control_prompt=prompt(qpar,options1),reverse_prompt=prompt(qrev,options1),control_response=response(ev,rule,a0,para_l),reverse_response=response(ev,rule,a1,reverse_l)))
            reviewed.append(dict(**meta,skill=skill,upstream=src["mmfinereason_source"],source_question=src["question"],source_answer=src["answer"],
                                 original_q=q,same_direction_q=qpar,reverse_direction_q=qrev,evidence=ev,rule=rule,forward_application=a0,inverted_application=a1,
                                 original_options=options0,alternate_options=options1,forward_value=values[0],inverse_value=values[1],
                                 original_letter=orig_l,paraphrase_letter=para_l,reverse_letter=reverse_l,
                                 image_component=visual_group[sid],visual_review="selected operands, labels and evidence checked against archived original pixels",
                                 source_use="The ZIP image is original; directional Q/options are authored for this experiment and the original ZIP record is retained separately."))
    assert Counter(r["source_type"] for r in reviewed)=={"V":12,"C":12}
    rows.sort(key=lambda r:r["id"])
    pair_file=HERE/"direction_pairs.jsonl"
    seal(pair_file,("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)).encode())
    store(HERE/"reviewed_groups.json",reviewed)
    transition={}
    for col in ("paraphrase_letter","reverse_letter"):
        trans=Counter((r["original_letter"],r[col]) for r in reviewed)
        assert len(trans)==16 and sum(n for (l,v),n in trans.items() if l==v)==6
        transition[col]={a+"->"+b:n for (a,b),n in sorted(trans.items())}
    for ty in "VC":
        for key in ("original_letter","paraphrase_letter","reverse_letter"):
            assert set(Counter(r[key] for r in reviewed if r["source_type"]==ty).values())=={3}
    receipt=dict(verified=True,n=48,groups=24,sha256=sha(pair_file.read_bytes()),group_review_sha256=sha((HERE/"reviewed_groups.json").read_bytes()),
                 source_zip_sha256=ARCHIVE_SHA,source_counts={"V":12,"C":12},variant_counts={"matched_original":24,"paraphrase_or_reverse":24},seed=SEED,
                 finals_per_letter_per_role_per_arm=6,correct_letter_stayed={"control":6,"reverse":6},letter_transitions=transition,
                 protected_corpora=list(summary["reference_sizes"]),possible_cross_candidates_selected=0,distinct_internal_components=24,
                 notes="Every question in this release is a grounded directional adaptation on an original ZIP image. Common, paraphrase and reverse questions have the same underlying states, options and ERA evidence/rule within each alternate exposure. The two alternates differ in requested relation, application and final choice. All originals and alternates belong to train only. No model outputs used for selection or labels.")
    store(HERE/"direction_receipt.json",receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=="__main__":
    main()
