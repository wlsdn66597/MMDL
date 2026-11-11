"""Construct the matched D226 SFT targets. Run from any cwd."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from PIL import Image

OUT = Path(__file__).resolve().parent
SRC = OUT.parents[1] / "d_era_extension_20261105"
REPO = SRC.parents[1]
SOURCE_SHA = "e5d45de57da2461345c859359ef8201952e62e42eb1224a769fac9daf2f9ef94"
def sha(b):
    return hashlib.sha256(b).hexdigest()
def rows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines()]
def write(p, rr):
    p.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rr),encoding="utf-8")
def target(r):
    return f"Evidence: {r['evidence']}\nRule: {r['rule']}\nApplication: {r['application']}\nFinal answer: {r['answer']}"

# Each original numerical claim, result and mapping is retained when direct speech is removed.
AP = {
    0: "The two output guides give 8,000 and 9,600 billion constant dollars at price levels 85 and 105, respectively. Using those values, the rise is 9,600 − 8,000 = 1,600 billion constant dollars, option B.",
    12: "The amount claimed was $250,000. No transaction is established by the headline, so this was a $250,000 claim only.",
    17: "The poster says “Enlist To-Day.” Recalling the sinking serves to recruit viewers through anger over the deaths, option A.",
    46: "The $1,000 in checkable deposits leaves M1 and goes into a CD inside M2 but outside M1. M1 therefore falls $1,000 while the transfer within M2 leaves M2 unchanged, option B.",
    65: "Total ink cost rises by a factor of $900 ÷ $300 = 3. The unit costs $300/2,000 and $900/6,000 both give $0.15. Thus total cost triples while unit cost stays the same, option A.",
    77: "The laptop or handwritten note-taking method is varied. Learning performance is measured. These are the roles in option B.",
    100: "The bank has 42,400,000 baht ÷ 50 baht/$ = $848,000. The shortfall is $1,050,000 − $848,000 = $202,000, option A.",
    116: "Equilibrium output of 8,800 is about 700 below potential output of 9,500. That shortfall indicates a recessionary output gap, option D.",
    134: "Available cost is $75,000 + $63,000 = $138,000. Using the hypothetical $50,000 ending balance leaves $138,000 − $50,000 = $88,000 completed, option B.",
    136: "At 60 units total cost is about $400 and fixed cost is $160. Their difference is about $240, option B.",
    189: "Accumulated cost is $75,000 + $63,000 = $138,000. Subtracting the $43,000 ending balance gives $95,000 completed.",
    207: "One extra unit contributes $100 − $20 = $80. Covering the $4,000 loss takes $4,000 ÷ $80 = 50 additional units, option C.",
    211: "Increased cervical pressure triggers oxytocin release and stronger uterine contractions. Those contractions push the baby harder against the cervix, increasing the original pressure, so this is positive feedback.",
    222: "At 50% saturation the pressures are about 19 for fetal and 26.8 for adult hemoglobin. Because 19 is lower, the fetal curve reaches 50% at lower pressure.",
}
SPECIAL = {
    12: "The headline says she claims $250,000 for loss of kingdom; it does not report payment.",
    27: "The eagle is centered on the United States, with its wings stretching toward the Philippine Islands and Puerto Rico. The caption emphasizes a span of ten thousand miles.",
    53: "Blue water fills almost the entire background behind the closely framed boat occupants; no distant horizon is visible.",
    70: "The colored line follows the expedition through named places and Indigenous regions; it traces a path, not an enclosed border.",
    79: "At 625 boats the table shows $2,496 per boat and $1,560,000 total fixed cost. The question holds that total fixed.",
}
# Authors' English numeric strings like G5, M1, and $1.05 million are distinguished.
def normal_application(i,r):
    t = r["application"]
    if i in AP:
        new=AP[i]
    elif r["task"]=="named_interval":
        new=t.replace("What is the interval number? Counting", "Counting").replace("What is its size and direction? The pitches", "The pitches")
    elif r["task"]=="chord_member":
        m=re.fullmatch(r"How far is (.+?) above root (.+?) modulo an octave\? The offset is (.+?) semitones\. (.+)",t)
        assert m
        note, root, size, conclusion=m.groups()
        new=f"{note} is {size} semitones above root {root} modulo an octave. {conclusion}"
    else:
        new=t
    assert "?" not in new and (t!=new) == ("?" in t)
    assert not set(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?",t)) - set(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?",new)), i
    return new

def main():
    sys.path.insert(0,str(REPO/".runtime/tokenizers"))
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(REPO/".runtime/qwen_tokenizer/tokenizer.json"))
    def n(s):
        return len(tokenizer.encode(s,add_special_tokens=False).ids)
    original=SRC/"data/d_era.jsonl"
    assert sha(original.read_bytes())==SOURCE_SHA
    base=rows(original)
    audits={x["id"]:x for x in rows(SRC/"metadata/row_audit.jsonl")}
    candidates={}
    for line in (OUT/"echeck_drafts.tsv").read_text(encoding="utf-8").splitlines():
        i, reason, declar, question=line.split("|")
        assert int(i) not in candidates and "?" not in declar and question.count("?")==1 and question.endswith("?")
        candidates[int(i)]=(reason,declar,question)
    pair=[[],[]]; selection=[]; observed=set(); diffs=[]; all_lengths=[[],[]]; question_changes=0
    for r in base:
        a=audits[r["id"]];i=a["source_index_zero_based"]
        assert sha(r["input_prompt"].encode())==a["input_prompt_sha256"] and sha(r["answer"].encode())==a["reference_answer_sha256"]
        assert sha(r["target"].encode())==a["target_sha256"]
        app=normal_application(i,r)
        question_changes+=(app!=r["application"])
        stems=["",""]
        observation=r["evidence"]
        if i in candidates:
            reason, *stems=candidates[i]
            observation=SPECIAL.get(i,observation)
            assert not set(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?",r["evidence"])) - set(re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?",observation)), i
            selection.append(dict(id=r["id"],source_index_zero_based=i,reason=reason,control_focus=stems[0],
                                  checked_focus=stems[1],common_observation=observation,source_image=r["images"][0],
                                  image_sha256=a["original_image_sha256"][0],inherited_review=a["inherited_source_status"]))
        for arm,stem in enumerate(stems):
            evidence=(stem+" "+observation).strip()
            out={**r,"evidence":evidence,"application":app,"reasoning_style":"evidence_check" if arm and stem else "declarative",
                 "upstream_reasoning_style":r["reasoning_style"]}
            out["target"]=target(out);out["target_sha256"]=sha(out["target"].encode())
            assert out["target"].count("?")==int(bool(arm and stem))
            assert out["target"].splitlines()[-1]=="Final answer: "+r["answer"]
            pair[arm].append(out)
            all_lengths[arm].append(a["total_tokens_same_prompt_plus_new_target"]-n(r["target"])+n(out["target"]))
        left,right=(x[-1] for x in pair)
        for key in set(left)-{"target","target_sha256","evidence","reasoning_style"}: assert left[key]==right[key], (i,key)
        assert target(left).split("\n",1)[1]==target(right).split("\n",1)[1]
        if stems[0]:
            assert left["evidence"].endswith(observation) and right["evidence"].endswith(observation)
            diffs.append(n(right["target"])-n(left["target"]))
        else:
            assert left==right
        for j,img in enumerate(r["images"]):
            if img in observed: continue
            observed.add(img); original_image=SRC/img; dest=OUT/img
            assert sha(original_image.read_bytes())==a["original_image_sha256"][j]
            with Image.open(original_image) as im: im.load()
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original_image,dest)
            assert sha(dest.read_bytes())==a["original_image_sha256"][j]
    assert len({r["id"] for r in base})==len(base)==226 and len(observed)==130 and question_changes==40
    assert len(selection)==len(candidates)
    assert all(max(lengths)<=4096 for lengths in all_lengths)
    for name, rr in zip(("standard","checked"),pair):
        write(OUT/f"{name}.jsonl",rr)
        assert rows(OUT/f"{name}.jsonl")==rr
    write(OUT/"selection.jsonl",selection)
    manifest={f:sha((OUT/f).read_bytes()) for f in ["standard.jsonl","checked.jsonl","selection.jsonl","build_pairs.py","echeck_drafts.tsv"]}
    metrics=dict(source_sha256=SOURCE_SHA,paired_rows=len(base),selected=len(selection),identical_other_rows=len(base)-len(selection),
                 old_application_questions_removed=question_changes,unchanged_original_images=len(observed),source_groups=len({r["source_group"] for r in base}),
                 selected_by_source=dict(Counter(r["source_slug"] for r in pair[1] if r["reasoning_style"]=="evidence_check")),
                 token_estimate=dict(standard_max=max(all_lengths[0]),checked_max=max(all_lengths[1]),total_difference=sum(diffs),
                                     min_paired_delta=min(diffs),max_paired_delta=max(diffs)),
                 selected_observation_strings_identical=True,rule_application_final_identical_all=True,
                 file_hashes=manifest)
    (OUT/"validation.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(metrics,ensure_ascii=False))
if __name__=="__main__":main()
