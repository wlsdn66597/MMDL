"""Independently confirm the generated prompt families and supplied-image lookup."""
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CHECK=ROOT/"results/vic_era_2000_review/overlap"
sys.path.insert(0,str(CHECK))
from check_overlap import load_refs, qbody


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    rows=[json.loads(s) for s in (HERE/"direction_pairs.jsonl").read_text(encoding="utf8").splitlines()]
    items=read(HERE/"reviewed_groups.json")
    receipt=read(HERE/"direction_receipt.json")
    legacy,_=load_refs()
    stems={name:{q for _,q,_ in data} for name,data in legacy.items()}
    chosen={row["source_id"] for row in rows}
    matches=[]
    for row in rows:
        for arm in ("control","reverse"):
            prompt=row[arm+"_prompt"]; response=row[arm+"_response"]
            stem=qbody(prompt)
            for name,blocked in stems.items():
                if stem in blocked:
                    matches.append((row["id"],arm,name))
            opts=re.findall(r"^([A-D])\. (.+)$",prompt,re.M)
            assert [k for k,v in opts]==list("ABCD")
            assert [m[1] for m in re.finditer(r"(?m)^(Evidence|Rule|Application|Final answer):",response)]==["Evidence","Rule","Application","Final answer"]
            assert len(response.splitlines())==4 and response.splitlines()[-1][-1] in "ABCD"
    assert not matches,matches
    assert len(rows)==48 and len(chosen)==24 and receipt["sha256"]==sha((HERE/"direction_pairs.jsonl").read_bytes())
    for row in items:
        assert sha((HERE/row["image"]).read_bytes())==row["image_sha256"]
        assert len(set(row["alternate_options"]))==4 and set(row["original_options"])==set(row["alternate_options"])
        assert row["original_options"][ord(row["original_letter"])-65]==row["alternate_options"][ord(row["paraphrase_letter"])-65]==row["forward_value"]
        assert row["alternate_options"][ord(row["reverse_letter"])-65]==row["inverse_value"]!=row["forward_value"]
    literal={row[arm+"_prompt"].split("\nOptions:")[0] for row in rows for arm in ("control","reverse")}
    assert len(literal)==72
    out=dict(n=48,groups=24,unique_literal_directional_stems=len(literal),unique_normalized_directional_stems=len({qbody(row[arm+"_prompt"]) for row in rows for arm in ("control","reverse")}),
             exact_derived_stem_hits=matches,protected_sources={k:len(v) for k,v in legacy.items()},
             release_sha256=receipt["sha256"],verification="Stem check uses normalized training questions excluding option suffix. Removing punctuation collapses the two c_239 questions about A and A-prime; their literal prompts and answers are distinct. Every group uses four distinct options.")
    (HERE/"derived_prompt_verification.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(out)


if __name__=="__main__":
    main()
