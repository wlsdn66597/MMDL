"""Make a small native ScienceQA validation roster disjoint from the proposed teacher pool."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

import pyarrow.parquet as pq

from audit_sources import HERE, ROOT, checked_scope, collides, fingerprint, normalized, read, signatures

VAL_SHA = "e66b5215ce71e748ade4ee629bc14fbf86762071d355a4fcd831581cc04d72d8"
SKILLS = {
    "Read a map: cardinal directions", "Identify the Thirteen Colonies", "Compare magnitudes of magnetic forces",
    "Identify magnets that attract or repel", "Identify how particle motion affects temperature and pressure",
    "Compare concentrations of solutions", "Analyze data to compare properties of planets",
    "Compare strengths of magnetic forces", "Use Punnett squares to calculate ratios of offspring types",
    "Use climate data to make predictions", "Classify elementary substances and compounds using models",
    "Use lines of latitude and longitude", "Interpret food webs II", "Use a letter-number grid",
    "Interpret food webs I", "Identify directions of forces", "Compare ages of fossils in a rock sequence",
    "Identify appeals to ethos, pathos, and logos in advertisements", "Identify and compare air masses",
    "Read graphic organizers", "Identify chemical formulas for ball-and-stick models",
    "Match chemical formulas to ball-and-stick models", "Describe and graph water on Earth",
    "Diffusion across membranes", "Determine the meaning of domain-specific words with pictures",
}

def quality_exclusion(row):
    question = row["question"].casefold()
    if row["skill"] == "Analyze data to compare properties of planets":
        return "The question asks a general fact about the solar system without directing the solver to tabulated readings."
    if question.startswith("which i in") or question.startswith("which i is"):
        return "Official question is missing the noun being selected."
    if row["skill"] == "Classify elementary substances and compounds using models":
        return "The named substance in the question or hint can identify the composition without reading the picture."
    if row["skill"] == "Identify directions of forces" and ("pull" in question or "push" in question):
        return "The direction of the named push/pull follows from the hint and question without the diagram."
    if row["skill"] == "Use climate data to make predictions":
        claimed = re.findall(r'Choice "([^"]+)" is incorrect', row["solution"])
        answer = row["choices"][row["answer"]].lower()
        if any(claim.lower() in answer for claim in claimed):
            return "Source explanation calls the gold month incorrect even though the graph supports the gold choice."
    return ""


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf8")


def main(parquet, size):
    assert sha(parquet) == VAL_SHA
    sr = ROOT / "reports/era_dataset_examples_20261014"
    data, split = read(sr / "scienceqa_problems.json"), read(sr/"scienceqa_splits.json")
    train_ids = {r["scienceqa_id"]:r for r in read(HERE/"candidate_ids.json")}
    train_signatures = {}
    questions, refs = checked_scope()
    train = ROOT / "tmp/era_prereview_20261021/scienceqa_train.parquet"
    output = HERE / "native_validation"
    images = output / "images"
    images.mkdir(exist_ok=True,parents=True)
    queue, excluded = [], {}
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        pic = Path(temp) / "picture.png"
        index = 0
        for batch in pq.ParquetFile(train).iter_batches(batch_size=64,columns=["image"]):
            for row in batch.to_pylist():
                sid = str(split["train"][index])
                index += 1
                if sid in train_ids:
                    pic.write_bytes(row["image"]["bytes"])
                    fp = fingerprint(pic)
                    assert fp["pixel_sha256"] == train_ids[sid]["image_signature"]
                    train_signatures[sid] = signatures(fp)
        assert index == len(split["train"])
        index = 0
        counts = Counter()
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                sid = str(split["val"][index])
                index += 1
                m = data[sid]
                assert m["split"]=="val" and all(m[k]==row[k] for k in row if k!="image")
                if not m["image"] or not m["solution"].strip() or m["skill"] not in SKILLS:
                    continue
                counts["image_reference_and_skill"] += 1
                if normalized(m["question"]) in questions:
                    continue
                pic.write_bytes(row["image"]["bytes"])
                sig = signatures(fingerprint(pic))
                if collides(sig,refs):
                    continue
                counts["eligible_prior_to_internal_dedup"] += 1
                issue = quality_exclusion(m)
                if issue:
                    excluded[sid] = issue
                    continue
                key = hashlib.sha256(("era-retention-native-val-v1/"+sid).encode()).digest()
                queue.append((key,sid,row["image"]["bytes"],sig))
        assert index == len(split["val"]) == 4241
    accepted, known_qs, known_sigs, skills = [], set(), [], Counter()
    for key,sid,blob,sig in sorted(queue):
        m = data[sid]
        question = normalized(m["question"])
        if question in known_qs or collides(sig,known_sigs) or skills[m["skill"]]>=4:
            continue
        accepted.append((sid,blob,sig))
        skills[m["skill"]] += 1
        known_qs.add(question)
        known_sigs.append(sig)
        if len(accepted)==size:
            break
    assert len(accepted) > 0
    final_pool = [r for sid,r in train_ids.items() if normalized(data[sid]["question"]) not in known_qs
                  and not collides(train_signatures[sid], known_sigs)]
    dump(HERE/"candidate_ids_after_native_holdout.json",final_pool)
    inputs, gold = [], []
    for sid,blob,sig in accepted:
        m = data[sid]
        item_id = "sqa-val-"+sid
        img = images/(item_id+".png")
        img.write_bytes(blob)
        options = m["choices"]
        choices = "".join(f"{chr(65+i)}. {opt}\n" for i,opt in enumerate(options))
        hint = f"Hint: {m['hint']}\n" if m.get("hint") else ""
        prompt = (f"{hint}Question: {m['question']}\nOptions:\n{choices}Please select the correct answer from the options above.\n"
                  'On the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.')
        inputs.append(dict(id=item_id,prompt=prompt,image=img.relative_to(output).as_posix(),image_sha256=sha(img)))
        gold.append(dict(id=item_id,original_source_id=sid,option_letter=chr(65+m["answer"]),option_text=options[m["answer"]],
                         original_solution=m["solution"],subject=m["subject"],skill=m["skill"]))
    for file,out in (("inputs.json",inputs),("gold.json",gold)):
        dump(output/file,out)
    active = {output/r["image"] for r in inputs}
    for file in images.glob("sqa-val-*.png"):
        if file not in active:
            file.unlink()
    assert len(list(images.glob("*.png"))) == len(inputs)
    dump(output/"exclusions.json",excluded)
    dump(output/"receipt.json",dict(source_revision="f18b0a70359ebfb41f658fd564208d0355b013f4",
         parquet_sha256=VAL_SHA, native_rows=index, counts=counts,selected=len(inputs),skills=skills,
         sha256_inputs=sha(output/"inputs.json"),sha256_gold=sha(output/"gold.json"),
         candidates_before_reservation=len(train_ids),candidates_after_reservation=len(final_pool),
         disjoint_train_roles=dict(Counter(r["role"] for r in final_pool)), model_outputs_used=0))
    print((output/"receipt.json").read_text(encoding="utf8"))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,default=HERE/"validation_source.parquet")
    parser.add_argument("--size",type=int,default=24)
    args=parser.parse_args()
    main(args.source,args.size)
