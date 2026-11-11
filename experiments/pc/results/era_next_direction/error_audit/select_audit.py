"""Select fresh MMMU disagreement cases without using the response text to choose them."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
RESULTS = OUT.parents[1]
R4 = RESULTS / "era_a_pair_round"
BASE = RESULTS / "mmmu_first_baseline_20261009"
SEED = "era-next-error-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return {r["id"]:r for r in read(path)["records"]}


def write(name, body):
    (OUT/name).write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")


def family(base, control, pair):
    if base and not pair:
        return "new_loss" if control else "shared_loss"
    if control and not pair:
        return "other_loss"
    if pair and not control:
        return "pair_gain"
    return None


def main():
    base = rows(BASE / "baseline_seed3407/results.json")
    control = rows(R4 / "control_recovery/score/results.json")
    pair = rows(R4 / "mmmu/candidate/candidate/full/results.json")
    images = read(BASE / "analysis/images/index.json")
    seen = {r["id"] for r in read(R4/"reviews/blind_key.json")}
    seen.update(r["id"] for r in read(R4/"reviews/change_key.json").values())
    previous_images = {f["rgb_bytes_sha256"] for rid in seen for f in images[rid]}
    armrows = dict(base=base,control=control,pair=pair)
    hit = lambda r:r["grading"]["final"]["hit"]
    cases = {k:[] for k in ("shared_loss","new_loss","other_loss","pair_gain")}
    totals = Counter()
    for rid in sorted(control):
        fam = family(hit(base[rid]),hit(control[rid]),hit(pair[rid]))
        if fam is None:
            continue
        totals[fam]+=1
        if rid not in seen and not any(f["rgb_bytes_sha256"] in previous_images for f in images[rid]):
            cases[fam].append(rid)
    picks = []
    counts = dict(shared_loss=5,new_loss=5,other_loss=3,pair_gain=5)
    image_seen=set(previous_images)
    for fam,n in counts.items():
        subject_seen=set()
        for rid in sorted(cases[fam],key=lambda rid:hashlib.sha256(f"{SEED}/{fam}/{rid}".encode()).digest()):
            subject=base[rid]["subject"]
            hashes = {f["rgb_bytes_sha256"] for f in images[rid]}
            if subject in subject_seen or hashes & image_seen:
                continue
            subject_seen.add(subject)
            image_seen.update(hashes)
            picks.append(dict(id=rid,family=fam))
            if len(subject_seen)==n:
                break
        assert len(subject_seen)==n
    write("selection.json",dict(seed=SEED,stratum_sizes=totals,eligible={k:len(v) for k,v in cases.items()},picks=picks,
                                exclusion_ids=sorted(seen),sample_sizes=counts))
    cards=[]
    for item in picks:
        rid=item["id"]
        # Show both matched arms for comparison. The labels below identify the comparator
        # but are not used to pick the items or to establish official correctness.
        cards.append(dict(**item,question=base[rid]["question"],options=base[rid]["options"],gold=base[rid]["reference_answer"],
                          images=[r["file"] for r in images[rid]],
                          solutions={a:dict(raw=t[rid]["model_response"]["raw"],
                                            finish=t[rid]["model_response"]["finish_reason"],hit=hit(t[rid]))
                                     for a,t in armrows.items()}))
    write("cards.json",cards)
    # Count literal abstentions; do not treat every use of NONE in prose as a final answer.
    literal=re.compile(r"(?im)^\s*(?:\*\*)?Final answer(?:\*\*)?\s*:\s*(?:\*\*)?(?:none(?:\s+of\s+(?:the\s+)?(?:above|options|choices))?|N/A)[.!*\s]*\Z")
    stats={}
    for a,lookup in armrows.items():
        ids=[rid for rid in control if literal.search(lookup[rid]["model_response"]["raw"])]
        stats[a]=dict(count=len(ids),wrong=sum(not hit(lookup[rid]) for rid in ids),ids=ids)
    write("literal_none.json",stats)
    mentions = re.compile(r"\b(?:options?|choices?|answers?|select|choose|therefore|hence)\s*(?:(?:would be|must be|is|should be)\s*)?[:=]?\s*\(?([A-H])\b")
    explicit = []
    screened = 0
    for rid in sorted(control):
        row = pair[rid]
        if not hit(base[rid]) or hit(row) or row["question_type"]!="multiple-choice":
            continue
        raw = row["model_response"]["raw"]
        chunks = re.split(r"(?im)^\s*(?:\*\*)?Application(?:\*\*)?\s*:",raw)
        if len(chunks)==1:
            continue
        app = re.split(r"(?im)^\s*(?:\*\*)?Final answer(?:\*\*)?\s*:",chunks[-1])[0]
        screened += 1
        if row["reference_answer"] in mentions.findall(app):
            explicit.append(dict(id=rid, gold=row["reference_answer"],application_start=app.strip()[:320],
                                 application_end=app.strip()[-320:],application_characters=len(app),
                                 official=row["grading"]["final"]["extracted_answer"],finish=row["model_response"]["finish_reason"]))
    write("wrong_final_gold_mention.json",dict(screened=screened,rule=mentions.pattern,hits=explicit))


if __name__=="__main__":
    main()
