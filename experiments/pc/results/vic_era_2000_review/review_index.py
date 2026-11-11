"""Build an intake queue, not labels or a new training split, from the ZIP and overlap evidence."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
ORDER_CUE=re.compile(r"\bratio\b|\b(?:counterclockwise|clockwise|anticlockwise|displacement|momentum|relative to|before and after|after and before|younger|older|ascending|descending)\b",re.I)
CHECKED={
    "v_22":"Reused donor D_ScienceQA_1022 map and question, changed distractors; already seen training group.",
    "v_219":"Reused donor D_ScienceQA_10518 US map, new question; already seen visual group.",
    "c_380":"Visually checked ordered-ratio augmentation example; recheck original teacher if used for preservation.",
    "v_106":"Graph shows before/after prices at the demand intersections; candidate for a reversed-order companion.",
    "c_170":"Named segments are not corresponding sides; actual GM/KL is 18/18 while the similarity scale is 2/3. Review wording and answer.",
    "c_38":"Sample review verified counts and percentage-change denominator.",
    "c_52":"The provided option D contains a malformed mathematical expression.",
    "c_767":"The question text suffices; only axes are visible in the picture.",
    "v_84":"Printed labels in the image: treat as OCR if used.",
    "v_153":"The answer text is printed on the image: OCR use.",
}
CONFIRMED_INTERNAL=[("c_51","c_716"),("c_483","c_854"),("c_743","c_847"),("v_202","v_291")]


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def main(path):
    with ZipFile(path) as z:
        rows=json.loads(z.read("data.json"))
    internal=read(HERE/"overlap/internal_groups.json")
    near={}
    for component in internal["perceptual_image"]:
        for rid in component:
            near[rid]=dict(id=min(component),size=len(component))
    matches=defaultdict(list)
    for c in read(HERE/"overlap/cross_candidates.json"):
        matches[c["id"]].append(c)
    stem_sets=defaultdict(set)
    for entry in read(HERE/"overlap/question_stems.json")["matches"]:
        for pair in entry["exact"].values():
            for rid in pair["vic"]:
                stem_sets[rid].add(entry["set"])
    profile=read(HERE/"profile.json")
    final_review=set(profile["flags"]["nonidentical_terminal_text"])
    cue_review=set(profile["flags"]["possible_answer_only_instruction"])
    total=Counter()
    entries=[]
    for row in rows:
        rid=row["id"]
        mm=row.get("mmfinereason_source")
        flags=[]
        if rid in final_review:flags.append("final_answer_text_differs")
        if rid in cue_review:flags.append("answer_only_wording")
        if "original explanation" in row["explanation_era"].lower():flags.append("references_source_explanation")
        if mm is None:flags.append("medical_source_option_bias_review")
        if near[rid]["size"]>1:flags.append("possible_internal_image_relative")
        if matches[rid]:flags.append("cross_corpus_candidate_consult_visual_adjudication")
        if stem_sets[rid]:flags.append("prior_question_stem_match")
        if rid in ("v_22","v_219"):flags.append("confirmed_old_training_image")
        if any(rid in pair for pair in CONFIRMED_INTERNAL):flags.append("confirmed_internal_shared_image_family")
        if rid in ("c_170","c_52","c_767"):flags.append("specific_content_issue")
        if rid in ("v_84","v_153"):flags.append("printed_label_reading")
        if ORDER_CUE.search(row["question"]):flags.append("relation_word_in_question")
        total.update(flags)
        entries.append(dict(id=rid,type=row["vic_type"],question_type=row["question_type"],
                            group_hint=near[rid]["id"],group_size=near[rid]["size"],
                            source=(mm["source"] if mm else (row["original_source"] or {}).get("dataset")),
                            cross_corpora=sorted({x["set"] for x in matches[rid]} | stem_sets[rid]), review_flags=flags,
                            sample_note=CHECKED.get(rid)))
    assert len(entries)==len(near)==2000
    (HERE/"review_index.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in entries),encoding="utf8")
    (HERE/"review_index_summary.json").write_text(json.dumps(dict(rows=len(entries),flags=total,
        meaning="Flags overlap. Image relative and cross-corpus matches are algorithmic review candidates. Missing flags are not semantic clearance; group hints connect near images, not all template/source relations."),
        ensure_ascii=False,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("zip",type=Path)
    main(a.parse_args().zip)
