"""Read-only structural audit; simulates safe MC prompt construction, without training or extracting images."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
IMG=re.compile(r"<image>",re.I)
SECTIONS=re.compile(r"^Evidence: +(.+?)\nRule: +(.+?)\nApplication: +(.+?)\nFinal answer: +(.+)$",re.S)


def norm(x):
    return " ".join(x.split()).strip()


def option_suffix(body,opts):
    """Extract a consecutive exact-option suffix. Leave a source that cannot be proven intact for manual review."""
    parts=[]
    for i,opt in enumerate(opts):
        letter=chr(65+i)
        # Case-sensitive labels: alphabetical variables in the stem are not accepted unless every
        # consecutive match also includes the full offered option text and exhausts the suffix.
        label=rf"(?:\({letter}\)[.：:]?|\[{letter}\][.：:]?|{letter}[.):：]|{letter}\s*[、．])"
        text=re.escape(norm(opt)).replace(r"\ ",r"\s+")
        parts.append(label+r"\s*"+text)
    pattern=r"(?<![\w])"+r"\s*(?:,\s*)?".join(parts)+r"\s*[).]?\s*$"
    found=re.search(pattern,body)
    if not found:
        return None
    stem=re.sub(r"(?:\(?\s*(?:Options?|Choices?)\s*[:：]\s*)$","",body[:found.start()],flags=re.I).strip()
    return stem or None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("zip",type=Path)
    args=parser.parse_args()
    with ZipFile(args.zip) as z:
        raw=z.read("data.json")
        rows=json.loads(raw)
    stats=Counter()
    ids=defaultdict(list)
    sources=Counter()
    types=Counter()
    letters=defaultdict(Counter)
    by_mm_id=defaultdict(list)
    for row in rows:
        rid=row["id"];src=row["mmfinereason_source"];source=src["source"] if src else "OpenMedReason"
        qtype=row["question_type"];q=row["question"];opts=row["options"];kind=row["vic_type"];gold=row["answer"]
        sources[(source,kind,qtype)]+=1;types[(kind,qtype)]+=1
        if src:
            by_mm_id[src["id"]].append((rid,src["source"],src["row_index"]))
        if not row["original_source"]["id"]:
            stats["missing_original_source_item_id"]+=1
        if "<think>" in row["explanation"]:
            stats["source_explanation_contains_think_tags"]+=1
        if "First perform reasoning, then finally select the question" in q:
            ids["embedded_answer_format_instruction"].append(rid)
        if "\n" in gold.strip():
            ids["answer_multiline"].append(rid)
        qr=IMG.sub("",q).strip()
        if qr!=q:
            stats["question_contains_image_marker"]+=1
        match=SECTIONS.fullmatch(row["explanation_era"].strip())
        if match:
            stats["strict_ERA_parse"]+=1
            if "\n" in match.group(4).strip():
                ids["ERA_final_content_multiline"].append(rid)
            if match.group(4).strip()==gold.strip():
                stats["strict_final_equals_answer_exact"]+=1
            else:
                ids["strict_final_diff"].append(rid)
        else:
            ids["strict_ERA_parse_fail"].append(rid)
        if qtype=="open":
            if re.fullmatch(r"[A-H]",gold):
                ids["open_with_single_choice_letter_answer"].append(rid)
            if re.search(r"(?m)^\s*[A-D][.\)]\s",qr):
                ids["open_with_line_initial_options"].append(rid)
            continue
        assert opts and gold in "ABCDEFGH"[:len(opts)]
        letters["MMFineReason" if src else "OpenMedReason"][gold]+=1
        if all(norm(o) in norm(qr) for o in opts):
            stats["mc_all_option_strings_in_question"]+=1
        else:
            ids["mc_one_or_more_option_strings_missing"].append(rid)
        if option_suffix(qr,opts):
            stats["mc_exact_option_suffix_strip_proved"]+=1
        else:
            ids["mc_suffix_needs_review"].append(rid)
    upstream_groups=Counter("MMFineReason" if r["mmfinereason_source"] else "OpenMedReason" for r in rows)
    out=dict(zip_sha256=hashlib.sha256(args.zip.read_bytes()).hexdigest(),data_json_sha256=hashlib.sha256(raw).hexdigest(),
             n=len(rows), source_counts=upstream_groups, statistics=stats,types={"/".join(k):v for k,v in types.items()},
             source_types={"/".join(k):v for k,v in sources.items()},letter_distribution=letters,
             source_id_collisions={str(i):r for i,r in by_mm_id.items() if len(r)>1},
             ids_to_review=dict(ids),note="Consecutive exact textual suffix proves only text alignment. It does not establish correct labels, question sufficiency or whether an open letter refers to options drawn on an image.")
    (HERE/"contract.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(stats,{k:len(v) for k,v in ids.items()},letters)


if __name__=="__main__":
    main()
