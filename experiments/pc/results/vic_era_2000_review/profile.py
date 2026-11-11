"""Read the supplied ZIP as data; profile its fit to the existing Qwen ERA targets."""
import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import statistics
import sys
import zipfile

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / ".runtime/tokenizers"))
from tokenizers import Tokenizer

TARGET = re.compile(r"\AEvidence:\s*(.*?)\s*\nRule:\s*(.*?)\s*\nApplication:\s*(.*?)\s*\nFinal answer:\s*(.*?)\s*\Z", re.S)
TOKENIZER_SHA = "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def example_pair(archive, rows):
    out = HERE/"examples"
    out.mkdir(exist_ok=True)
    indexed = {row["id"]:row for row in rows}
    for rid in ("c_170","c_380","v_106"):
        rel = indexed[rid]["image_1"]
        (out/Path(rel).name).write_bytes(archive.read(rel))
    row = indexed["c_380"]
    options = ["5:11", "11:5", "5:6", "6:5"]
    spec = [("shaded area to the entire area", "5:11"),("entire area to the shaded area", "11:5")]
    # The same new options can express either ratio; each question is tested in two orders.
    variants = []
    for direction, answer in spec:
        for arrangement in (options, options[1:]+options[:1]):
            variants.append(dict(question=f"In figure 10 of the illustrated pattern, what is the ratio of the {direction}?",
                                 options=arrangement, answer=chr(65+arrangement.index(answer)),
                                 reference_value=answer))
    assert 10*11//2 == 55 and 11**2 == 121 and len(variants)==4
    write(HERE/"examples/c_380_order_pair.json",
          dict(source_id=row["id"], image="c_380.jpg", image_sha256=digest(archive.read(row["image_1"])),
               role="proposed training/review example; new questions and options, never split this source image group",
               common_evidence="The four diagrams divide each side into 2, 3, 4 and 5 parts; shaded-cell counts are 1, 3, 6 and 10.",
               common_intermediate="Figure 10 has 121 equal cells and 55 shaded cells. Keep total and shaded counts bound to their labels.",
               variants=variants))


def main(path):
    tokenizer_file = ROOT/".runtime/qwen_tokenizer/tokenizer.json"
    assert digest(tokenizer_file.read_bytes()) == TOKENIZER_SHA
    tok = Tokenizer.from_file(str(tokenizer_file))
    groups = defaultdict(Counter)
    lengths = defaultdict(list)
    flagged = defaultdict(list)
    with zipfile.ZipFile(path) as archive:
        rows=json.loads(archive.read("data.json"))
        assert len(rows)==len({r["id"] for r in rows})==2000
        for row in rows:
            source = "MMFineReason" if row["mmfinereason_source"] else "OpenMedReason"
            category = row["vic_type"]
            groups["type"][category]+=1
            groups["category_source"][category+"/"+source]+=1
            groups["category_question_type"][category+"/"+row["question_type"]]+=1
            prompt=row["question"]
            target=row["explanation_era"]
            fields=TARGET.fullmatch(target)
            assert fields and all(fields.groups()),row["id"]
            for name, data in (("question",prompt),("era",target),("source_explanation",row["explanation"])):
                count=len(tok.encode(data,add_special_tokens=False).ids)
                lengths[name].append((count,row["id"]))
            if fields[4]!=row["answer"].strip():
                flagged["nonidentical_terminal_text"].append(row["id"])
            if re.search(r"(?i)just (?:write|give|output) (?:the )?(?:answer|result)|do not (?:show|write) (?:the|your|any) (?:work|solution|reasoning)",prompt):
                flagged["possible_answer_only_instruction"].append(row["id"])
            if row["options"]:
                groups["mc_gold_"+source][row["answer"]]+=1
                groups["mc_gold_"+category][row["answer"]]+=1
            with Image.open(BytesIO(archive.read(row["image_1"]))) as im:
                if max(im.size)/min(im.size)>200:
                    flagged["extreme_aspect_ratio"].append(row["id"])
        example_pair(archive,rows)
    summary={}
    for k,entries in lengths.items():
        s=sorted(n for n,_ in entries)
        summary[k]=dict(median=statistics.median(s),p95=s[1899],max=s[-1],max_id=max(entries)[1])
    write(HERE/"profile.json",dict(zip_sha256=digest(path.read_bytes()),rows=2000,counts=groups,
            token_counts_text_only=summary,tokenizer_sha256=TOKENIZER_SHA,flags=flagged,
            note="Counts exclude image/chat-template tokens; this is a structural check, not a training approval."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("zip",type=Path)
    main(p.parse_args().zip)
