"""Read-only audit of the supplied corpus. The source ZIP is never extracted."""
from collections import Counter, defaultdict
from pathlib import Path
from io import BytesIO
import hashlib
import json
import re
import sys
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
KEY = "MMDL-vic2k-content-audit-v1/"
HEAD = re.compile(r"(?im)^(Evidence|Rule|Application|Final answer):[ \t]*")


def rank(key):
    return hashlib.sha256((KEY+key).encode()).hexdigest()


def stem(question):
    return re.sub(r"\s+"," ",question.splitlines()[0]).strip().lower()


def run(path):
    with ZipFile(path) as archive:
        rows=json.loads(archive.read("data.json"))
        groups=defaultdict(list)
        forms=Counter()
        exact=[]
        subsets=defaultdict(list)
        for row in rows:
            source="MMFineReason" if row["mmfinereason_source"] else "OpenMedReason"
            group=(source,row["vic_type"],row["question_type"])
            groups[group].append(row)
            markers=list(HEAD.finditer(row["explanation_era"]))
            forms[tuple(m[1] for m in markers)]+=1
            final=row["explanation_era"][markers[-1].end():].strip()
            if row["answer"].strip()!=final:
                exact.append(dict(id=row["id"],answer=row["answer"],era_final=final))
            if source=="OpenMedReason":
                subsets[tuple(row["options"])].append(row)
        samples=[r for group,items in sorted(groups.items()) for r in sorted(items,key=lambda item:rank(item["id"]))[:2]]
        canonical=[(r["id"],r["vic_type"],r["question_type"],(r["mmfinereason_source"] or {}).get("source","OpenMedReason")) for r in samples]
        receipt={"sampling":"two lowest SHA256('"+KEY+"'+id) in each populated top-source x V/I/C x MC/open cell",
                 "corpus_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"rows":len(rows),
                 "strata":{"|".join(k):len(v) for k,v in sorted(groups.items())},
                 "header_sequences":{"|".join(k):v for k,v in forms.items()},
                 "literal_final_exceptions":exact,"sample":canonical}
        med=[r for r in rows if not r["mmfinereason_source"]]
        receipt["medical_letters"]=dict(Counter(r["answer"] for r in med))
        receipt["normal_abnormal_option_order"]={}
        for choices in (("Normal","Abnormal"),("Yes","No"),("True","False")):
            receipt["normal_abnormal_option_order"]["|".join(choices)]=dict(Counter(r["answer"] for r in subsets[choices]))
        repeated=defaultdict(list)
        for row in med:
            repeated[stem(row["question"])].append(row)
        receipt["repeated_medical_stems"]=[
            {"stem":s,"n":len(rs),"correct_letters":dict(Counter(r["answer"] for r in rs))}
            for s,rs in sorted(repeated.items(),key=lambda t:-len(t[1]))[:12]]
        # Reproducible out-of-fold text-only challenge, without peeking at test labels.
        # A stem memorizer falls back to the most frequent training label.
        folds=[sorted(med,key=lambda r:rank("probe/"+r["id"]))[x::5] for x in range(5)]
        probe=[]; y_total=0; y_covered=0; covered=0
        for f in range(5):
            train=[r for k in range(5) if k!=f for r in folds[k]]
            test=folds[f]; train_stem=defaultdict(Counter)
            for r in train:
                train_stem[stem(r["question"])][r["answer"]]+=1
            dist=Counter(r["answer"] for r in train)
            majority=sorted(dist,key=lambda l:(-dist[l],l))[0]
            for r in test:
                c=train_stem.get(stem(r["question"]),None)
                prediction=sorted(c,key=lambda l:(-c[l],l))[0] if c else majority
                hit=prediction==r["answer"]; y_total+=hit
                y_covered+=bool(c) and hit; covered+=bool(c)
                probe.append(dict(id=r["id"],fold=f,seen_stem=bool(c),prediction=prediction,hit=hit))
        receipt["medical_text_probe"]={"rule":"Exact first-line normalized stem lookup; fallback to training fold majority letter; lexicographic ties; five fixed hash folds. Excludes all images and option text. Measures within-corpus sampling artifacts.","n":len(med),"hits":y_total,"stem_seen_n":covered,"stem_seen_hits":y_covered}
        receipt["medical_descriptive_shortcut_hits"]=sum(
            r["answer"]==("B" if r["options"]==["Normal","Abnormal"] else "A") for r in med)
        receipt["upstream_think_markup_ids"]=[r["id"] for r in rows if "<think>" in r["explanation"].lower()]
        receipt["era_literally_references_original_explanation"]=[r["id"] for r in rows if "original explanation" in r["explanation_era"].lower()]
        receipt["era_question_mark_ids"]=[r["id"] for r in rows if "?" in r["explanation_era"]]
        (HERE/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf8")
        (HERE/"text_probe.json").write_text(json.dumps(probe,indent=2),encoding="utf8")
        (HERE/"sample.json").write_text(json.dumps(samples,indent=2,ensure_ascii=False),encoding="utf8")
        view=HERE/"panels";view.mkdir(exist_ok=True)
        for i in range(0,len(samples),4):
            canvas=Image.new("RGB",(1200,880),"white")
            draw=ImageDraw.Draw(canvas)
            for j,r in enumerate(samples[i:i+4]):
                x,y=600*(j%2),440*(j//2)
                with Image.open(BytesIO(archive.read(r["image_1"]))) as im:
                    im=ImageOps.contain(im.convert("RGB"),(590,405))
                    canvas.paste(im,(x+(600-im.width)//2,y+27))
                draw.text((x+6,y+5),r["id"],fill="black")
            canvas.save(view/f"sample_{i//4}.jpg",quality=95)


if __name__=="__main__":
    run(Path(sys.argv[1]))
