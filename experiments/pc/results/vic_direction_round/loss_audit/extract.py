"""Extract the fixed paired cohort and make image contact sheets for review."""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1]
BASE = RESULTS / "mmmu_first_baseline_20261009/baseline_seed3407/results.json"
MODEL = HERE.parent / "mmmu/reverse/full/results.json"
IMAGES = RESULTS / "mmmu_first_baseline_20261009/analysis/images"
PROVENANCE = RESULTS / "mmmu_first_baseline_20261009/provenance/input-validation.json"


def load(path, digest):
    blob = path.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == digest
    return json.loads(blob)["records"]


def main():
    base = load(BASE, "3791f54f30b9e127fa995f2f442ab0460c988addb069377596ea5830a7a8f0d8")
    model = load(MODEL, "f2c7adbe44ee7debe07f5aa7182eff77c21a91bbaf1a4f4e13b722075ba02cff")
    pixels=PROVENANCE.read_bytes()
    assert hashlib.sha256(pixels).hexdigest()=="eafe5603b1de1af895d55a431a8792f7e53f700b89dcae2ecaf5ad9ff0ac6b74"
    reference={(r["id"],r["field"]):r for r in json.loads(pixels)["images"]}
    assert len({r["id"] for r in base}) == len(base) == len(model) == 900
    cards = []
    for old, new in zip(base, model):
        for field in ("id","question","options","reference_answer","image_fields","input_prompt"):
            assert old[field] == new[field], (old["id"],field)
        if not all(row["model_response"]["finish_reason"] == "stop" for row in (old,new)):
            continue
        if not old["grading"]["final"]["hit"] or new["grading"]["final"]["hit"]:
            continue
        files = [IMAGES / (new["id"] + "_" + field + ".png") for field in new["image_fields"]]
        assert files and all(p.is_file() for p in files)
        for field,path in zip(new["image_fields"],files):
            ref=reference[new["id"],field]
            with Image.open(path) as im:
                assert list(im.size) == ref["size_wh"]
                assert hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()==ref["rgb_bytes_sha256"],path
        cards.append(dict(idx=len(cards)+1,id=new["id"],subject=new["subject"],question=new["question"],
                          options=new["options"],gold=new["reference_answer"],raw=new["model_response"]["raw"],
                          choice=new["grading"]["final"]["extracted_answer"],method=new["grading"]["final"]["method"],
                          images=[p.relative_to(RESULTS.parent).as_posix() for p in files]))
    assert len(cards) == len({c["id"] for c in cards}) == 148
    (HERE/"cards.jsonl").write_text("".join(json.dumps(c,ensure_ascii=False)+"\n" for c in cards),encoding="utf8")
    (HERE/"sheets").mkdir(exist_ok=True)
    for start in range(0,len(cards),4):
        page=Image.new("RGB",(1280,1000),"white")
        draw=ImageDraw.Draw(page)
        for pos,c in enumerate(cards[start:start+4]):
            left,top=640*(pos%2),500*(pos//2)
            draw.text((left+4,top+4),f"{c['idx']:03d} {c['id']} ({len(c['images'])} image/s)",fill="black")
            for k,f in enumerate(c["images"]):
                width=640//len(c["images"])
                with Image.open(RESULTS.parent/f) as im:
                    im=ImageOps.contain(im.convert("RGB"),(width-6,473))
                    page.paste(im,(left+k*width+(width-im.width)//2,top+21+(473-im.height)//2))
        page.save(HERE/"sheets"/f"w{start+1:03d}.jpg",quality=88)


if __name__=="__main__":
    main()
