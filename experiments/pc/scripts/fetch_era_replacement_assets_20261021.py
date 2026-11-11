"""Download a pinned public source, verify it, and extract only selected image bytes."""
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".runtime/era_review_deps"))
import pyarrow.parquet as pq
from PIL import Image

OUT = ROOT / "tmp/era_prereview_20261021"
rev = "f18b0a70359ebfb41f658fd564208d0355b013f4"
url = f"https://huggingface.co/datasets/derek-thomas/ScienceQA/resolve/{rev}/data/train-00000-of-00001-1028f23e353fbe3e.parquet"
cache = OUT / "scienceqa_train.parquet"
if not cache.exists():
    partial = cache.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=45) as src, partial.open("wb") as dst:
        while chunk := src.read(4 * 1024 * 1024):
            dst.write(chunk)
    partial.replace(cache)
digest = hashlib.file_digest(cache.open("rb"), "sha256").hexdigest()
assert digest == "62c90a28e3fb1bc0ad7bbcab1ac62b483ae6758291a655944d8f494bf6445745"
table = pq.read_table(cache)
source = json.loads((ROOT/"reports/era_dataset_examples_20261014/scienceqa_problems.json").read_text(encoding="utf-8"))
train = json.loads((ROOT/"reports/era_dataset_examples_20261014/scienceqa_splits.json").read_text(encoding="utf-8"))["train"]
candidates = json.loads((OUT/"replacement_candidates.json").read_text(encoding="utf-8"))
manifest=[]
for candidate in candidates:
    sid=candidate["source_id"]
    row_index=[str(x) for x in train].index(sid)
    row=table.slice(row_index,1).to_pylist()[0]
    for field in ["question","choices","answer","hint","lecture","solution","task","grade","subject","topic","category","skill"]:
        assert row[field]==source[sid][field], (sid,field)
    blob=row["image"]["bytes"]
    image=Image.open(io.BytesIO(blob))
    assert image.format=="PNG"
    path=OUT/f"{sid}.png"
    path.write_bytes(blob)
    manifest.append(dict(source_id=sid,row_index=row_index,path=path.name,sha256=hashlib.sha256(blob).hexdigest(),
                         size=list(image.size),format=image.format,metadata_exact_match=True))
report=dict(mirror_repository="derek-thomas/ScienceQA",revision=rev,url=url,parquet_sha256=digest,
            images=manifest,image_bytes="embedded original bytes, without conversion or resizing",
            limitation="Matched official metadata exactly; official Google Drive archive byte equivalence not independently checked.")
(OUT/"replacement_asset_provenance.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print(json.dumps(report,indent=2))
