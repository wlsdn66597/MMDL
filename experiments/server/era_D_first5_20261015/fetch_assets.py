"""Run on practice PC; download pinned public mirror and extract original image bytes."""
import json, hashlib, urllib.request, io, zipfile
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent
def sha(b): return hashlib.sha256(b).hexdigest()
def get(url):
    with urllib.request.urlopen(url, timeout=120) as r: return r.read()
meta = json.loads(get('https://huggingface.co/api/datasets/derek-thomas/ScienceQA'))
rev = meta['sha']
(ROOT/'mirror_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
name = 'data/train-00000-of-00001-1028f23e353fbe3e.parquet'
url = f'https://huggingface.co/datasets/derek-thomas/ScienceQA/resolve/{rev}/{name}'
dest = ROOT/'scienceqa_train.parquet'
if not dest.exists():
    print('Downloading pinned train parquet', rev, flush=True)
    urllib.request.urlretrieve(url, dest)
table = pq.read_table(dest)
targets = json.loads((ROOT/'asset_targets.json').read_text(encoding='utf-8'))
manifest=[]
with zipfile.ZipFile(ROOT/'assets.zip','w',zipfile.ZIP_DEFLATED) as z:
    for target in targets:
        row = table.slice(target['row_index'],1).to_pylist()[0]
        src = target['original']
        for key in ['question','choices','answer','hint','lecture','solution','task','grade','subject','topic','category','skill']:
            assert row[key] == src[key], (target['source_id'],key)
        b = row['image']['bytes']
        image = Image.open(io.BytesIO(b))
        ext = {'PNG':'png','JPEG':'jpg'}.get(image.format)
        assert ext, image.format
        path=f"images/{target['source_id']}.{ext}"
        z.writestr(path,b)
        manifest.append(dict(source_id=target['source_id'],row_index=target['row_index'],path=path,
            sha256=sha(b),pixel_sha256=sha(image.convert('RGB').tobytes()),size=list(image.size),
            format=image.format,mirror_revision=rev,mirror_file=name,metadata_exact_match=True,
            original_image_name=src['image']))
    z.writestr('image_manifest.json',json.dumps(manifest,indent=2))
report={'mirror_repository':'derek-thomas/ScienceQA','revision':rev,'url':url,
    'parquet_sha256':sha(dest.read_bytes()),'rows':table.num_rows,'images_extracted':len(manifest),
    'image_bytes':'embedded bytes, no viewer JPEG conversion or resize',
    'limitation':'Official JSON fields matched exactly; official Google Drive image archive byte equivalence not independently checked.'}
(ROOT/'asset_acquisition.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
