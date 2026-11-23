"""선별 문항과 MMMU의 문제·이미지 중복 검사 (D 묶음과 같은 방법).

방법: 정규화한 문제 문장 일치, 단어 Jaccard >= 0.8, RGB 크기+픽셀 SHA256 일치, 64비트 dHash 거리 <= 4.
dHash 후보는 단순한 그림에서 오탐이 많으므로 사람이 두 이미지를 보고 판정한다 (--dump 폴더에 나란히 저장).

사용 예 (서버)
  ~/mmdl/mmdl_test/.venv/bin/python mmmu_overlap.py --selected work/C_v9/selected_questions.jsonl \
      --images ~/mmdl/scienceqa_images --splits validation dev test --out work/C_v9/mmmu_overlap.json --dump /tmp/mmmu_cand
"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
from pathlib import Path

REPO, REV = "MMMU/MMMU", "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
DHASH_MAX, JACCARD_MIN = 4, 0.8


def norm(t):
    return " ".join(re.findall(r"[a-z0-9]+", re.sub(r"<image \d+>", " ", str(t).lower())))


def jaccard(a, b):
    a, b = set(a.split()), set(b.split())
    return len(a & b) / len(a | b) if a and b else 0.0


def pix_hash(im):
    im = im.convert("RGB")
    return hashlib.sha256(f"{im.size}".encode() + im.tobytes()).hexdigest()


def dhash(im):
    from PIL import Image
    g = im.convert("L").resize((9, 8), Image.LANCZOS)
    p = list(g.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (p[y * 9 + x] > p[y * 9 + x + 1])
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--splits", nargs="+", default=["validation"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", help="dHash 후보 이미지를 나란히 저장할 폴더")
    a = ap.parse_args()
    import pandas as pd
    from PIL import Image

    rows = [json.loads(l) for l in open(a.selected, encoding="utf-8") if l.strip()]
    ours = []
    for r in rows:
        im = Image.open(Path(a.images).expanduser() / r["image_path"])
        ours.append({"source_id": r["source_id"], "q": norm(r["question"]), "pix": pix_hash(im), "dh": dhash(im),
                     "path": str(Path(a.images).expanduser() / r["image_path"])})
    snap = os.path.expanduser(f"~/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/{REV}")
    report = {"dataset": {"repo": REPO, "revision": REV}, "selected": len(rows),
              "methods": ["normalized exact question", f"word Jaccard >= {JACCARD_MIN}", "RGB size+pixel SHA256",
                          f"64-bit dHash distance <= {DHASH_MAX}"], "splits": {}}
    if a.dump:
        Path(a.dump).mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        files = sorted(glob.glob(f"{snap}/*/{split}-*.parquet"))
        nq = ni = 0
        exact_t, near_t, exact_i, near_i = [], [], [], []
        for f in files:
            df = pd.read_parquet(f)
            for _, m in df.iterrows():
                nq += 1
                mq = norm(m["question"])
                for o in ours:
                    if o["q"] == mq:
                        exact_t.append({"source_id": o["source_id"], "mmmu_id": m["id"]})
                    elif jaccard(o["q"], mq) >= JACCARD_MIN:
                        near_t.append({"source_id": o["source_id"], "mmmu_id": m["id"],
                                       "jaccard": round(jaccard(o["q"], mq), 3)})
                for k in range(1, 8):
                    cell = m.get(f"image_{k}")
                    if cell is None or not isinstance(cell, dict) or not cell.get("bytes"):
                        continue
                    ni += 1
                    mim = Image.open(io.BytesIO(cell["bytes"]))
                    ph, dh = pix_hash(mim), dhash(mim)
                    name = f"{m['id']}_image_{k}"
                    for o in ours:
                        if o["pix"] == ph:
                            exact_i.append({"source_id": o["source_id"], "mmmu_image": name})
                        d = bin(o["dh"] ^ dh).count("1")
                        if d <= DHASH_MAX:
                            near_i.append({"source_id": o["source_id"], "mmmu_image": name, "dhash_distance": d})
                            if a.dump:
                                a_im = Image.open(o["path"]).convert("RGB")
                                b_im = mim.convert("RGB")
                                h = 300
                                a_im = a_im.resize((max(1, a_im.width * h // a_im.height), h))
                                b_im = b_im.resize((max(1, b_im.width * h // b_im.height), h))
                                s = Image.new("RGB", (a_im.width + b_im.width + 20, h), "white")
                                s.paste(a_im, (0, 0))
                                s.paste(b_im, (a_im.width + 20, 0))
                                s.save(Path(a.dump) / f"C_{o['source_id']}__{name}.png")
        report["splits"][split] = {"questions_compared": nq, "images_compared": ni, "parquet_files": len(files),
                                   "exact_text_matches": exact_t, "near_text_candidates": near_t,
                                   "exact_image_matches": exact_i, "near_image_candidates": near_i}
        print(split, f"문항 {nq}, 이미지 {ni}, 문장 일치 {len(exact_t)}, 문장 유사 {len(near_t)}, "
                     f"이미지 일치 {len(exact_i)}, dHash 후보 {len(near_i)}")
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("저장:", a.out)


if __name__ == "__main__":
    main()
