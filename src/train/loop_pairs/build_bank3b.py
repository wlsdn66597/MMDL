"""세 번째 문제 은행 재료 (받은 데이터): MedXpertQA 이미지 문항 + WikiArt 양식·장르 객관식.

medx: MedXpertQA MM test 2,000문항 중 이미지가 1장인 문항에서 N개를 medical_task 비율대로 뽑는다.
  - 질문에서 "Answer Choices: …" 부분을 떼고 보기는 options 필드를 쓴다 (5지선다 그대로).
  - 고른 문항의 이미지만 images.zip에서 꺼낸다.
wikiart: 받은 parquet 일부에서 양식(style) 문항과 장르(genre) 문항을 만든다.
  - 정답은 데이터의 라벨. 오답 보기는 다른 라벨에서 3개. 서로 헷갈리는 양식 묶음(CONFUSABLE)은 같은 문항에 넣지 않는다.
  - 양식별로 고르게 뽑는다. 한 이미지는 한 문항에만 쓴다.

MMMU 중복 검사는 여기서 하지 않는다 (합칠 때 한 번에 한다).

사용 (서버): python build_bank3b.py --out ~/mmdl/datasets/bank3_extra
"""
import argparse
import glob
import hashlib
import io
import json
import os
import random
import re
import string
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arxivqa_sft import student_prompt  # noqa: E402

DS = Path(os.path.expanduser("~/mmdl/datasets"))
SEED = 3407
CONFUSABLE = [
    {"Cubism", "Analytical_Cubism", "Synthetic_Cubism"},
    {"Realism", "Contemporary_Realism", "New_Realism"},
    {"Early_Renaissance", "High_Renaissance", "Northern_Renaissance", "Mannerism_Late_Renaissance"},
    {"Impressionism", "Post_Impressionism", "Pointillism"},
    {"Abstract_Expressionism", "Action_painting", "Color_Field_Painting"},
    {"Baroque", "Rococo"},
    {"Expressionism", "Fauvism"},
    {"Art_Nouveau", "Symbolism"},
    {"Naive_Art_Primitivism", "Pop_Art"},
]
GENRE_SKIP = {"Unknown Genre", "nude_painting", "sketch_and_study"}


def pretty(name):
    return name.replace("_", " ").replace("Mannerism Late Renaissance", "Mannerism (Late Renaissance)") \
        .replace("Naive Art Primitivism", "Naive Art (Primitivism)").replace("Ukiyo e", "Ukiyo-e")


def build_medx(out, n):
    root = DS / "medxpertqa_raw"
    rows = [json.loads(l) for l in open(root / "MM/test.jsonl", encoding="utf-8")]
    single = [r for r in rows if len(r["images"]) == 1]
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in single:
        by[r["medical_task"]].append(r)
    picked = []
    for t, v in sorted(by.items()):
        rng.shuffle(v)
        picked += v[:round(n * len(v) / len(single))]
    z = zipfile.ZipFile(root / "images.zip")
    members = {Path(m).name: m for m in z.namelist() if not m.endswith("/")}
    img_dir = out / "images_medx"
    img_dir.mkdir(parents=True, exist_ok=True)
    kept, drop = [], Counter()
    for r in picked:
        name = r["images"][0]
        if name not in members:
            drop["이미지 없음"] += 1
            continue
        stem = r["question"].split("\nAnswer Choices:")[0].strip()
        letters = sorted(r["options"])
        if letters != list(string.ascii_uppercase[:len(letters)]) or r["label"] not in letters:
            drop["보기·정답 형식 이상"] += 1
            continue
        opts = [r["options"][L].strip() for L in letters]
        dst = img_dir / name
        if not dst.exists():
            dst.write_bytes(z.read(members[name]))
        kept.append({"id": "medx_" + r["id"], "bank_source": "medxpertqa_mm_single_image", "source_slug": "MedXpertQA",
                     "domain": "medicine_clinical", "role": None, "task": r["medical_task"], "group": r["id"],
                     "body_system": r["body_system"], "question_type": r["question_type"],
                     "images": [str(dst)], "question_stem": stem, "options": opts, "answer_letter": r["label"],
                     "input_prompt": student_prompt(stem, opts)})
    with open(out / "medx_mc.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"total": len(rows), "single_image": len(single), "kept": len(kept), "dropped": dict(drop),
           "by_task": Counter(r["task"] for r in kept), "by_body_system": Counter(r["body_system"] for r in kept),
           "answer_letters": Counter(r["answer_letter"] for r in kept)}
    print("medx", json.dumps(rep, ensure_ascii=False))
    return rep


def build_wikiart(out, n_style, n_genre):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(DS / "wikiart_raw/*.parquet")))
    # 라벨 이름은 parquet에 없어서 datasets-server의 info에서 받아 둔 label_names.json을 쓴다
    names = json.loads((DS / "wikiart_raw/label_names.json").read_text(encoding="utf-8"))
    style_names, genre_names = names["style"], names["genre"]
    group_of = {s: i for i, g in enumerate(CONFUSABLE) for s in g}
    rng = random.Random(SEED)
    cand = []
    for f in files:
        t = pq.read_table(f, columns=["style", "genre", "artist"]).to_pydict()
        for i in range(len(t["style"])):
            cand.append((f, i, style_names[t["style"][i]], genre_names[t["genre"][i]]))
    rng.shuffle(cand)
    by_style = defaultdict(list)
    for c in cand:
        by_style[c[2]].append(c)
    per = max(1, n_style // max(1, len(by_style)))
    style_pick, used = [], set()
    for s in sorted(by_style):
        for c in by_style[s][:per]:
            style_pick.append(c)
            used.add((c[0], c[1]))
    by_genre = defaultdict(list)
    for c in cand:
        if (c[0], c[1]) not in used and c[3] not in GENRE_SKIP:
            by_genre[c[3]].append(c)
    perg = max(1, n_genre // max(1, len(by_genre)))
    genre_pick = [c for g in sorted(by_genre) for c in by_genre[g][:perg]]

    img_dir = out / "images_wikiart"
    img_dir.mkdir(parents=True, exist_ok=True)
    need = defaultdict(dict)
    for kind, picks in (("style", style_pick), ("genre", genre_pick)):
        for c in picks:
            need[c[0]][c[1]] = (kind, c)
    kept = []
    for f, idx in need.items():
        tb = pq.read_table(f, columns=["image"])
        col = tb.column("image")
        for i, (kind, c) in sorted(idx.items()):
            cell = col[i].as_py()
            raw = cell["bytes"]
            sha = hashlib.sha256(raw).hexdigest()
            ext = ".png" if raw[:4] == b"\x89PNG" else ".jpg"
            dst = img_dir / (sha[:24] + ext)
            if not dst.exists():
                dst.write_bytes(raw)
            r2 = random.Random(int(sha[:12], 16))
            if kind == "style":
                ans = c[2]
                pool = [s for s in style_names if s != ans and group_of.get(s, -1 - style_names.index(s)) != group_of.get(ans, -99)]
                ds, seen_groups = [], set()
                for s in r2.sample(pool, len(pool)):
                    g = group_of.get(s)
                    if g is not None and g in seen_groups:
                        continue
                    if g is not None:
                        seen_groups.add(g)
                    ds.append(s)
                    if len(ds) == 3:
                        break
                q = "Which art movement or style does this painting best represent?"
            else:
                ans = c[3]
                pool = [g for g in genre_names if g != ans and g not in GENRE_SKIP]
                ds = r2.sample(pool, 3)
                q = "Which genre best describes this painting?"
            opts = [pretty(x) for x in ds + [ans]]
            order = list(range(4))
            r2.shuffle(order)
            opts = [opts[k] for k in order]
            letter = string.ascii_uppercase[order.index(3)]
            kept.append({"id": f"wa_{kind}_{sha[:16]}", "bank_source": f"wikiart_{kind}_label_mc", "source_slug": "WikiArt",
                         "domain": "arts_design", "role": None, "task": f"art_{kind}_identification",
                         "group": sha[:16], "images": [str(dst)], "images_sha256": [sha], "question_stem": q,
                         "options": opts, "answer_letter": letter, "label_raw": ans,
                         "answer_origin": "WikiArt label (not expert-verified)", "input_prompt": student_prompt(q, opts)})
    with open(out / "wikiart_mc.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"parquet_files": [Path(f).name for f in files], "rows_available": len(cand), "styles_available": len(by_style),
           "kept": len(kept), "by_task": Counter(r["task"] for r in kept),
           "by_style": Counter(r["label_raw"] for r in kept if "style" in r["task"]),
           "by_genre": Counter(r["label_raw"] for r in kept if "genre" in r["task"]),
           "answer_letters": Counter(r["answer_letter"] for r in kept)}
    print("wikiart", json.dumps(rep, ensure_ascii=False))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-medx", type=int, default=500)
    ap.add_argument("--n-style", type=int, default=300)
    ap.add_argument("--n-genre", type=int, default=100)
    ap.add_argument("--only", choices=["medx", "wikiart"])
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    if a.only in (None, "medx"):
        rep["medx"] = build_medx(out, a.n_medx)
    if a.only in (None, "wikiart"):
        rep["wikiart"] = build_wikiart(out, a.n_style, a.n_genre)
    p = out / "report_medx_wikiart.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    old.update(rep)
    p.write_text(json.dumps(old, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
