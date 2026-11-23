"""ArxivQA 정리 + 논문 단위 8:2 분리 + MMMU 중복 검사.

1. 정리: 보기 앞머리("A) ", "A. ")를 떼고, 정답(label)을 보기 문자로 통일한다. 맞출 수 없으면 버린다.
2. 분리: 이미지 파일명의 arXiv ID(예: 0805.4509_1.jpg → 0805.4509)로 논문을 묶고,
   논문 목록을 고정 시드로 섞어 20%를 테스트로 둔다. 같은 논문의 그림은 한쪽에만 들어간다.
3. MMMU 중복: mmmu_overlap.py와 같은 기준(정규화 문장 일치, 단어 Jaccard >= 0.8, 픽셀 SHA256 일치,
   dHash 거리 <= 4)을 numpy로 한 번에 계산한다. 걸린 문항은 학습·테스트 모두에서 뺀다.

사용 (서버, 평가 환경 python)
  python arxivqa_prep.py --jsonl <snapshot>/arxivqa.jsonl --images ~/mmdl/datasets/arxivqa \
      --out ~/mmdl/datasets/arxivqa_prepared --mmmu-splits validation dev test
"""
import argparse
import ast
import glob
import hashlib
import io
import json
import os
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mmmu_overlap import DHASH_MAX, JACCARD_MIN, REV, dhash, norm, pix_hash  # noqa: E402

SEED = 3407
TEST_FRAC = 0.2
PREFIX = re.compile(r"^\s*\(?([A-Z])\s*[\).:]\s*")


TRAIL_OK = re.compile(r"^\s*(-*|#.*|(For )?Figure.*|Question.*)\s*$", re.I)
JUNK = re.compile(r"plausible answer choice|correct answer choice|correct choice|rationale|please note|content policy", re.I)


def clean(r):
    opts = r["options"]
    if isinstance(opts, str):
        opts = ast.literal_eval(opts)
    opts = [str(o) for o in opts]
    m_all = [PREFIX.match(o) for o in opts]
    n_pref = 0
    while n_pref < len(m_all) and m_all[n_pref]:
        n_pref += 1
    head = [m.group(1) for m in m_all[:n_pref]]
    if n_pref >= 2 and head == list(string.ascii_uppercase[:n_pref]) and not any(m_all[n_pref:]):
        # A부터 순서대로 앞머리가 있는 보기 뒤에 붙은 다음 문항 제목·구분선은 떼어 낸다. 그 밖의 잔여물이면 버린다.
        if not all(TRAIL_OK.match(o) for o in opts[n_pref:]):
            return None, "생성 잔여물이 보기에 섞임"
        opts, m_all = opts[:n_pref], m_all[:n_pref]
    letters = [m.group(1) if m else None for m in m_all]
    pref = [l for l in letters if l]
    if pref and not all(letters) and len(set(pref)) < len(pref):
        letters = [None] * len(letters)                # "B. pumilus"처럼 앞머리 문자가 겹치면 원문 보기로 본다
    if all(letters) and letters == list(string.ascii_uppercase[:len(letters)]):
        texts = [o[m.end():].strip() for o, m in zip(opts, m_all)]
    elif not any(letters) or (all(letters) and len(set(letters)) < len(letters)):
        texts = [o.strip() for o in opts]          # 앞머리 없음 (또는 "B. pumilus" 같은 학명 표기)
        letters = list(string.ascii_uppercase[:len(texts)])
    else:
        return None, "보기 앞머리 혼합"
    if len(texts) < 2 or any(not t for t in texts):
        return None, "보기 수 부족·빈 보기"
    if any(JUNK.search(t) for t in texts) or JUNK.search(str(r["label"])):
        return None, "생성 잔여물이 보기에 섞임"
    if len(set(t.lower() for t in texts)) < len(texts):
        return None, "보기 중복"
    lab = str(r["label"]).strip()
    ans = None
    m = re.match(r"^\(?([A-Z])\s*[\).:]\s*(.*)$", lab)
    if m and m.group(1) in letters:
        rest = m.group(2).strip()
        if not rest or rest.lower() == texts[letters.index(m.group(1))].lower():
            ans = m.group(1)
    if ans is None and re.fullmatch(r"[A-Z]", lab) and lab in letters:
        ans = lab
    if ans is None:
        hit = [l for l, t in zip(letters, texts) if t.lower() == lab.lower()]
        if len(hit) == 1:
            ans = hit[0]
    if ans is None:
        return None, "정답을 보기와 맞출 수 없음"
    if not str(r.get("rationale", "")).strip():
        return None, "풀이 없음"
    paper = Path(r["image"]).stem.rsplit("_", 1)[0]
    return {"id": r["id"], "paper_id": paper, "image": r["image"], "question": r["question"].strip(),
            "options": texts, "answer_letter": ans, "rationale": r["rationale"].strip()}, None


def fine_sig(im):
    """2차 검사용: 256비트 dHash와 32x32 흑백 축소본."""
    import numpy as np
    from PIL import Image
    g = im.convert("L")
    d = np.asarray(g.resize((17, 16), Image.LANCZOS), dtype=np.int16)
    bits = (d[:, :-1] > d[:, 1:]).flatten()
    small = np.asarray(g.resize((32, 32), Image.LANCZOS), dtype=np.float32)
    return bits, small


def fine_match(sa, sb):
    import numpy as np
    return int((sa[0] != sb[0]).sum()), float(np.abs(sa[1] - sb[1]).mean())


def popcount64(x):
    import numpy as np
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return table[x.view(np.uint8).reshape(-1, 8)].sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmmu-splits", nargs="+", default=["validation", "dev", "test"])
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    from PIL import Image

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = [json.loads(l) for l in open(a.jsonl, encoding="utf-8") if l.strip()]
    kept, dropped = [], []
    seen = set()
    for r in raw:
        c, why = clean(r)
        if c is None:
            dropped.append({"id": r["id"], "reason": why})
            continue
        if not (Path(a.images) / c["image"]).exists():
            dropped.append({"id": r["id"], "reason": "이미지 없음"})
            continue
        key = (c["image"], norm(c["question"]))
        if key in seen:
            dropped.append({"id": r["id"], "reason": "같은 이미지·질문 중복"})
            continue
        seen.add(key)
        kept.append(c)
    print(f"정리: 원본 {len(raw)}, 유지 {len(kept)}, 버림 {len(dropped)}", Counter(d["reason"] for d in dropped))

    # MMMU 중복
    snap = os.path.expanduser(f"~/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/{REV}")
    mq, mimgs, mfine, mthumb = {}, [], {}, {}
    for split in a.mmmu_splits:
        for f in sorted(glob.glob(f"{snap}/*/{split}-*.parquet")):
            for _, m in pd.read_parquet(f).iterrows():
                mq[m["id"]] = norm(m["question"])
                for k in range(1, 8):
                    cell = m.get(f"image_{k}")
                    if isinstance(cell, dict) and cell.get("bytes"):
                        im = Image.open(io.BytesIO(cell["bytes"]))
                        name = f"{m['id']}_image_{k}"
                        mimgs.append((name, dhash(im), pix_hash(im)))
                        mfine[name] = fine_sig(im)
                        mthumb[name] = im.convert("RGB").resize((max(1, im.width * 200 // max(1, im.height)), 200))
    print(f"MMMU {a.mmmu_splits}: 문항 {len(mq)}, 이미지 {len(mimgs)}")
    uniq = sorted({c["image"] for c in kept})
    our_dh = np.zeros(len(uniq), dtype=np.uint64)
    our_ph = {}
    for i, p in enumerate(uniq):
        im = Image.open(Path(a.images) / p)
        our_dh[i] = np.uint64(dhash(im))
        our_ph.setdefault(pix_hash(im), []).append(p)
    img_hits = defaultdict(list)
    min_dist = np.full(len(uniq), 64, dtype=np.int64)
    for name, dh, ph in mimgs:
        for p in our_ph.get(ph, []):
            img_hits[p].append({"mmmu_image": name, "type": "pixel_sha256"})
        d = popcount64(our_dh ^ np.uint64(dh)).astype(np.int64)
        np.minimum(min_dist, d, out=min_dist)
        for i in np.nonzero(d <= DHASH_MAX)[0]:
            img_hits[uniq[i]].append({"mmmu_image": name, "type": "dhash", "distance": int(d[i])})
    # 문장: 정확 일치 + 역색인으로 Jaccard 후보만 계산
    by_norm = defaultdict(list)
    df = Counter()
    toks = {}
    for c in kept:
        q = norm(c["question"])
        by_norm[q].append(c["id"])
        toks[c["id"]] = set(q.split())
        df.update(toks[c["id"]])
    post = defaultdict(list)
    for cid, ts in toks.items():
        for t in ts:
            if df[t] <= 2000:
                post[t].append(cid)
    text_hits = defaultdict(list)
    for mid, q in mq.items():
        for cid in by_norm.get(q, []):
            text_hits[cid].append({"mmmu_id": mid, "type": "exact"})
        mt = set(q.split())
        if not mt:
            continue
        cnt = Counter(cid for t in mt if t in post for cid in post[t])
        for cid, n in cnt.items():
            if n < JACCARD_MIN * len(mt) * 0.5:
                continue
            j = len(mt & toks[cid]) / len(mt | toks[cid])
            if j >= JACCARD_MIN and not any(h["mmmu_id"] == mid for h in text_hits[cid]):
                text_hits[cid].append({"mmmu_id": mid, "type": "jaccard", "jaccard": round(j, 3)})
    # 2차 검사: 64비트 dHash 후보를 256비트 dHash(<= 32)와 32x32 흑백 평균 차이(<= 12)로 다시 거른다
    review = out / "mmmu_candidates"
    review.mkdir(exist_ok=True)
    stage1 = sum(len(v) for v in img_hits.values())
    for p in list(img_hits):
        sig = fine_sig(Image.open(Path(a.images) / p))
        keep_hits = []
        for h in img_hits[p]:
            if h["type"] == "pixel_sha256":
                keep_hits.append(h)
                continue
            ham, mad = fine_match(sig, mfine[h["mmmu_image"]])
            h.update(dhash256=ham, gray_mad=round(mad, 2))
            if ham <= 32 and mad <= 12:
                keep_hits.append(h)
                A = Image.open(Path(a.images) / p).convert("RGB")
                A = A.resize((max(1, A.width * 200 // max(1, A.height)), 200))
                B = mthumb[h["mmmu_image"]]
                sheet = Image.new("RGB", (A.width + B.width + 10, 200), "white")
                sheet.paste(A, (0, 0))
                sheet.paste(B, (A.width + 10, 0))
                sheet.save(review / f"{Path(p).stem}__{h['mmmu_image']}.png")
        if keep_hits:
            img_hits[p] = keep_hits
        else:
            del img_hits[p]
    print(f"이미지 후보: 1차(64비트 dHash) {stage1}쌍 → 2차 {sum(len(v) for v in img_hits.values())}쌍", flush=True)
    flagged = {c["id"] for c in kept if c["image"] in img_hits or c["id"] in text_hits}
    for c in kept:
        if c["id"] in flagged:
            dropped.append({"id": c["id"], "reason": "MMMU 중복 후보",
                            "image_hits": img_hits.get(c["image"], []), "text_hits": text_hits.get(c["id"], [])})
    kept = [c for c in kept if c["id"] not in flagged]
    print(f"MMMU 중복 후보: 이미지 {len(img_hits)}장, 문장 {len(text_hits)}문항 → 문항 {len(flagged)}개 제외")

    # 논문 단위 분리
    papers = sorted({c["paper_id"] for c in kept})
    rng = random.Random(SEED)
    rng.shuffle(papers)
    test_papers = set(papers[:round(len(papers) * TEST_FRAC)])
    for c in kept:
        c["split"] = "test" if c["paper_id"] in test_papers else "train"
    for name in ("train", "test"):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for c in kept:
                if c["split"] == name:
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out / "dropped.jsonl", "w", encoding="utf-8") as f:
        for d in dropped:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    n_tr = sum(c["split"] == "train" for c in kept)
    rep = {"source": a.jsonl, "seed": SEED, "test_frac_papers": TEST_FRAC, "raw": len(raw), "kept": len(kept),
           "train": n_tr, "test": len(kept) - n_tr, "papers": len(papers), "test_papers": len(test_papers),
           "train_test_paper_overlap": len({c["paper_id"] for c in kept if c["split"] == "train"} & test_papers),
           "dropped_reasons": Counter(d["reason"] for d in dropped),
           "answer_letter_dist": {s: Counter(c["answer_letter"] for c in kept if c["split"] == s) for s in ("train", "test")},
           "n_options_dist": Counter(len(c["options"]) for c in kept),
           "mmmu": {"revision": REV, "splits": a.mmmu_splits, "questions": len(mq), "images": len(mimgs),
                    "methods": ["normalized exact question", f"word Jaccard >= {JACCARD_MIN}",
                                "RGB size+pixel SHA256", f"64-bit dHash distance <= {DHASH_MAX}",
                                "2nd stage: 256-bit dHash <= 32 and 32x32 gray MAD <= 12"],
                    "image_candidates": len(img_hits), "text_candidates": len(text_hits),
                    "excluded_questions": len(flagged),
                    "min_dhash_distance_hist": Counter(int(x) for x in min_dist)}}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("answer_letter_dist",)}, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
