"""수학 외 분야 문제 은행 합치기 + MMMU 중복 검사.

입력: bank2_own_mc/bank2_mc.jsonl (270), bank3_extra/{music,medx,wikiart,arxivqa_eng}_mc.jsonl
검사 (MMMU validation·dev·test):
  - 이미지: 픽셀 SHA256 일치, 64비트 dHash <= 4 → 256비트 dHash <= 32 그리고 32x32 흑백 평균 차이 <= 12
  - 문장: 정규화한 질문 일치, 단어 Jaccard >= 0.8 (질문이 5단어 이상일 때만. 틀에 박힌 짧은 질문은 제외)
걸린 문항은 빼고, 이미지 쌍은 mmmu_candidates/에 나란히 저장한다.
출력: bank_nonmath/bank_all.jsonl (sample_bank.py 입력 형식, 이미지는 절대 경로)

사용 (서버): python merge_bank_nonmath.py --out ~/mmdl/datasets/bank_nonmath
"""
import argparse
import glob
import io
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arxivqa_prep import fine_match, fine_sig, popcount64  # noqa: E402
from mmmu_overlap import DHASH_MAX, JACCARD_MIN, REV, dhash, norm, pix_hash  # noqa: E402

DS = Path(os.path.expanduser("~/mmdl/datasets"))
PARTS = [DS / "bank2_own_mc/bank2_mc.jsonl", DS / "bank3_extra/music_mc.jsonl", DS / "bank3_extra/medx_mc.jsonl",
         DS / "bank3_extra/wikiart_mc.jsonl", DS / "bank3_extra/arxivqa_eng_mc.jsonl"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmmu-splits", nargs="+", default=["validation", "dev", "test"])
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    from PIL import Image
    out = Path(os.path.expanduser(a.out))
    (out / "mmmu_candidates").mkdir(parents=True, exist_ok=True)

    rows, seen = [], set()
    for p in PARTS:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            r["part"] = p.stem
            rows.append(r)
    print("합친 문항", len(rows), Counter(r["part"] for r in rows), flush=True)

    snap = os.path.expanduser(f"~/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/{REV}")
    mq, mimg, mfine, mthumb = {}, [], {}, {}
    for split in a.mmmu_splits:
        for f in sorted(glob.glob(f"{snap}/*/{split}-*.parquet")):
            for _, m in pd.read_parquet(f).iterrows():
                mq[m["id"]] = norm(m["question"])
                for k in range(1, 8):
                    cell = m.get(f"image_{k}")
                    if isinstance(cell, dict) and cell.get("bytes"):
                        mi = Image.open(io.BytesIO(cell["bytes"]))
                        name = f"{m['id']}_image_{k}"
                        mimg.append((name, dhash(mi), pix_hash(mi)))
                        mfine[name] = fine_sig(mi)
                        mthumb[name] = mi.convert("RGB").resize((max(1, mi.width * 200 // max(1, mi.height)), 200))
    print("MMMU 문항", len(mq), "이미지", len(mimg), flush=True)

    paths = sorted({p for r in rows for p in r["images"]})
    pdh = np.array([dhash(Image.open(p)) for p in paths], dtype=np.uint64)
    pph = defaultdict(list)
    for p in paths:
        pph[pix_hash(Image.open(p))].append(p)
    bad_img = defaultdict(list)
    stage1 = 0
    for name, dh, ph in mimg:
        for p in pph.get(ph, []):
            bad_img[p].append({"mmmu_image": name, "type": "pixel_sha256"})
        d = popcount64(pdh ^ np.uint64(dh))
        for i in np.nonzero(d <= DHASH_MAX)[0]:
            stage1 += 1
            ham, mad = fine_match(fine_sig(Image.open(paths[i])), mfine[name])
            if ham <= 32 and mad <= 12:
                bad_img[paths[i]].append({"mmmu_image": name, "type": "dhash2", "dhash256": ham, "gray_mad": round(mad, 2)})
                A = Image.open(paths[i]).convert("RGB")
                A = A.resize((max(1, A.width * 200 // max(1, A.height)), 200))
                B = mthumb[name]
                sheet = Image.new("RGB", (A.width + B.width + 10, 200), "white")
                sheet.paste(A, (0, 0))
                sheet.paste(B, (A.width + 10, 0))
                sheet.save(out / "mmmu_candidates" / f"{Path(paths[i]).stem[:40]}__{name}.png")
    print(f"이미지: 1차 후보 {stage1}쌍 → 2차 통과 {sum(len(v) for v in bad_img.values())}쌍 ({len(bad_img)}장)", flush=True)

    def qtext(r):
        s = r.get("question_stem")
        if not s:
            s = r["input_prompt"].split("\nOptions:")[0].replace("Question:", "")
        return norm(s)
    mset = defaultdict(list)
    for mid, q in mq.items():
        mset[q].append(mid)
    mtoks = [(mid, set(q.split())) for mid, q in mq.items() if len(q.split()) >= 5]
    bad_txt = defaultdict(list)
    for r in rows:
        q = qtext(r)
        toks = set(q.split())
        if len(toks) < 5:
            continue
        for mid in mset.get(q, []):
            bad_txt[r["id"]].append({"mmmu_id": mid, "type": "exact_text"})
        for mid, mt in mtoks:
            inter = len(toks & mt)
            if inter >= JACCARD_MIN * max(len(toks), len(mt)) * 0.9 and inter / len(toks | mt) >= JACCARD_MIN:
                if not any(h["mmmu_id"] == mid for h in bad_txt[r["id"]]):
                    bad_txt[r["id"]].append({"mmmu_id": mid, "type": "jaccard", "jaccard": round(inter / len(toks | mt), 3)})
    bad_txt = {k: v for k, v in bad_txt.items() if v}
    print("문장 후보", len(bad_txt), flush=True)

    kept, flagged = [], []
    for r in rows:
        hits = [h for p in r["images"] for h in bad_img.get(p, [])] + bad_txt.get(r["id"], [])
        if hits:
            flagged.append({"id": r["id"], "part": r["part"], "hits": hits})
        else:
            kept.append({"id": r["id"], "source": r["part"], "source_slug": r.get("source_slug"), "domain": r.get("domain"),
                         "task": r.get("task"), "images": r["images"], "input_prompt": r["input_prompt"],
                         "options": r["options"], "answer_letter": r["answer_letter"]})
    with open(out / "bank_all.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "mmmu_flagged.jsonl", "w", encoding="utf-8") as f:
        for r in flagged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"input": len(rows), "kept": len(kept), "flagged": len(flagged),
           "flagged_by_part": Counter(r["part"] for r in flagged), "kept_by_part": Counter(r["source"] for r in kept),
           "kept_by_n_options": Counter(len(r["options"]) for r in kept),
           "answer_letters": Counter(r["answer_letter"] for r in kept), "image_stage1_pairs": stage1}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
