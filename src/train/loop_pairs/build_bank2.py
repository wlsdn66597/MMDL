"""두 번째 문제 은행 (수학 외 분야, 자체 데이터): 샘플링용 객관식 문항으로 정리한다.

- 입력: 이전 실험에서 만든 D/R 데이터 (retention_dr, targeted_augmentation, preservation_balanced R 풀).
- 객관식만: input_prompt에서 질문과 "A. …" 보기를 뽑는다. 정답 문자는 answer 필드나 target의 마지막
  "Final answer: X" 줄에서 얻고, 보기 안에 있는지 확인한다. 주관식은 별도 파일로만 남긴다.
- 입력 문장은 베이스라인 객관식 템플릿으로 다시 만든다 (원래 데이터에는 "Please select …" 줄이 빠져 있음).
- 이미지가 있는 문항만 쓴다 (MMMU는 모두 이미지 문항). 이미지 경로는 절대 경로로 바꾼다.
- 개발 평가 문항 제외: dev111 (retention_dr dev_questions), dev195, 보존64. 같은 이미지(SHA256)나 같은
  출처 그룹을 쓰는 문항도 뺀다.
- 중복 정리: 같은 id, 또는 같은 이미지+같은 질문이면 하나만 남긴다 (먼저 읽은 출처 우선).
- MMMU 이미지 중복: mmfr_sft.py와 같은 2단계 검사 (64비트 dHash <= 4 → 256비트 dHash <= 32, 32x32 평균 차이 <= 12).

사용 (서버, 평가 환경 python)
  python build_bank2.py --out ~/mmdl/datasets/bank2_own_mc
"""
import argparse
import ast
import glob
import hashlib
import io
import json
import os
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arxivqa_prep import fine_match, fine_sig, popcount64  # noqa: E402
from arxivqa_sft import student_prompt  # noqa: E402
from mmmu_overlap import DHASH_MAX, REV, dhash, pix_hash  # noqa: E402

DS = Path(os.path.expanduser("~/mmdl/datasets"))
TR = Path(os.path.expanduser("~/mmdl/train_runs"))
SOURCES = [  # (파일, 이미지 기준 폴더, 출처 이름)
    (DS / "retention_dr_20261025/train_general.jsonl", DS / "retention_dr_20261025", "retention_dr_train218"),
    (DS / "targeted_augmentation_20261101/reviewed_bank400.jsonl", DS / "targeted_augmentation_20261101", "targeted400"),
    (DS / "preservation_balanced_20261030/R_pool477_partitioned.jsonl", DS / "preservation_balanced_20261030", "R_pool477"),
]
DEV_FILES = [
    (DS / "retention_dr_20261025/dev_questions.jsonl", DS / "retention_dr_20261025"),
    (TR / "preservation_baseline_eval_20261028/dev195_baseline/graded.jsonl", None),
    (DS / "preservation_balanced_20261030/retention_holdout64/R_candidates_questions.jsonl",
     DS / "preservation_balanced_20261030/retention_holdout64"),
    (DS / "preservation_balanced_20261030/retention_holdout64/source_and_review.jsonl", None),
]
OPT = re.compile(r"^\s*([A-J])[\.\)]\s*(.+?)\s*$")
FINAL = re.compile(r"Final answer\s*:\s*\(?([A-J])\)?\s*\.?\s*$", re.I | re.M)


def as_list(x):
    if isinstance(x, list):
        return x
    if not x:
        return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else [v]
    except Exception:
        return [x]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def parse_prompt(prompt):
    """input_prompt → (질문, 보기 목록). 보기가 없으면 (질문, [])."""
    lines = prompt.split("\n")
    if "Options:" not in [l.strip() for l in lines]:
        q = re.sub(r"\n?On the last line, write.*$", "", prompt, flags=re.S).strip()
        return re.sub(r"^Question:\s*", "", q).strip(), []
    i = [l.strip() for l in lines].index("Options:")
    stem = "\n".join(lines[:i]).strip()
    stem = re.sub(r"^Question:\s*", "", stem, flags=re.M).strip()
    opts = []
    for l in lines[i + 1:]:
        m = OPT.match(l)
        if not m:
            break
        opts.append((m.group(1), m.group(2)))
    letters = [o[0] for o in opts]
    if letters != list(string.ascii_uppercase[:len(opts)]):
        return stem, []
    return stem, [o[1] for o in opts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmmu-splits", nargs="+", default=["validation", "dev", "test"])
    a = ap.parse_args()
    import numpy as np
    import pandas as pd
    from PIL import Image

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 개발 평가 문항: id, 출처 그룹, 이미지 해시
    dev_ids, dev_groups, dev_img = set(), set(), set()
    for f, root in DEV_FILES:
        if not f.exists():
            print("경고: 개발 세트 파일 없음", f)
            continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            dev_ids.add(r["id"])
            for g in ("split_group_id", "source_group"):
                if r.get(g):
                    dev_groups.add(r[g])
            if root:
                for p in as_list(r.get("images")):
                    if (root / p).exists():
                        dev_img.add(sha_file(root / p))
    print(f"개발 평가 문항 {len(dev_ids)}, 그룹 {len(dev_groups)}, 이미지 {len(dev_img)}")

    # 2. 후보 읽기
    kept, opens, drop = [], [], Counter()
    seen_id, seen_key = set(), set()
    for f, root, src in SOURCES:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            rid = r["id"]
            if rid in seen_id:
                drop["중복 id"] += 1
                continue
            seen_id.add(rid)
            imgs = [str((root / p).resolve()) for p in as_list(r.get("images"))]
            if not imgs or not all(os.path.exists(p) for p in imgs):
                drop["이미지 없음(텍스트 전용 등)"] += 1
                continue
            if rid in dev_ids or any(r.get(g) in dev_groups for g in ("split_group_id", "source_group") if r.get(g)):
                drop["개발 평가 문항/그룹"] += 1
                continue
            hashes = [sha_file(p) for p in imgs]
            if any(h in dev_img for h in hashes):
                drop["개발 평가와 같은 이미지"] += 1
                continue
            stem, opts = parse_prompt(r.get("input_prompt", ""))
            meta = {"id": rid, "bank_source": src, "source_slug": r.get("source_slug"), "domain": r.get("domain"),
                    "role": r.get("role"), "task": r.get("task") or r.get("skill"),
                    "group": r.get("split_group_id") or r.get("source_group"), "images": imgs,
                    "images_sha256": hashes}
            key = (tuple(hashes), stem.lower())
            if key in seen_key:
                drop["같은 이미지+질문 중복"] += 1
                continue
            seen_key.add(key)
            if not opts:
                opens.append(dict(meta, question=stem, target=r.get("target")))
                drop["주관식(별도 보관)"] += 1
                continue
            ans = str(r.get("answer") or "").strip()
            if not re.fullmatch(r"[A-J]", ans):
                m = FINAL.findall(str(r.get("target") or ""))
                ans = m[-1].upper() if m else ""
            if len(ans) != 1 or ans not in string.ascii_uppercase[:len(opts)]:
                drop["정답 문자를 찾지 못함"] += 1
                continue
            kept.append(dict(meta, question_stem=stem, options=opts, answer_letter=ans,
                             input_prompt=student_prompt(stem, opts)))
    print("객관식", len(kept), "버림", dict(drop), flush=True)

    # 3. MMMU 이미지 중복
    snap = os.path.expanduser(f"~/.cache/huggingface/hub/datasets--MMMU--MMMU/snapshots/{REV}")
    mimg, mfine = [], {}
    for split in a.mmmu_splits:
        for f in sorted(glob.glob(f"{snap}/*/{split}-*.parquet")):
            for _, m in pd.read_parquet(f).iterrows():
                for k in range(1, 8):
                    cell = m.get(f"image_{k}")
                    if isinstance(cell, dict) and cell.get("bytes"):
                        mi = Image.open(io.BytesIO(cell["bytes"]))
                        name = f"{m['id']}_image_{k}"
                        mimg.append((name, dhash(mi), pix_hash(mi)))
                        mfine[name] = fine_sig(mi)
    paths = sorted({p for c in kept for p in c["images"]})
    pdh = np.array([dhash(Image.open(p)) for p in paths], dtype=np.uint64)
    pph = defaultdict(list)
    for i, p in enumerate(paths):
        pph[pix_hash(Image.open(p))].append(p)
    bad = defaultdict(list)
    for name, dh, ph in mimg:
        for p in pph.get(ph, []):
            bad[p].append({"mmmu_image": name, "type": "pixel_sha256"})
        d = popcount64(pdh ^ np.uint64(dh))
        for i in np.nonzero(d <= DHASH_MAX)[0]:
            ham, mad = fine_match(fine_sig(Image.open(paths[i])), mfine[name])
            if ham <= 32 and mad <= 12:
                bad[paths[i]].append({"mmmu_image": name, "type": "dhash2", "dhash256": ham, "gray_mad": round(mad, 2)})
    flagged = [c for c in kept if any(p in bad for p in c["images"])]
    kept = [c for c in kept if not any(p in bad for p in c["images"])]
    print("MMMU 이미지 중복 후보로 제외", len(flagged), flush=True)

    def dump(name, rows):
        with open(out / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dump("bank2_mc.jsonl", kept)
    dump("bank2_open_reserved.jsonl", opens)
    dump("mmmu_flagged.jsonl", [dict(c, mmmu_hits=[h for p in c["images"] for h in bad.get(p, [])]) for c in flagged])
    rep = {"sources": [str(s[0]) for s in SOURCES], "dev_excluded_files": [str(d[0]) for d in DEV_FILES],
           "dev_ids": len(dev_ids), "dropped": dict(drop), "mmmu_flagged": len(flagged), "mc_kept": len(kept),
           "open_reserved": len(opens),
           "by_source_slug": Counter(c["source_slug"] for c in kept), "by_domain": Counter(str(c["domain"]) for c in kept),
           "by_role": Counter(str(c["role"]) for c in kept), "answer_letters": Counter(c["answer_letter"] for c in kept),
           "n_options": Counter(len(c["options"]) for c in kept),
           "open_by_source_slug": Counter(c["source_slug"] for c in opens)}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if kept:
        print("\n=== 예시", kept[0]["id"], kept[0]["source_slug"], "정답", kept[0]["answer_letter"], "===\n" + kept[0]["input_prompt"])


if __name__ == "__main__":
    main()
