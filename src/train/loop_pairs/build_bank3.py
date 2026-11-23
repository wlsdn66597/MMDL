"""세 번째 문제 은행 재료 (받을 것 없는 부분): 음악 객관식 변환 + ArxivQA 분야별 선별.

music: bank2_open_reserved.jsonl의 MusiXQA 주관식을 4지선다로 바꾼다.
  - 정답은 검토를 통과한 원래 값(target의 Final answer) 그대로 쓴다.
  - 오답 보기는 과제별 규칙으로 만든다 (같은 방향의 이웃 음정, 다른 조표, 다른 화음 구성음 등).
    규칙으로 3개를 못 채우면 같은 과제의 다른 정답에서 가져온다. 정답과 같은 보기는 만들지 않는다.
  - 보기 순서는 문항 id 기반 시드로 섞는다 (재현 가능).
arxivqa: sft_pool_all.jsonl(학습 쪽, 테스트·DPO 풀과 논문이 겹치지 않음)에서 arXiv 분류별로 4지선다를 뽑는다.
  - 분류별 문항 수는 QUOTA. 정답 문자가 고르게 나오도록 문자별로 나눠 뽑는다.

사용 (서버): python build_bank3.py --out ~/mmdl/datasets/bank3_extra
"""
import argparse
import hashlib
import json
import os
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arxivqa_sft import student_prompt  # noqa: E402

DS = Path(os.path.expanduser("~/mmdl/datasets"))
QUOTA = {"eess": 300, "cs": 250, "physics": 200, "q-fin": 120, "econ": 30, "q-bio": 100}   # 합계 1,000
SHARPS = ["F#", "C#", "G#", "D#", "A#", "E#", "B#"]
FLATS = ["Bb", "Eb", "Ab", "Db", "Gb", "Cb", "Fb"]
KEYSIGS = ["none"] + [", ".join(SHARPS[:i]) for i in range(1, 8)] + [", ".join(FLATS[:i]) for i in range(1, 8)]
INTERVALS = ["minor second", "major second", "minor third", "major third", "perfect fourth", "augmented fourth",
             "perfect fifth", "minor sixth", "major sixth", "minor seventh", "major seventh", "perfect octave"]
STEPS = ["C", "D", "E", "F", "G", "A", "B"]
TIMESIGS = ["2/4", "3/4", "4/4", "6/8", "2/2", "3/8", "12/8", "5/4"]
CLEFS = ["Treble clef", "Bass clef", "Treble and Bass Clefs", "Alto clef"]
FINAL = re.compile(r"Final answer\s*:\s*(.+)")


def rng_for(rid):
    return random.Random(int(hashlib.sha256(rid.encode()).hexdigest()[:12], 16))


def shift_pitch(p, k):
    m = re.fullmatch(r"([A-G])([#b]?)(\d)", p)
    if not m:
        return None
    i = STEPS.index(m.group(1)) + k
    return f"{STEPS[i % 7]}{m.group(2)}{int(m.group(3)) + i // 7}"


def distractors(task, ans, rng):
    out = []
    if task == "key_signature" and ans in KEYSIGS:
        if ans == "none":
            out = ["F#", "Bb", "F#, C#", "Bb, Eb"]
        else:
            n = ans.count(",") + 1
            same, other = (SHARPS, FLATS) if ans.endswith("#") else (FLATS, SHARPS)
            # 같은 종류로 하나 적거나 많은 조표, 같은 개수의 반대 종류 조표
            out = [", ".join(same[:k]) for k in (n - 1, n + 1) if 1 <= k <= 7] + [", ".join(other[:n])]
            out += ["none"] + rng.sample(KEYSIGS[1:], 6)
    elif task == "named_interval":
        m = re.fullmatch(r"(ascending|descending) (.+)", ans)
        if m and m.group(2) in INTERVALS:
            d, iv = m.group(1), m.group(2)
            i = INTERVALS.index(iv)
            other = "descending" if d == "ascending" else "ascending"
            out = [f"{other} {iv}"] + [f"{d} {INTERVALS[j]}" for j in (i - 1, i + 1, i - 2, i + 2) if 0 <= j < len(INTERVALS)]
    elif task == "scale_degree" and re.fullmatch(r"[1-7]", ans):
        out = [str(k) for k in rng.sample(range(1, 8), 7)]
    elif task == "chord_member":
        m = re.fullmatch(r"([A-G][#b]?\d); (root|third|fifth)", ans)
        if m:
            p, role = m.group(1), m.group(2)
            roles = [r for r in ("root", "third", "fifth") if r != role]
            out = [f"{p}; {roles[0]}", f"{p}; {roles[1]}"] + [f"{q}; {role}" for q in (shift_pitch(p, 1), shift_pitch(p, -1)) if q]
    elif task == "time_signature":
        out = rng.sample(TIMESIGS, len(TIMESIGS))
    elif task == "clef":
        out = list(CLEFS)
    elif task in ("measure_count", "tempo") and ans.isdigit():
        n = int(ans)
        deltas = [1, -1, 2, -2, 3] if task == "measure_count" else [-12, 9, -21, 16, 27]
        out = [str(n + d) for d in deltas if n + d > 0]
    return out


def build_music(out):
    rows = [json.loads(l) for l in open(DS / "bank2_own_mc/bank2_open_reserved.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["source_slug"] == "MusiXQA"]
    pool = defaultdict(set)
    ans_of = {}
    for r in rows:
        m = FINAL.findall(r.get("target") or "")
        if m:
            ans_of[r["id"]] = m[-1].strip()
            pool[r["task"]].add(m[-1].strip())
    kept, drop = [], Counter()
    for r in rows:
        ans = ans_of.get(r["id"])
        if not ans:
            drop["정답 없음"] += 1
            continue
        if r["task"] == "scale_notes":
            drop["scale_notes(이미지 없이 풀리는 지식 문항)"] += 1
            continue
        if r["task"] == "clef" and " and " in ans:
            drop["clef: 정답이 두 음자리표라 단일 보기와 겹쳐 애매함"] += 1
            continue
        rng = rng_for(r["id"])
        ds = []
        for d in distractors(r["task"], ans, rng) + rng.sample(sorted(pool[r["task"]]), len(pool[r["task"]])):
            if d and d.lower() != ans.lower() and d.lower() not in [x.lower() for x in ds]:
                ds.append(d)
            if len(ds) == 3:
                break
        if len(ds) < 3:
            drop["오답 보기 3개를 못 만듦"] += 1
            continue
        opts = ds + [ans]
        rng.shuffle(opts)
        letter = string.ascii_uppercase[opts.index(ans)]
        kept.append({"id": r["id"] + "_mc", "bank_source": "musixqa_mc_converted", "source_slug": "MusiXQA",
                     "domain": "music", "role": r.get("role"), "task": r["task"], "group": r.get("group"),
                     "images": r["images"], "images_sha256": r.get("images_sha256"),
                     "question_stem": r["question"], "options": opts, "answer_letter": letter,
                     "original_answer": ans, "distractor_policy": "task_rule_then_same_task_pool",
                     "input_prompt": student_prompt(r["question"], opts)})
    with open(out / "music_mc.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"kept": len(kept), "dropped": dict(drop), "by_task": Counter(r["task"] for r in kept),
           "answer_letters": Counter(r["answer_letter"] for r in kept)}
    print("music", json.dumps(rep, ensure_ascii=False))
    for t in ("named_interval", "key_signature", "chord_member"):
        ex = next((r for r in kept if r["task"] == t), None)
        if ex:
            print(f"  [{t}] 정답 {ex['answer_letter']}:", ex["options"])
    return rep


def build_arxiv(out):
    root = DS / "arxivqa"
    rows = [json.loads(l) for l in open(DS / "arxivqa_prepared/sft_pool_all.jsonl", encoding="utf-8")]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cat = r["id"].rsplit("-", 1)[0]
        if cat in QUOTA and len(r["options"]) == 4 and r["answer_letter"] in "ABCD":
            by[cat][r["answer_letter"]].append(r)
    rng = random.Random(3407)
    kept = []
    for cat, q in QUOTA.items():
        per = q // 4
        picked = []
        for L in "ABCD":
            c = by[cat][L]
            rng.shuffle(c)
            picked += c[:per]
        rest = [r for L in "ABCD" for r in by[cat][L][per:]]
        rng.shuffle(rest)
        picked += rest[:q - len(picked)]
        seen_paper = Counter()
        for r in picked:
            seen_paper[r["paper_id"]] += 1
            kept.append({"id": "ax_" + r["id"], "bank_source": "arxivqa_by_category", "source_slug": "ArxivQA",
                         "domain": f"arxiv:{cat}", "role": None, "task": "figure_mc", "group": r["paper_id"],
                         "images": [str(root / r["image"])], "question_stem": None, "options": r["options"],
                         "answer_letter": r["answer_letter"], "answer_origin": "GPT-4V generated (unverified)",
                         "input_prompt": r["input_prompt"]})
    missing = [r["id"] for r in kept if not os.path.exists(r["images"][0])]
    with open(out / "arxivqa_eng_mc.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"kept": len(kept), "by_category": Counter(r["domain"] for r in kept),
           "answer_letters": Counter(r["answer_letter"] for r in kept), "missing_images": len(missing),
           "available": {c: sum(len(v) for v in by[c].values()) for c in QUOTA}}
    print("arxivqa", json.dumps(rep, ensure_ascii=False))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    rep = {"music": build_music(out), "arxivqa": build_arxiv(out)}
    (out / "report_music_arxivqa.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
