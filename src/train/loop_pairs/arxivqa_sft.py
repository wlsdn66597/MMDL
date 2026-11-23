"""ArxivQA → SFT 학습 데이터 (베이스라인 입력 템플릿 + "풀이\\nFinal answer: X").

입력: arxivqa_prep.py 결과 (train.jsonl, test.jsonl)
1. train 쪽 논문의 10%를 DPO 후보(dpo_pool)로 먼저 떼어 둔다. 논문 단위라 SFT 문항과 겹치지 않는다.
2. 풀이 점검: 풀이에 정답과 다른 보기 문자가 결론으로 적혀 있으면 뺀다. "Rationale:" 같은 머리말은 지운다.
3. 입력은 베이스라인 객관식 템플릿(hint 없음), 정답은 풀이 + "\\nFinal answer: X".
4. SFT 풀에서 고정 시드로 N개를 뽑는다. 이미지 토큰 수(평가와 같은 min/max pixels)와 텍스트 토큰 수를 기록한다.
5. 테스트에서 빠른 확인용 1,000개를 뽑는다 (전체 테스트는 그대로 둔다).

사용 (서버, 평가 환경 python)
  python arxivqa_sft.py --prepared ~/mmdl/datasets/arxivqa_prepared --images ~/mmdl/datasets/arxivqa \
      --baseline ~/mmdl/mmdl_test/outputs/baseline_results.json --diag ~/mmdl/mmdl_test/diag --n 20000
"""
import argparse
import json
import math
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path

SEED = 3407
DPO_FRAC = 0.10
MC_TEMPLATE = "{optional_hint}Question: {question}\nOptions:\n{lettered_options}Please select the correct answer from the options above."
MC_SUFFIX = 'On the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'
HEADER = re.compile(r"^\s*(\*\*)?\s*(rationale|explanation|reasoning)\s*(\*\*)?\s*[:：]\s*", re.I)
CONCL = re.compile(r"(?:answer|option|choice)\s*(?:is|would be|:)?\s*\(?\b([A-J])\b\)?(?![a-z])", re.I)


def student_prompt(question, options):
    letters = "".join(f"{l}. {v}\n" for l, v in zip(string.ascii_uppercase, options))
    return MC_TEMPLATE.format(optional_hint="", question=question, lettered_options=letters).rstrip() + "\n" + MC_SUFFIX


def image_tokens(w, h, vision):
    """qwen_vl_utils.smart_resize와 같은 규칙으로 이미지 토큰 수 계산 (patch 16, merge 2 → 32px당 1토큰)."""
    f = vision.get("patch_size", 16) * 2
    lo, hi = vision["min_pixels"], vision["max_pixels"]
    hb, wb = max(f, round(h / f) * f), max(f, round(w / f) * f)
    if hb * wb > hi:
        b = math.sqrt(h * w / hi)
        hb, wb = math.floor(h / b / f) * f, math.floor(w / b / f) * f
    elif hb * wb < lo:
        b = math.sqrt(lo / (h * w))
        hb, wb = math.ceil(h * b / f) * f, math.ceil(w * b / f) * f
    return (hb // f) * (wb // f)


def check_rationale(c):
    rat = HEADER.sub("", c["rationale"]).strip()
    letters = string.ascii_uppercase[:len(c["options"])]
    tail = rat[-200:]
    hits = [m.group(1).upper() for m in CONCL.finditer(tail) if m.group(1).upper() in letters]
    if hits and hits[-1] != c["answer_letter"]:
        return None, f"풀이 결론 문자({hits[-1]})가 정답({c['answer_letter']})과 다름"
    if len(rat.split()) < 5:
        return None, "풀이가 너무 짧음"
    return rat, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--baseline", required=True, help="vision 설정(min/max pixels)을 읽는다")
    ap.add_argument("--diag", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--n-test-quick", type=int, default=1000)
    a = ap.parse_args()
    sys.path.insert(0, a.diag)
    from common import load_baseline
    from PIL import Image
    from transformers import AutoTokenizer

    settings, _ = load_baseline(a.baseline)
    vision = settings["vision"]
    tok = AutoTokenizer.from_pretrained(settings["model"]["repo"], revision=settings["model"]["revision"])
    P = Path(a.prepared)
    train = [json.loads(l) for l in open(P / "train.jsonl", encoding="utf-8")]
    test = [json.loads(l) for l in open(P / "test.jsonl", encoding="utf-8")]

    papers = sorted({c["paper_id"] for c in train})
    rng = random.Random(SEED)
    rng.shuffle(papers)
    dpo_papers = set(papers[:round(len(papers) * DPO_FRAC)])

    def to_row(c, rat):
        target = f"{rat}\nFinal answer: {c['answer_letter']}"
        return {"id": c["id"], "paper_id": c["paper_id"], "image": c["image"],
                "input_prompt": student_prompt(c["question"], c["options"]),
                "options": c["options"], "answer_letter": c["answer_letter"], "target": target,
                "source": "MMInstruction/ArxivQA@85a6dca0", "format": "rationale+final_answer"}

    pool, dpo, dropped = [], [], []
    for c in train:
        rat, why = check_rationale(c)
        if rat is None:
            dropped.append({"id": c["id"], "reason": why})
            continue
        (dpo if c["paper_id"] in dpo_papers else pool).append(to_row(c, rat))
    rng = random.Random(SEED)
    sft = rng.sample(pool, min(a.n, len(pool)))
    test_rows = []
    for c in test:
        rat, why = check_rationale(c)
        test_rows.append(dict(to_row(c, rat or c["rationale"]), rationale_check=why))
    quick = random.Random(SEED).sample(test_rows, min(a.n_test_quick, len(test_rows)))

    # 토큰 수 (SFT 선택분)
    for r in sft:
        w, h = Image.open(Path(a.images) / r["image"]).size
        r["image_tokens"] = image_tokens(w, h, vision)
        r["prompt_text_tokens"] = len(tok(r["input_prompt"], add_special_tokens=False)["input_ids"])
        r["target_tokens"] = len(tok(r["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"])

    def dump(name, rows):
        with open(P / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(f"sft_train_{len(sft) // 1000}k.jsonl", sft)
    dump("sft_pool_all.jsonl", pool)
    dump("dpo_pool.jsonl", dpo)
    dump("test_full.jsonl", test_rows)
    dump("test_quick1k.jsonl", quick)
    dump("sft_dropped.jsonl", dropped)

    def stats(xs):
        xs = sorted(xs)
        q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]
        return {"mean": round(sum(xs) / len(xs), 1), "p50": q(0.5), "p90": q(0.9), "p99": q(0.99), "max": xs[-1]}

    tot = [r["image_tokens"] + r["prompt_text_tokens"] + r["target_tokens"] + 10 for r in sft]
    rep = {"seed": SEED, "train_in": len(train), "dropped_rationale": len(dropped),
           "dropped_reasons": Counter(d["reason"].split("(")[0] for d in dropped),
           "dpo_frac_papers": DPO_FRAC, "dpo_pool": len(dpo), "sft_pool": len(pool), "sft_selected": len(sft),
           "sft_papers": len({r["paper_id"] for r in sft}),
           "overlap_papers_sft_dpo": len({r["paper_id"] for r in sft} & {r["paper_id"] for r in dpo}),
           "overlap_papers_sft_test": len({r["paper_id"] for r in sft} & {r["paper_id"] for r in test_rows}),
           "test_full": len(test_rows), "test_quick": len(quick),
           "test_rationale_flagged": sum(1 for r in test_rows if r["rationale_check"]),
           "answer_letter_dist_sft": Counter(r["answer_letter"] for r in sft),
           "tokens_sft": {"image": stats([r["image_tokens"] for r in sft]),
                          "target": stats([r["target_tokens"] for r in sft]),
                          "total_approx": stats(tot), "sum_total_M": round(sum(tot) / 1e6, 2)},
           "vision": {k: vision[k] for k in ("min_pixels", "max_pixels")}}
    (P / "sft_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    ex = sft[0]
    print("\n=== 예시", ex["id"], "===\n[입력]\n" + ex["input_prompt"] + "\n[정답]\n" + ex["target"])


if __name__ == "__main__":
    main()
