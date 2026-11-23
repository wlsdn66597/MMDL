"""2라운드 샘플링 대상: 아직 샘플링하지 않은 1번 은행(MMFR 5k) 문항 중 루프 쌍이 잘 나오는 것만 고른다.

근거 (1라운드 2,000문항 결과)
  - 루프 쌍이 나온 비율: WaltonColdStart 12%, visualwebinstruct 9%, BMMR 8%, geometry3k 7%, MMR1 4%, LLaVA-CoT 3%, ViRL39K 1%
  - MMR1은 교사 풀이가 짧은 아래 3분의 1에서 3%, 나머지에서 7%
선정
  1단: MMR1·LLaVA-CoT·ViRL39K를 뺀 출처 전부
  2단: MMR1 중 교사 풀이(target) 길이가 --mmr1-min-chars 이상인 문항
  LLaVA-CoT, ViRL39K, 짧은 MMR1은 뺀다.
순서: 1단 → 2단. 단 안에서는 seed로 섞는다. 앞에서부터 샘플링하다가 중간에 끊어도 된다.
MMMU 중복 검사는 mmfr123k_prepared를 만들 때 이미 했다.

출력: bank_r2.jsonl (sample_bank.py 입력 형식, 이미지는 절대 경로), report.json
사용 (서버): python build_bank_r2.py --out ~/mmdl/datasets/bank_r2
"""
import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

HOME = Path(os.path.expanduser("~/mmdl"))
ROOT = HOME / "datasets/mmfr123k_prepared"
SAMPLED = HOME / "train_runs/bank_sampling_base_n8/samples.jsonl"
SKIP = {"LLaVA-CoT", "ViRL39K"}
SEED = 3407


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmr1-min-chars", type=int, default=6200)
    a = ap.parse_args()
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(SAMPLED, encoding="utf-8")}
    rows = [json.loads(l) for l in open(ROOT / "sft_mc_5k.jsonl", encoding="utf-8")]
    un = [r for r in rows if r["id"] not in done]
    t1 = [r for r in un if r["source"] not in SKIP and r["source"] != "MMR1"]
    t2 = [r for r in un if r["source"] == "MMR1" and len(r["target"]) >= a.mmr1_min_chars]
    rng = random.Random(SEED)
    rng.shuffle(t1)
    rng.shuffle(t2)
    kept, missing = [], 0
    for tier, part in ((1, t1), (2, t2)):
        for r in part:
            p = str(ROOT / r["image"])
            if not os.path.exists(p):
                missing += 1
                continue
            kept.append({"id": r["id"], "source": r["source"], "tier": tier, "images": [p], "input_prompt": r["input_prompt"],
                         "options": r["options"], "answer_letter": r["answer_letter"], "teacher_chars": len(r["target"])})
    with open(out / "bank_r2.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"unsampled": len(un), "kept": len(kept), "missing_images": missing,
           "by_tier": Counter(r["tier"] for r in kept), "by_source": Counter(r["source"] for r in kept),
           "excluded": {"LLaVA-CoT/ViRL39K": sum(r["source"] in SKIP for r in un),
                        "short MMR1": sum(r["source"] == "MMR1" and len(r["target"]) < a.mmr1_min_chars for r in un)},
           "answer_letters": Counter(r["answer_letter"] for r in kept),
           "settings": {"mmr1_min_chars": a.mmr1_min_chars, "seed": SEED}}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
