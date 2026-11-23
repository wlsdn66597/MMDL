"""2차 선호 학습(DPO) 데이터 만들기: 같은 문항에서 고르는 풀이와 버리는 풀이의 쌍.

입력: sample_bank.py 결과 두 개 (문항당 응답 8개, 출력 상한 8192).
고르는 풀이(chosen): 정답, 정상 종료, 마지막 줄 "Final answer: X", MIN_TOK 토큰 이상인 응답 중 가장 짧은 것.
버리는 풀이(rejected): 두 종류. 문항마다 종류별로 최대 1개씩 쌍을 만든다.
  - wrong : 정상 종료했고 답을 냈지만 오답 (k가 가장 작은 것). 8번 중 --min-pass-wrong번 이상 맞힌 문항만 쓴다
            (1~2번만 맞힌 문항은 찍어서 맞힌 풀이가 고르는 풀이로 들어갈 위험이 커서 뺀다).
  - loop  : 8192토큰에서 잘렸고 반복 루프가 확인된 응답 (k가 가장 작은 것). 8번 중 --min-pass-loop번 이상 맞힌 문항.
            rej_loop_start = 루프가 시작되는 토큰 위치. 학습 때는 그 앞을 문맥으로 주고 루프 구간 SEG토큰의 확률만 낮춘다.
            잘렸지만 루프가 아닌 응답(그냥 긴 풀이)은 쓰지 않는다.
  루프 시작: 12토큰 구가 앞에 나온 적 있는 위치의 비율이 256토큰 창에서 0.8 이상이고, 그 뒤 전체에서 0.5 이상인 첫 위치.
문항: 고르는 풀이와 버리는 풀이가 모두 있는 문항. 수학 은행은 출처별 문항 상한(--cap-math-source)을 둔다.
입력 + 풀이 길이가 --max-tokens를 넘는 쌍은 뺀다 (학습 메모리).
출력: train.jsonl, val.jsonl (id, pair_id, images, input_prompt, chosen, rejected, rej_type, …), report.json

버리는 풀이가 길어서(최대 8192토큰) 학습 때 앞부분만 잘라 쓸 수 있도록 토큰 수를 같이 적는다.

사용 (서버): python build_dpo_pairs.py --out ~/mmdl/datasets/dpo_r1
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

HOME = Path(os.path.expanduser("~/mmdl"))
MATH = {"samples": HOME / "train_runs/bank_sampling_base_n8/samples.jsonl",
        "data": [HOME / "datasets/mmfr123k_prepared/sft_mc_5k.jsonl", HOME / "datasets/mmfr123k_prepared/val_200.jsonl"],
        "root": HOME / "datasets/mmfr123k_prepared"}
NONMATH = {"samples": HOME / "train_runs/bank_sampling_nonmath_n8/samples.jsonl",
           "data": [HOME / "datasets/bank_nonmath/bank_all.jsonl"], "root": Path("/")}
FINAL = re.compile(r"Final answer: ([A-J])")
MIN_TOK = 50
SEED = 3407
SEG = 512          # 루프 구간에서 확률을 낮출 토큰 수
REPO, REV = "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def loop_start(ids, n=12, win=256, th=0.8):
    seen, rep = set(), []
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        rep.append(1 if g in seen else 0)
        seen.add(g)
    if len(rep) < win:
        return None
    pre = [0]
    for x in rep:
        pre.append(pre[-1] + x)
    for i in range(len(rep) - win + 1):
        if rep[i] and pre[i + win] - pre[i] >= th * win and (pre[-1] - pre[i]) / (len(rep) - i) >= 0.5:
            return i
    return None


def loop_rate(text, window=4096, n=12):
    w = text.split()[-window:]
    if len(w) < n * 4:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    c = Counter(grams)
    return round(sum(v for v in c.values() if v > 1) / len(grams), 3)


def collect(bank, name, tok, a):
    meta = {}
    for p in bank["data"]:
        for l in open(p, encoding="utf-8"):
            r = json.loads(l)
            meta.setdefault(r["id"], r)
    by = defaultdict(list)
    for sp in [bank["samples"]] + list(bank.get("extra", [])):
        for l in open(sp, encoding="utf-8"):
            r = json.loads(l)
            by[r["id"]].append(r)
    items, why = [], Counter()
    for rid, v in by.items():
        good, wrong, trunc = [], [], []
        for x in v:
            last = x["raw"].rstrip().split("\n")[-1].strip()
            m = FINAL.fullmatch(last)
            if x["finish_reason"] == "length":
                trunc.append(x)
            elif x["correct"] and x["finish_reason"] == "stop" and x["output_tokens"] >= MIN_TOK and m \
                    and m.group(1) == x["answer_letter"]:
                good.append(x)
            elif x["finish_reason"] == "stop" and x["correct"] is False and x["parse"].get("answer"):
                wrong.append(x)
        if not good:
            why["고르는 풀이 없음"] += 1
            continue
        if not wrong and not trunc:
            why["버리는 풀이 없음"] += 1
            continue
        m = meta[rid]
        paths = [str(bank["root"] / p) for p in (m.get("images") or [m["image"]])]
        if not all(os.path.exists(p) for p in paths):
            why["이미지 없음"] += 1
            continue
        src = m.get("source") or name
        if src.startswith("arxivqa"):
            src = m.get("domain") or src
        ch = min(good, key=lambda g: (g["output_tokens"], g["k"]))
        pairs = []
        n_pass = sum(bool(x["correct"]) for x in v)
        for typ, pool in (("wrong", wrong), ("loop", trunc)):
            if pool:
                rj = min(pool, key=lambda g: g["k"])
                ls = None
                if typ == "wrong":
                    if a.only_loop:
                        continue
                    if n_pass < a.min_pass_wrong:
                        why["wrong: 정답 횟수 부족"] += 1
                        continue
                    need = max(ch["output_tokens"], rj["output_tokens"])
                else:
                    if n_pass < a.min_pass_loop:
                        why["loop: 정답 횟수 부족"] += 1
                        continue
                    if a.loop_pick == "first":
                        cand = [(loop_start(tok(rj["raw"].rstrip(), add_special_tokens=False)["input_ids"]), rj)]
                    else:   # best: 잘린 응답을 모두 보고 루프가 가장 일찍 시작하는 것을 고른다
                        cand = [(loop_start(tok(x["raw"].rstrip(), add_special_tokens=False)["input_ids"]), x) for x in pool]
                    cand = [(s, x) for s, x in cand if s is not None]
                    if not cand:
                        why["잘렸지만 루프 아님"] += 1
                        continue
                    ls, rj = min(cand, key=lambda c: (c[0], c[1]["k"]))
                    need = max(ch["output_tokens"], ls + SEG)
                if ch["input_tokens"] + need + 8 > a.max_tokens:
                    why[f"{typ}: 길이 초과"] += 1
                    continue
                pairs.append({"id": rid, "pair_id": f"{rid}__{typ}", "bank": name, "source": src, "images": paths,
                              "input_prompt": m["input_prompt"], "chosen": ch["raw"].rstrip(), "rejected": rj["raw"].rstrip(),
                              "rej_type": typ, "rej_answer": rj["parse"].get("answer"), "answer_letter": ch["answer_letter"],
                              "pass_count": n_pass, "n_trunc": len(trunc), "rej_loop_start": ls, "seg": SEG if ls is not None else None,
                              "chosen_tokens": ch["output_tokens"], "rejected_tokens": rj["output_tokens"],
                              "input_tokens": ch["input_tokens"],
                              "loop_rate": loop_rate(rj["raw"]) if typ == "loop" else None})
        if pairs:
            items.append((rid, src, pairs))
    return items, why


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap-math-source", type=int, default=100)
    ap.add_argument("--val-items", type=int, default=40)
    ap.add_argument("--min-pass-wrong", type=int, default=3)
    ap.add_argument("--min-pass-loop", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--only-loop", action="store_true", help="루프 쌍만 만든다 (오답 쌍 제외)")
    ap.add_argument("--loop-pick", choices=["first", "best"], default="first",
                    help="first: k가 가장 작은 잘린 응답만 본다 (dpo_r1, dpo_r2_loop를 만든 방식). best: 잘린 응답 전부에서 고른다")
    ap.add_argument("--extra-math", nargs="*", default=[], help="추가 샘플링 결과 samples.jsonl (1번 은행과 같은 문항 파일 기준)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO, revision=REV)
    out = Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    MATH["extra"] = [os.path.expanduser(p) for p in a.extra_math]
    math, why_m = collect(MATH, "math", tok, a)
    non, why_n = collect(NONMATH, "nonmath", tok, a)
    by = defaultdict(list)
    for it in sorted(math):
        by[it[1]].append(it)
    kept = []
    for s in sorted(by):
        rng.shuffle(by[s])
        kept += by[s][:a.cap_math_source]
    items = kept + sorted(non)
    rng.shuffle(items)
    val_items, train_items = items[:a.val_items], items[a.val_items:]
    parts = {}
    for name, its in (("train", train_items), ("val", val_items)):
        rows = [p for it in its for p in it[2]]
        rng.shuffle(rows)
        parts[name] = rows
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tr = parts["train"]
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(len(xs) * p))] if xs else None
    ls = [r["rej_loop_start"] for r in tr if r["rej_type"] == "loop"]
    rep = {"math_items_available": len(math), "math_items_kept": len(kept), "nonmath_items": len(non),
           "dropped_math": dict(why_m), "dropped_nonmath": dict(why_n),
           "train_items": len(train_items), "train_pairs": len(tr), "val_items": len(val_items), "val_pairs": len(parts["val"]),
           "train_pairs_by_type": Counter(r["rej_type"] for r in tr),
           "train_pairs_by_bank_type": Counter(f"{r['bank']}:{r['rej_type']}" for r in tr),
           "train_pairs_by_source": Counter(r["source"] for r in tr),
           "train_items_by_pass_count": Counter(it[2][0]["pass_count"] for it in train_items),
           "chosen_tokens": {"median": q([r["chosen_tokens"] for r in tr], .5), "p90": q([r["chosen_tokens"] for r in tr], .9),
                             "max": max(r["chosen_tokens"] for r in tr)},
           "wrong_rejected_tokens": {"median": q([r["rejected_tokens"] for r in tr if r["rej_type"] == "wrong"], .5),
                                     "p90": q([r["rejected_tokens"] for r in tr if r["rej_type"] == "wrong"], .9)}
           if any(r["rej_type"] == "wrong" for r in tr) else None,
           "loop_start_tokens": {"median": q(ls, .5), "p90": q(ls, .9), "n": len(ls)},
           "train_pairs_by_pass_count": Counter(r["pass_count"] for r in tr),
           "settings": {"min_tok": MIN_TOK, "cap_math_source": a.cap_math_source, "seed": SEED, "seg": SEG,
                        "only_loop": a.only_loop, "loop_pick": a.loop_pick, "extra_math": a.extra_math, "min_pass_wrong": a.min_pass_wrong, "min_pass_loop": a.min_pass_loop, "max_tokens": a.max_tokens}}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
