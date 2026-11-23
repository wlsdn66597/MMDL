"""문제 은행 다중 샘플링 (학습 데이터용): 베이스 모델이 문항당 n번 풀게 하고 답·정오·종료 상태를 기록한다.

- 입력은 베이스라인과 같은 방식으로 만든다: process_vision_info(min/max pixels) → chat template → do_resize=False.
- 샘플링 설정은 베이스라인 그대로 (temperature·top_p·top_k·presence/repetition penalty). 출력 상한만 --max-tokens.
- 한 요청에서 n개를 같이 생성한다 (이미지 인코딩 한 번). 루프 조기 종료는 하지 않는다:
  베이스 루프는 대부분 8k 토큰 뒤에 시작돼 8192 상한 안에서는 아끼는 시간이 거의 없다 (베이스라인 900문항으로 확인).
- 문항 묶음(--chunk)마다 결과를 저장하고, 같은 --out으로 다시 실행하면 끝난 문항은 건너뛴다.
- 이 스크립트의 응답은 MMMU가 아닌 학습용 문제 은행에서만 만든다. MMMU 응답을 학습에 쓰지 않는다.

출력 한 줄 = 한 응답: id, k, source, answer_letter, raw, output_tokens, finish_reason, parse(status/answer/reason), correct
요약: <out>.summary.json (문항별 정답 수 분포, 잘림 수, 답 추출 실패 수)

사용 (서버, 평가 환경)
  HF_HUB_OFFLINE=1 ~/mmdl/mmdl_test/.venv/bin/python sample_bank.py \
      --data ~/mmdl/datasets/mmfr123k_prepared/sft_mc_5k.jsonl ~/mmdl/datasets/mmfr123k_prepared/val_200.jsonl \
      --images ~/mmdl/datasets/mmfr123k_prepared --baseline ~/mmdl/mmdl_test/outputs/baseline_results.json \
      --diag ~/mmdl/mmdl_test/diag --out ~/mmdl/train_runs/bank_sampling_base_n8/samples.jsonl
"""
import argparse
import json
import os
import string
import sys
import time
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--baseline", required=True, help="모델·샘플링·vision 설정을 읽는다")
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--eager", action="store_true", help="CUDA graph를 끈다 (기본은 켬, 속도 우선)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    sys.path.insert(0, a.diag)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from common import load_baseline, parse_mc
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    settings, _ = load_baseline(a.baseline)
    model, s, vision = settings["model"], settings["sampling"], settings["vision"]
    rows = []
    for p in a.data:
        rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["id"])
    todo = [r for r in rows if r["id"] not in done]
    print(f"문항 {len(rows)}, 완료 {len(done)}, 실행 {len(todo)}, 문항당 {a.n}개, 출력 상한 {a.max_tokens}", flush=True)

    info = {"model": model, "sampling": s, "vision": vision, "n": a.n, "max_tokens": a.max_tokens,
            "seed": a.seed, "data": a.data, "max_model_len": a.max_model_len, "eager": a.eager,
            "loop_stop": False, "code": "sample_bank.py"}
    out.with_suffix(".run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    if todo:
        proc = AutoProcessor.from_pretrained(model["repo"], revision=model["revision"])
        llm = LLM(model=model["repo"], revision=model["revision"], dtype=model.get("dtype", "bfloat16"),
                  max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem, max_num_seqs=a.max_num_seqs,
                  enforce_eager=a.eager, limit_mm_per_prompt={"image": 1}, seed=a.seed,
                  mm_processor_kwargs={"size": {"shortest_edge": vision["min_pixels"],
                                                "longest_edge": vision["max_pixels"]}, "do_resize": False})
        sp = SamplingParams(n=a.n, temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
                            repetition_penalty=s["repetition_penalty"], presence_penalty=s["presence_penalty"],
                            seed=a.seed, max_tokens=a.max_tokens)
        t0 = time.time()
        n_done = 0
        for c in range(0, len(todo), a.chunk):
            chunk = todo[c:c + a.chunk]
            prompts = []
            for r in chunk:
                paths = r.get("images") or [r["image"]]
                content = [dict(type="image", image=Image.open(Path(a.images) / p).convert("RGB"),
                                min_pixels=vision["min_pixels"], max_pixels=vision["max_pixels"]) for p in paths]
                msgs = [{"role": "user", "content": content + [{"type": "text", "text": r["input_prompt"]}]}]
                imgs, _, kw = process_vision_info(msgs, image_patch_size=vision.get("patch_size", 16),
                                                  return_video_kwargs=True, return_video_metadata=True)
                kw = dict(kw or {})
                kw["do_resize"] = False
                text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                prompts.append({"prompt": text, "multi_modal_data": {"image": imgs}, "mm_processor_kwargs": kw})
            outs = llm.generate(prompts, sp, use_tqdm=False)
            with open(out, "a", encoding="utf-8") as f:
                for r, o in zip(chunk, outs):
                    opts = dict(zip(string.ascii_uppercase, r["options"]))
                    for k, g in enumerate(o.outputs):
                        st, ans, why = parse_mc(g.text, opts)
                        f.write(json.dumps({
                            "id": r["id"], "k": k, "source": r.get("source"), "answer_letter": r["answer_letter"],
                            "raw": g.text, "output_tokens": len(g.token_ids), "input_tokens": len(o.prompt_token_ids),
                            "finish_reason": g.finish_reason,
                            "parse": {"status": st, "answer": ans, "reason": why},
                            "correct": None if st != "extracted" else ans == r["answer_letter"]},
                            ensure_ascii=False) + "\n")
            n_done += len(chunk)
            el = time.time() - t0
            print(f"[{len(done) + n_done}/{len(rows)}] {el / 60:.1f}분 경과, 남은 예상 "
                  f"{el / n_done * (len(todo) - n_done) / 60:.0f}분", flush=True)

    # 요약
    per = {}
    trunc = unparsed = total = 0
    for l in open(out, encoding="utf-8"):
        x = json.loads(l)
        total += 1
        per.setdefault(x["id"], 0)
        per[x["id"]] += bool(x["correct"])
        trunc += x["finish_reason"] == "length"
        unparsed += x["parse"]["status"] != "extracted"
    dist = Counter(per.values())
    summ = {"items": len(per), "responses": total, "truncated_8192": trunc, "unparsed": unparsed,
            "pass_count_dist": {str(k): dist.get(k, 0) for k in range(a.n + 1)},
            "usable_items_1_to_n-2": sum(v for k, v in dist.items() if 1 <= k <= a.n - 2)}
    out.with_suffix(".summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main()
