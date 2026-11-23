"""LoRA 선호 학습 (DPO + 생성 손실, Qwen3-VL-4B-Instruct). 데이터는 build_dpo_pairs.py 출력.

손실 (쌍 하나)
  m    = [log π(c) − log π_ref(c)] − [log π(r) − log π_ref(r)]         π_ref = 학습 전 베이스 (어댑터를 끈 모델)
  loss = −log σ(β·m) + λ·(CE_reason(c) + CE_answer(c))                 뒤 항은 train_sft.py와 같은 생성 손실
  - c = 고르는 풀이 전체 (+<|im_end|>). 로그 확률은 토큰 합.
  - r (wrong) = 오답 풀이 전체 (+<|im_end|>).
  - r (loop)  = 루프 시작 앞부분을 문맥으로 주고, 루프 구간 seg토큰의 로그 확률만 쓴다
                (루프 앞의 정상 풀이까지 확률을 낮추지 않기 위해서).
메모리: 두 풀이를 한꺼번에 역전파하지 않는다. r을 먼저 계산해 기울기를 따로 받아 두고(∇log π(r)),
  c를 계산한 뒤 계수 w = β·σ(−β·m)를 곱해 더한다. 결과 기울기는 보통의 DPO와 같다.
참조 로그 확률은 쌍마다 처음 볼 때 한 번 계산해 ref_logp.json에 저장한다.
save-every 스텝마다 어댑터를 저장하고 검증 쌍에서 선호 정확도(m > 0 비율) 등을 잰다.

사용 (서버, 학습 환경)
  ~/mmdl/.venv-train/bin/python train_dpo.py --data ~/mmdl/datasets/dpo_r1/train.jsonl --val ~/mmdl/datasets/dpo_r1/val.jsonl \
      --baseline ~/mmdl/mmdl_test/outputs/baseline_results.json --diag ~/mmdl/mmdl_test/diag --out ~/mmdl/train_runs/dpo_r1
  메모리 확인:  ... --longest-first --max-steps 2 --grad-accum 4
  merge:        ... --merge-only --adapter-name adapter_step20 --out <같은 폴더>
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_sft import LORA_TARGET, md5, merge_and_save, split_target  # noqa: E402


class PairData:
    def __init__(self, rows, proc, vision):
        self.rows, self.proc, self.vision = rows, proc, vision

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        from qwen_vl_utils import process_vision_info
        r = self.rows[i]
        content = [dict(type="image", image=Image.open(p).convert("RGB"), min_pixels=self.vision["min_pixels"],
                        max_pixels=self.vision["max_pixels"]) for p in r["images"]]
        msgs = [{"role": "user", "content": content + [{"type": "text", "text": r["input_prompt"]}]}]
        imgs, _, _ = process_vision_info(msgs, image_patch_size=self.vision.get("patch_size", 16),
                                         return_video_kwargs=True, return_video_metadata=True)
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.proc(text=[text], images=imgs, do_resize=False, return_tensors="pt")
        tok = self.proc.tokenizer
        ids = lambda s: tok(s, add_special_tokens=False, return_tensors="pt")["input_ids"]
        reason, answer = split_target(r["chosen"])
        c_r, c_a = ids(reason), ids(answer + "<|im_end|>")
        if r["rej_type"] == "loop":
            full = ids(r["rejected"])
            rej = full[:, :r["rej_loop_start"] + r["seg"]]
            rej_n = rej.shape[1] - r["rej_loop_start"]
        else:
            rej = ids(r["rejected"].rstrip() + "<|im_end|>")
            rej_n = rej.shape[1]
        import torch
        return {"pair_id": r["pair_id"], "rej_type": r["rej_type"], "prompt_ids": enc["input_ids"],
                "pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"],
                "cho": torch.cat([c_r, c_a], dim=1), "r_len": c_r.shape[1], "rej": rej, "rej_n": rej_n}


def token_logp(model, b, tgt, n_score):
    """프롬프트 + tgt를 넣고 tgt의 마지막 n_score 토큰 각각의 로그 확률을 돌려준다."""
    import torch
    import torch.nn.functional as F
    ids = torch.cat([b["prompt_ids"], tgt], dim=1).cuda()
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=b["pixel_values"].cuda(),
              image_grid_thw=b["image_grid_thw"].cuda())
    try:
        logits = model(**kw, logits_to_keep=n_score + 1).logits[:, :-1]
    except TypeError:
        logits = model(**kw).logits[:, -n_score - 1:-1]
    return -F.cross_entropy(logits[0].float(), ids[0, -n_score:], reduction="none")


def ref_logp(model, b, cache):
    import torch
    if b["pair_id"] not in cache:
        was = model.training
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            c = token_logp(model, b, b["cho"], b["cho"].shape[1]).sum().item()
            r = token_logp(model, b, b["rej"], b["rej_n"]).sum().item()
        model.train(was)
        cache[b["pair_id"]] = [c, r]
    return cache[b["pair_id"]]


def too_long(b, max_tokens):
    p = b["prompt_ids"].shape[1]
    return p + max(b["cho"].shape[1], b["rej"].shape[1]) > max_tokens


def evaluate(model, data, cache, beta, max_tokens):
    import torch
    model.eval()
    rec = []
    with torch.no_grad():
        for i in range(len(data)):
            b = data[i]
            if too_long(b, max_tokens):
                continue
            rc, rr = ref_logp(model, b, cache)
            lc = token_logp(model, b, b["cho"], b["cho"].shape[1]).sum().item()
            lr = token_logp(model, b, b["rej"], b["rej_n"]).sum().item()
            rec.append((b["rej_type"], lc - rc, lr - rr, b["cho"].shape[1], b["rej_n"]))
    model.train()
    out = {"n": len(rec)}
    for typ in ("all", "wrong", "loop"):
        v = [x for x in rec if typ == "all" or x[0] == typ]
        if v:
            out[typ] = {"n": len(v), "pref_acc": round(sum((x[1] - x[2]) > 0 for x in v) / len(v), 3),
                        "margin_mean": round(sum(x[1] - x[2] for x in v) / len(v), 3),
                        "chosen_delta_per_tok": round(sum(x[1] for x in v) / sum(x[3] for x in v), 5),
                        "rejected_delta_per_tok": round(sum(x[2] for x in v) / sum(x[4] for x in v), 5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--w-gen", type=float, default=1.0, help="생성 손실 가중치 λ")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--longest-first", action="store_true", help="긴 쌍부터 (메모리 확인용)")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--adapter-name", default="adapter")
    a = ap.parse_args()
    sys.path.insert(0, a.diag)
    from common import load_baseline
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    settings, _ = load_baseline(a.baseline)
    repo, rev, vision = settings["model"]["repo"], settings["model"]["revision"], settings["vision"]
    out = Path(a.out)
    if a.merge_only:
        print("merge:", merge_and_save(repo, rev, out / a.adapter_name, out / "merged"))
        return
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"출력 폴더가 비어 있지 않다: {out}")
    out.mkdir(parents=True, exist_ok=True)

    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    est = lambda r: r["input_tokens"] + max(r["chosen_tokens"], (r["rej_loop_start"] or 0) + (r["seg"] or r["rejected_tokens"]))
    if a.longest_first:
        rows.sort(key=est, reverse=True)
    proc = AutoProcessor.from_pretrained(repo, revision=rev)
    data = PairData(rows, proc, vision)
    vdata = PairData([json.loads(l) for l in open(a.val, encoding="utf-8") if l.strip()], proc, vision) if a.val else None
    total_steps = a.max_steps or math.ceil(math.ceil(len(data) / a.grad_accum) * a.epochs)
    info = {"args": vars(a), "base": {"repo": repo, "revision": rev}, "vision": vision, "data_md5": md5(a.data),
            "n_pairs": len(rows), "total_optimizer_steps": total_steps, "code_md5": md5(__file__), "torch": torch.__version__}
    (out / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    model = Qwen3VLForConditionalGeneration.from_pretrained(repo, revision=rev, dtype=torch.bfloat16,
                                                            attn_implementation="flash_attention_2").cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                                             target_modules=LORA_TARGET, task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"학습 파라미터 {sum(p.numel() for p in params):,}, 쌍 {len(rows)}, 옵티마이저 스텝 {total_steps}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.999))
    warm = max(1, int(total_steps * a.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps)))))

    cache = {}
    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    vlog = open(out / "val_log.jsonl", "a", encoding="utf-8")
    keys = ("dpo", "gen_r", "gen_a", "margin", "w", "dc", "dr")
    acc = dict.fromkeys(keys, 0.0) | {"n": 0, "skipped": 0, "tok": 0}
    step, micro, epoch, t0 = 0, 0, 0, time.time()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    while step < total_steps:
        order = list(range(len(data)))
        if not a.longest_first:
            random.Random(a.seed + epoch).shuffle(order)
        loader = torch.utils.data.DataLoader(torch.utils.data.Subset(data, order), batch_size=None, shuffle=False,
                                             num_workers=a.workers)
        for b in loader:
            if too_long(b, a.max_tokens):
                acc["skipped"] += 1
                continue
            rc, rr = ref_logp(model, b, cache)
            lr_sum = token_logp(model, b, b["rej"], b["rej_n"]).sum()
            g_r = torch.autograd.grad(lr_sum, params, allow_unused=True)
            lr_v = lr_sum.item()
            del lr_sum
            lp_c = token_logp(model, b, b["cho"], b["cho"].shape[1])
            lc = lp_c.sum()
            m = (lc.item() - rc) - (lr_v - rr)
            w = a.beta / (1.0 + math.exp(min(50.0, a.beta * m)))          # β·σ(−β·m)
            gen_r, gen_a = -lp_c[:b["r_len"]].mean(), -lp_c[b["r_len"]:].mean()
            ((-w * lc + a.w_gen * (gen_r + gen_a)) / a.grad_accum).backward()
            for p, g in zip(params, g_r):
                if g is not None:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.add_(g, alpha=w / a.grad_accum)
            del g_r
            micro += 1
            bm = a.beta * m
            acc["dpo"] += math.log1p(math.exp(-bm)) if bm > -30 else -bm
            acc["gen_r"] += gen_r.item(); acc["gen_a"] += gen_a.item(); acc["margin"] += m; acc["w"] += w
            acc["dc"] += lc.item() - rc; acc["dr"] += lr_v - rr; acc["n"] += 1
            acc["tok"] += 2 * b["prompt_ids"].shape[1] + b["cho"].shape[1] + b["rej"].shape[1]
            if micro % a.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            step += 1
            el = time.time() - t0
            rec = {"step": step, "epoch": epoch} | {k: round(acc[k] / acc["n"], 4) for k in keys} | {
                "lr": sched.get_last_lr()[0], "tokens_per_s": round(acc["tok"] / max(1e-6, el), 1),
                "sec_per_step": round(el / step, 1), "eta_min": round(el / step * (total_steps - step) / 60, 1),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "skipped": acc["skipped"]}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
            for k in keys:
                acc[k] = 0.0
            acc["n"] = 0
            if step % a.save_every == 0 or step >= total_steps:
                name = "adapter" if step >= total_steps else f"adapter_step{step}"
                model.save_pretrained(out / name)
                (out / "ref_logp.json").write_text(json.dumps(cache) + "\n", encoding="utf-8")
                if vdata is not None:
                    v = {"step": step} | evaluate(model, vdata, cache, a.beta, a.max_tokens)
                    vlog.write(json.dumps(v, ensure_ascii=False) + "\n"); vlog.flush()
                    print("검증:", v, flush=True)
            if step >= total_steps:
                break
        epoch += 1
    print("끝. 어댑터:", out / "adapter", flush=True)


if __name__ == "__main__":
    main()
