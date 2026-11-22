"""베이스라인 모델로 MMMU val을 다시 생성 (진단용, 학습 없음).

베이스라인 JSON의 rendered_chat(실제 입력 문자열)과 설정(샘플링·이미지·런타임)을 그대로 쓴다.
GPU 서버(베이스라인을 돌린 vLLM 0.11 환경)에서 실행한다.

주요 옵션
  --seed            요청·엔진 seed (베이스라인은 3407)
  --max-num-seqs    동시에 처리할 요청 수. 1이면 공식 평가와 같은 순차 처리
  --loop-stop       반복 루프 감지 시 생성 중단 (진단용 실행에서만 사용)
  --limit / --ids-file / --types   일부 문항만 실행

출력: 문항별 JSONL (중간에 끊겨도 --out을 같게 주면 이어서 실행)

사용 예
  # 동작 확인 (5문항)
  python run_generate.py --baseline baseline_results.json --out runs/smoke.jsonl --seed 3407 --limit 5
  # 배치 동등성 확인 (조기 종료 없음)
  python run_generate.py --baseline baseline_results.json --out runs/seed3407_batch.jsonl --seed 3407 --max-num-seqs 16
  # seed 추가 실행 (루프 조기 종료)
  python run_generate.py --baseline baseline_results.json --out runs/seed1.jsonl --seed 1 --max-num-seqs 16 --loop-stop
"""
import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path

from common import (LOOP_CONSEC, LOOP_TH, LOOP_WINDOW, baseline_type,
                    load_baseline, ngram_repeat_rate)


class ImageSource:
    """MMMU 원본 이미지를 고정 revision에서 읽어 베이스라인과 같은 방식으로 resize."""

    def __init__(self, settings):
        self.ds_cfg = settings["dataset"]
        self.vision = settings["vision"]
        self.cache = {}
        # 여러 스레드가 동시에 load_dataset을 부르면 datasets 내부 tqdm 잠금에서 충돌이 난다
        # (AttributeError: type object 'tqdm' has no attribute '_lock'). 불러오기만 한 번에 하나씩 한다.
        import threading
        self.lock = threading.Lock()

    def _dataset(self, config):
        with self.lock:
            if config not in self.cache:
                from datasets import load_dataset
                ds = load_dataset(self.ds_cfg["repo"], config,
                                  split=self.ds_cfg.get("split", "validation"),
                                  revision=self.ds_cfg["revision"])
                self.cache[config] = (ds, {rid: i for i, rid in enumerate(ds["id"])})
            return self.cache[config]

    def images(self, rec):
        """src/evaluate.py와 같은 방식: 이미지 dict에 min/max_pixels를 넣고
        process_vision_info로 resize, 요청별 processor 인자에 do_resize=False."""
        from qwen_vl_utils import process_vision_info
        ds, index = self._dataset(rec["dataset_config"])
        row = ds[index[rec["id"]]]
        content = [dict(type="image", image=row[f].convert("RGB"),
                        min_pixels=self.vision["min_pixels"], max_pixels=self.vision["max_pixels"])
                   for f in rec["image_fields"]]
        msgs = [{"role": "user", "content": content + [{"type": "text", "text": rec["input_prompt"]}]}]
        images, videos, kwargs = process_vision_info(
            msgs, image_patch_size=self.vision.get("patch_size", 16),
            return_video_kwargs=True, return_video_metadata=True)
        kwargs = dict(kwargs or {})
        kwargs["do_resize"] = False
        return images, kwargs


def select_ids(records, a):
    ids = list(records)
    if a.ids_file:
        wanted = [l.strip() for l in Path(a.ids_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        ids = [i for i in wanted if i in records]
    if a.types:
        keep = set(a.types.split(","))
        ids = [i for i in ids if baseline_type(records[i]) in keep]
    if a.limit:
        ids = ids[:a.limit]
    return ids


async def run(a):
    import os
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # src/evaluate.py와 동일
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind

    settings, records = load_baseline(a.baseline)
    s, rt, model = settings["sampling"], settings["runtime"], settings["model"]
    vision = settings["vision"]
    ids = select_ids(records, a)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    todo = [i for i in ids if i not in done]
    print(f"대상 {len(ids)}개, 완료 {len(done)}개, 실행 {len(todo)}개")
    if not todo:
        return

    # 엔진 설정은 src/evaluate.py와 같게 둔다. 바꾸는 것은 max_num_seqs(동시 처리 수)와 seed뿐.
    engine_args = AsyncEngineArgs(
        model=a.model_path or model["repo"], revision=None if a.model_path else model["revision"],
        dtype=model.get("dtype", "bfloat16"),
        max_model_len=rt["max_model_len"], gpu_memory_utilization=a.gpu_mem,
        max_num_seqs=a.max_num_seqs, max_num_batched_tokens=a.max_num_batched_tokens,
        enforce_eager=not a.no_eager,
        limit_mm_per_prompt={"image": rt.get("max_images_per_prompt", 5)},
        seed=a.seed,
        mm_processor_kwargs={"size": {"shortest_edge": vision["min_pixels"],
                                      "longest_edge": vision["max_pixels"]},
                             "do_resize": False},
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    images = ImageSource(settings)
    sem = asyncio.Semaphore(a.max_num_seqs * 2)
    lock = asyncio.Lock()
    fout = open(out_path, "a", encoding="utf-8")
    t_start = time.time()
    stats = {"done": 0, "loop": 0, "length": 0, "tok_mismatch": 0}

    async def one(rid):
        rec = records[rid]
        async with sem:
            imgs, mm_kwargs = await asyncio.to_thread(images.images, rec)
            sp = SamplingParams(
                temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"],
                repetition_penalty=s["repetition_penalty"], presence_penalty=s["presence_penalty"],
                seed=a.seed, max_tokens=a.max_tokens, stop_token_ids=[],
                output_kind=RequestOutputKind.DELTA)
            prompt = {"prompt": rec["rendered_chat"], "multi_modal_data": {"image": imgs},
                      "mm_processor_kwargs": mm_kwargs}
            req_id = f"{rid}-{uuid.uuid4().hex[:8]}"
            parts, ntok, in_tok, finish = [], 0, None, None
            next_check, streak, loop_at = LOOP_WINDOW, 0, None
            t0 = time.time()
            async for out in engine.generate(prompt, sp, req_id):
                if in_tok is None and out.prompt_token_ids is not None:
                    in_tok = len(out.prompt_token_ids)
                o = out.outputs[0]
                parts.append(o.text)
                ntok += len(o.token_ids)
                if o.finish_reason is not None:
                    finish = o.finish_reason
                if a.loop_stop and ntok >= next_check and not out.finished:
                    text = "".join(parts)
                    cpt = len(text) / max(ntok, 1)
                    window = text[-int(LOOP_WINDOW * cpt):]
                    streak = streak + 1 if ngram_repeat_rate(window) >= a.loop_th else 0
                    next_check += LOOP_WINDOW
                    if streak >= a.loop_consec:
                        loop_at = ntok
                        await engine.abort(req_id)
                        break
                if out.finished:
                    break
            raw = "".join(parts)
            if loop_at is not None:
                status, finish = "loop_stopped", "abort"
            elif finish == "length":
                status = "output_token_limit"
            else:
                status = "normal_stop"
            row = {"id": rid, "seed": a.seed, "run_tag": a.tag or out_path.stem, "model_path": a.model_path,
                   "model_response": {
                       "raw": raw, "processed": raw.split("</think>")[-1],
                       "finish_reason": finish, "output_tokens": ntok, "input_tokens": in_tok,
                       "output_limit_reached": finish == "length", "generation_status": status,
                       "loop_stop_at_tokens": loop_at, "elapsed_seconds": time.time() - t0},
                   "baseline_input_tokens": rec["model_response"]["input_tokens"]}
            async with lock:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                stats["done"] += 1
                stats["loop"] += status == "loop_stopped"
                stats["length"] += status == "output_token_limit"
                ref_tok = rec["model_response"].get("input_tokens")
                stats["tok_mismatch"] += ref_tok is not None and in_tok != ref_tok
                if stats["done"] % 25 == 0 or stats["done"] == len(todo):
                    el = time.time() - t_start
                    print(f"[{stats['done']}/{len(todo)}] {el/60:.1f}분 경과, 루프중단 {stats['loop']}, "
                          f"길이초과 {stats['length']}, 입력토큰 불일치 {stats['tok_mismatch']}", flush=True)

    try:
        await asyncio.gather(*(one(r) for r in todo))
    finally:
        fout.close()
        try:
            engine.shutdown()
        except Exception:
            pass
    if stats["tok_mismatch"]:
        print("경고: 입력 토큰 수가 베이스라인과 다른 문항이 있다. 이미지 전처리(resize)가 달라졌을 수 있으니 "
              "본 실행 전에 원인을 확인할 것.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model-path", help="학습한 모델 폴더 (merge 저장본). 없으면 베이스라인 설정의 repo@revision")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=4096, help="공식 평가 값 4096")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--no-eager", action="store_true",
                    help="CUDA graph 사용 (빠르지만 메모리를 더 써서 64k 문맥 KV 캐시가 부족할 수 있음)")
    ap.add_argument("--loop-stop", action="store_true")
    ap.add_argument("--loop-th", type=float, default=LOOP_TH)
    ap.add_argument("--loop-consec", type=int, default=LOOP_CONSEC)
    ap.add_argument("--ids-file")
    ap.add_argument("--types", help="베이스라인 유형으로 제한, 예: T,C,I,V,OK")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
