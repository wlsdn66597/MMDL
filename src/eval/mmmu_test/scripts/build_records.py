"""평가할 문항 파일을 만든다. 문항은 Hugging Face 캐시(download.py로 받은 것)에서 읽는다.

문항 파일 = eval_settings.json의 설정 + 문항별 프롬프트·보기·정답·이미지 필드.
diag/run_generate.py, judge.py, grade.py가 --baseline으로 이 파일을 읽는다.

  python scripts/build_records.py mmmu-val    --out records/mmmu_val.json      # 900문항
  python scripts/build_records.py mmmu-test   --out records/mmmu_test.json     # 10,500문항
  python scripts/build_records.py pro-std10   --out records/pro_std10.json     # 1,730문항
  python scripts/build_records.py pro-vision  --out records/pro_vision.json    # 1,730문항

프롬프트
  MMMU val·test: 공식 베이스라인과 같다 ("Question: ... Options: ... Please select ..." + 마지막 줄 "Final answer: ..." 지시).
                 val은 공식 베이스라인 파일과 900문항 전부 글자 단위로 같은 것을 확인했다.
                 --check <baseline_results.json>을 주면 다시 비교한다.
  pro-std10:     MMMU와 같은 형식에 보기만 최대 10개.
  pro-vision:    문제와 보기가 찍힌 그림 한 장 + "Write out the multiple-choice question in the image and then solve it."
                 + MMMU와 같은 마지막 줄 지시. 채점기를 MMMU와 같게 쓰기 위한 우리 선택이며, Qwen 보고서의 프롬프트와 다르다.
"""
import argparse
import ast
import glob
import io
import json
import math
import os
import string
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRO_REPO, PRO_REV = "MMMU/MMMU_Pro", "563f3e84bb3b90893083a1f039cfa13077f2302b"
PRO_SETTING = {"pro-std10": "standard (10 options)", "pro-std4": "standard (4 options)", "pro-vision": "vision"}
VISION_TEXT = "Write out the multiple-choice question in the image and then solve it."


def snapshot(repo, rev):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=rev, repo_type="dataset", local_files_only=True)


def mmmu_prompt(row, prompts):
    letters = "".join(f"{l}. {v}\n" for l, v in zip(string.ascii_uppercase, row["options"])) \
        if row["question_type"] == "multiple-choice" else ""
    hint = row.get("hint")
    present = hint is not None and not (isinstance(hint, float) and math.isnan(hint))
    optional_hint = prompts["optional_hint"].format(hint=hint) if present else ""
    base = prompts[row["question_type"]].format(optional_hint=optional_hint, question=row["question"],
                                                 lettered_options=letters).rstrip()
    return base + "\n" + prompts["suffixes"][row["question_type"]]


def image_info(row, fields, vis, smart_resize, Image):
    sizes, tok = [], 0
    for f in fields:
        w, h = Image.open(io.BytesIO(row[f]["bytes"])).size
        sizes.append([w, h])
        rh, rw = smart_resize(h, w, factor=32, min_pixels=vis["min_pixels"], max_pixels=vis["max_pixels"])
        tok += rh * rw // (32 * 32) + 2
    return sizes, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=["mmmu-val", "mmmu-test", *PRO_SETTING])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=32768, help="문맥 초과 문항을 세는 데만 쓴다")
    ap.add_argument("--check", help="(mmmu-val) 공식 baseline_results.json과 문항을 비교")
    a = ap.parse_args()
    import pandas as pd
    from PIL import Image
    from qwen_vl_utils.vision_process import smart_resize
    from transformers import AutoProcessor

    settings = json.loads((ROOT / "eval_settings.json").read_text(encoding="utf-8"))["settings"]
    prompts, ds, vis, rt = settings["prompts"], settings["dataset"], settings["vision"], settings["runtime"]
    proc = AutoProcessor.from_pretrained(settings["model"]["repo"], revision=settings["model"]["revision"])

    def render(n_images, text):
        msgs = [{"role": "user", "content": [{"type": "image"} for _ in range(n_images)] + [{"type": "text", "text": text}]}]
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    records = []
    if a.bench.startswith("mmmu"):
        split = "validation" if a.bench == "mmmu-val" else "test"
        snap = snapshot(ds["repo"], ds["revision"])
        for subject in ds["configs"]:
            files = sorted(glob.glob(f"{snap}/{subject}/{split}-*.parquet"))
            assert files, f"parquet 없음: {subject} {split}. download.py mmmu를 먼저 실행"
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            for idx, row in df.iterrows():
                row = row.to_dict()
                row["options"] = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else list(row["options"])
                fields = [f"image_{i}" for i in range(1, 8)
                          if isinstance(row.get(f"image_{i}"), dict) and row[f"image_{i}"].get("bytes")]
                assert fields, f"이미지 없음: {row['id']}"
                sizes, img_tok = image_info(row, fields, vis, smart_resize, Image)
                text = mmmu_prompt(row, prompts)
                mc = row["question_type"] == "multiple-choice"
                records.append({"id": row["id"], "subject": subject, "dataset_config": subject, "dataset_index": int(idx),
                                "image_fields": fields, "original_image_sizes_wh": sizes,
                                "question_type": row["question_type"], "question": row["question"],
                                "options": dict(zip(string.ascii_uppercase, row["options"])) if mc else {},
                                "reference_answer": row["answer"], "input_prompt": text,
                                "rendered_chat": render(len(fields), text), "model_response": {"input_tokens": None},
                                "estimated_input_tokens": img_tok + len(proc.tokenizer(text)["input_ids"]) + 20})
            print(subject, len(df), flush=True)
        ds.update(split=split)
        name = f"MMMU {split}"
    else:
        setting = PRO_SETTING[a.bench]
        snap = snapshot(PRO_REPO, PRO_REV)
        files = sorted(glob.glob(f"{snap}/{setting}/test-*.parquet"))
        assert files, f"parquet 없음: {setting}. download.py pro를 먼저 실행"
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        vision = setting == "vision"
        for idx, row in df.iterrows():
            row = row.to_dict()
            opts = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else list(row["options"])
            letters = string.ascii_uppercase[:len(opts)]
            fields = ["image"] if vision else [f"image_{i}" for i in range(1, 8)
                                               if isinstance(row.get(f"image_{i}"), dict) and row[f"image_{i}"].get("bytes")]
            assert fields, row["id"]
            sizes, img_tok = image_info(row, fields, vis, smart_resize, Image)
            if vision:
                text = VISION_TEXT + "\n" + prompts["suffixes"]["multiple-choice"]
                question = "(The question and the options are shown in the image.)"
            else:
                lettered = "".join(f"{l}. {v}\n" for l, v in zip(letters, opts))
                text = prompts["multiple-choice"].format(optional_hint="", question=row["question"],
                                                         lettered_options=lettered).rstrip()
                text += "\n" + prompts["suffixes"]["multiple-choice"]
                question = row["question"]
            ans = str(row["answer"]).strip()
            assert ans in letters, (row["id"], ans, len(opts))
            records.append({"id": row["id"], "subject": row["id"].split("_", 1)[1].rsplit("_", 1)[0],
                            "dataset_config": setting, "dataset_index": int(idx), "image_fields": fields,
                            "original_image_sizes_wh": sizes, "question_type": "multiple-choice", "question": question,
                            "options": dict(zip(letters, opts)), "reference_answer": ans, "input_prompt": text,
                            "rendered_chat": render(len(fields), text), "model_response": {"input_tokens": None},
                            "estimated_input_tokens": img_tok + len(proc.tokenizer(text)["input_ids"]) + 20,
                            "source_split": row["id"].split("_", 1)[0]})
        ds.update(repo=PRO_REPO, revision=PRO_REV, split="test", configs=[setting])
        name = f"MMMU-Pro {setting}"
        print("출처 분할", dict(Counter(r["source_split"] for r in records)))

    n_img = Counter(len(r["image_fields"]) for r in records)
    too_long = [r["id"] for r in records if r["estimated_input_tokens"] + a.max_tokens > rt["max_model_len"]]
    print("문항", len(records), "유형", dict(Counter(r["question_type"] for r in records)), "이미지 수", dict(sorted(n_img.items())))
    print(f"입력+출력({a.max_tokens})이 문맥({rt['max_model_len']})을 넘는 문항 {len(too_long)}", too_long[:10])

    if a.check:
        ref = {r["id"]: r for r in json.loads(Path(a.check).read_text(encoding="utf-8"))["records"]}
        bad = Counter()
        for r in records:
            for k in ("subject", "dataset_index", "image_fields", "original_image_sizes_wh", "question_type", "question",
                      "options", "reference_answer", "input_prompt", "rendered_chat"):
                bad[k] += r[k] != ref[r["id"]][k]
        print("기준 파일과 다른 문항 수 (모두 0이어야 함):", dict(bad), "| 문항", len(records), "/", len(ref))

    ds["total_items"] = len(records)
    ds["question_type_counts"] = dict(Counter(r["question_type"] for r in records))
    ds.pop("items_per_config", None)
    rt["max_images_per_prompt"] = max(n_img)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"evaluation": {"name": f"{name} (문항만, 응답 없음)", "record_count": len(records),
                                              "settings": settings}, "records": records}, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print("저장:", out)


if __name__ == "__main__":
    main()
