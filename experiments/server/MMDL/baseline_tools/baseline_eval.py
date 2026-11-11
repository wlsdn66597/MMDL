#!/usr/bin/env python3
"""MMMU validation baseline client for a LOCAL vLLM server.

prepare: download validation parquet files only; freeze dataset + official scorer.
run: preflight all selected prompts, infer serially, and score saved responses.
No training, external model API, model-generated judge, or silent input truncation.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DATASET_ID = "MMMU/MMMU"
UPSTREAM = "https://github.com/MMMU-Benchmark/MMMU.git"
TOOL_VERSION = "1.0.0"
PROMPT_VERSION = "course_example_v1_png_interleaved"
IMAGE_RE = re.compile(r"<image\s+(\d+)>")
MC_INSTRUCTION = "Please select the correct answer from the options above."
OPEN_INSTRUCTION = "Answer the question using a single word or phrase."
SAMPLING = dict(max_tokens=2048, temperature=0.7, top_p=0.8, top_k=20,
                repetition_penalty=1.0, presence_penalty=1.5, seed=3407)
OFFICIAL_FILES = {"eval_utils.py": "mmmu/utils/eval_utils.py",
                  "data_utils.py": "mmmu/utils/data_utils.py", "LICENSE": "LICENSE"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def jd(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def append_json(path: Path, obj: Any) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(jd(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Malformed JSONL {path}:{n}. Back up and repair the last partial line; no automatic deletion.") from exc
    return result


def command_output(args: list[str], timeout: int = 20) -> str:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"UNAVAILABLE: {exc}"


def environment() -> dict:
    packages = {d.metadata['Name']: d.version for d in metadata.distributions()
                if d.metadata.get('Name')}
    return {"recorded_at": utc(), "python": sys.version, "executable": sys.executable,
            "packages": dict(sorted(packages.items())),
            "kernel": command_output(["uname", "-a"]),
            "cpu": command_output(["lscpu"]), "memory": command_output(["free", "-h"]),
            "disk": command_output(["df", "-hT"]),
            "gpu": command_output(["nvidia-smi"]),
            "nvcc": command_output([str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin/nvcc"), "--version"]),
            "selected_environment": {k: os.environ.get(k) for k in
              ["CUDA_HOME", "HF_HOME", "VLLM_USE_FLASHINFER_SAMPLER", "OMP_NUM_THREADS"]}}


def download_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "MMDL-baseline/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def freeze_official(path: Path, revision: str | None) -> dict:
    """Fetch unmodified upstream files at ONE frozen Git commit."""
    target = path / "official_mmmu"
    lock_path = target / "source.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if revision and revision != lock["commit"]:
            raise RuntimeError("Scorer revision changed. Use a new --data-path.")
        for name, digest in lock["sha256"].items():
            if sha_file(target / name) != digest:
                raise RuntimeError(f"Official scorer integrity mismatch: {name}")
        return lock
    if revision is None:
        p = subprocess.run(["git", "ls-remote", UPSTREAM, "HEAD"],
                           check=True, capture_output=True, text=True, timeout=120)
        revision = p.stdout.split()[0]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--parser-revision must be a full 40-character Git commit SHA.")
    target.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, remote in OFFICIAL_FILES.items():
        url = f"https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/{revision}/{remote}"
        payload = download_bytes(url)
        if name.endswith(".py"):
            ast.parse(payload.decode("utf-8"))
        (target / name).write_bytes(payload)
        checksums[name] = sha_bytes(payload)
    lock = {"repository": UPSTREAM, "commit": revision, "sha256": checksums,
            "retrieved_at": utc(), "modified": False}
    write_json(lock_path, lock)
    return lock


def domain_map(path: Path) -> dict[str, list[str]]:
    # Read ONLY the literal mapping. Do not call upstream's legacy data processor.
    tree = ast.parse((path / "official_mmmu/data_utils.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
           t.id == "DOMAIN_CAT2SUB_CAT" for t in node.targets):
            result = ast.literal_eval(node.value)
            if len(result) != 6 or sum(map(len, result.values())) != 30:
                raise RuntimeError("Unexpected official domain/subject mapping.")
            return result
    raise RuntimeError("Official subject mapping was not found.")


def parse_options(value: Any, question_type: str) -> list[str]:
    if question_type != "multiple-choice":
        return []
    options = ast.literal_eval(value) if isinstance(value, str) else value
    if not isinstance(options, list) or not 2 <= len(options) <= 26 or not all(isinstance(x, str) for x in options):
        raise ValueError(f"Invalid multiple-choice options: {value!r}")
    return options


def prompt_text(sample: dict) -> str:
    question = sample["question"]
    if sample["question_type"] == "multiple-choice":
        choices = "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(sample["options"]))
        return f"Question: {question}\nOptions:\n{choices}\n{MC_INSTRUCTION}"
    return f"Question: {question}\n{OPEN_INSTRUCTION}"


def content_plan(sample: dict) -> list[dict]:
    """Keep image positions in BOTH the question and options. Never drop images."""
    text = prompt_text(sample)
    if "<img=" in text:
        raise ValueError(f"Unsupported legacy image markup in {sample['id']}; inspect instead of dropping it.")
    parts, cursor, seen = [], 0, set()
    for match in IMAGE_RE.finditer(text):
        if match.start() > cursor:
            parts.append({"type": "text", "text": text[cursor:match.start()]})
        key = f"image_{int(match.group(1))}"
        if key not in sample["images"]:
            raise ValueError(f"{sample['id']}: missing referenced image {key}")
        parts.append({"type": "local_image", "image_key": key})
        seen.add(key)
        cursor = match.end()
    if cursor < len(text):
        parts.append({"type": "text", "text": text[cursor:]})
    # Retain all populated image fields even when a source lacks a placeholder.
    for key in sorted(set(sample["images"]) - seen, key=lambda s: int(s.split('_')[1])):
        parts.append({"type": "text", "text": f"\n[Additional image {key.split('_')[1]}]\n"})
        parts.append({"type": "local_image", "image_key": key})
    return parts


def messages(sample: dict, root: Path) -> list[dict]:
    content = []
    for part in content_plan(sample):
        if part["type"] == "text":
            content.append(part)
            continue
        info = sample["images"][part["image_key"]]
        p = (root / info["path"]).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError("Image path escapes --data-path.")
        raw = p.read_bytes()
        if sha_bytes(raw) != info["sha256"]:
            raise RuntimeError(f"Image checksum mismatch: {p}")
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii")}})
    # Only image/question/options are used. The gold answer and explanation NEVER enter the request.
    return [{"role": "user", "content": content}]


def audit_samples(samples: list[dict]) -> dict:
    types, images, requests_count = Counter(), Counter(), Counter()
    orphan_ids = []
    for s in samples:
        parts = content_plan(s)
        types[s["question_type"]] += 1
        images[len(s["images"])] += 1
        requests_count[sum(p["type"] == "local_image" for p in parts)] += 1
        referenced = {f"image_{int(n)}" for n in IMAGE_RE.findall(prompt_text(s))}
        if set(s["images"]) - referenced:
            orphan_ids.append(s["id"])
    return {"samples": len(samples), "question_types": dict(types),
            "images_per_sample": dict(sorted(images.items())),
            "image_items_per_request": dict(sorted(requests_count.items())),
            "max_request_images": max(requests_count, default=0),
            "samples_with_unreferenced_images": orphan_ids}


def prepare(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from PIL import Image
    root = Path(args.data_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        manifest, samples = load_prepared(root)
        if args.dataset_revision and args.dataset_revision != manifest["dataset_revision"]:
            raise RuntimeError("Dataset revision changed. Use a new data path.")
        if args.parser_revision and args.parser_revision != manifest["official_scorer"]["commit"]:
            raise RuntimeError("Parser revision changed. Use a new data path.")
        print(f"Already prepared and metadata verified: {len(samples)} samples in {root}")
        print(json.dumps(manifest["audit"], indent=2))
        return
    official = freeze_official(root, args.parser_revision)
    mapping = domain_map(root)
    lookup = {s: d for d, subjects in mapping.items() for s in subjects}
    api = HfApi()
    revision_file = root / "dataset_revision.txt"
    if revision_file.exists():
        revision = revision_file.read_text().strip()
        if args.dataset_revision and revision != args.dataset_revision:
            raise RuntimeError("Existing partial download has another revision; choose a new data path.")
    else:
        revision = api.dataset_info(DATASET_ID, revision=args.dataset_revision or "main").sha
        revision_file.write_text(revision + "\n")
    files = api.list_repo_files(DATASET_ID, repo_type="dataset", revision=revision)
    samples, source_files = [], []
    (root / "images").mkdir(exist_ok=True)
    for index, subject in enumerate(sorted(lookup), 1):
        validation_files = sorted(f for f in files if
            re.fullmatch(re.escape(subject) + r"/validation-\d+-of-\d+\.parquet", f))
        if not validation_files:
            raise RuntimeError(f"No validation parquet for {subject}. Dataset layout may have changed.")
        local_files = []
        for name in validation_files:
            downloaded = hf_hub_download(DATASET_ID, name, repo_type="dataset", revision=revision)
            local_files.append(downloaded)
            source_files.append({"filename": name, "sha256": sha_file(Path(downloaded))})
        ds = load_dataset("parquet", data_files={"validation": local_files}, split="validation")
        if len(ds) != 30:
            raise RuntimeError(f"Expected 30 validation samples for {subject}, got {len(ds)}.")
        for row in ds:
            sid = row["id"]
            if not re.fullmatch(r"validation_[A-Za-z_]+_\d+", sid):
                raise ValueError(f"Unexpected validation ID: {sid}")
            qt = row["question_type"]
            if qt not in ("multiple-choice", "open"):
                raise ValueError(f"Unexpected question type: {qt!r}")
            s = {"id": sid, "subject": subject, "category": lookup[subject],
                 "question_type": qt, "question": row["question"],
                 "options": parse_options(row["options"], qt), "answer": row["answer"],
                 "subfield": row.get("subfield"), "difficulty": row.get("topic_difficulty"),
                 "image_type": row.get("img_type"), "images": {}}
            if not s["answer"] or s["answer"] == "?":
                raise ValueError(f"Missing validation label: {sid}")
            for i in range(1, 8):
                key, image = f"image_{i}", row.get(f"image_{i}")
                if image is None:
                    continue
                if isinstance(image, dict):
                    image = Image.open(io.BytesIO(image["bytes"])) if image.get("bytes") else Image.open(image["path"])
                image = image.convert("RGB")
                dest = root / "images" / f"{sid}_{key}.png"
                image.save(dest, format="PNG")  # no resize, no new lossy JPEG compression
                s["images"][key] = {"path": str(dest.relative_to(root)),
                                     "sha256": sha_file(dest), "width": image.width, "height": image.height}
            content_plan(s)  # validate before considering preparation successful
            samples.append(s)
        print(f"[{index}/30] prepared {subject}: {len(ds)}", flush=True)
    samples.sort(key=lambda s: (s["subject"], int(s["id"].rsplit('_', 1)[1])))
    if len(samples) != 900 or len({s["id"] for s in samples}) != 900:
        raise RuntimeError("Expected exactly 900 unique validation samples.")
    data_file = root / "samples.jsonl"
    tmp = root / "samples.jsonl.tmp"
    tmp.write_text("".join(jd(s) + "\n" for s in samples), encoding="utf-8")
    tmp.replace(data_file)
    manifest = {"schema": 1, "tool_version": TOOL_VERSION, "prepared_at": utc(),
                "dataset": DATASET_ID, "dataset_revision": revision, "split": "validation",
                "samples_sha256": sha_file(data_file), "source_parquet_files": source_files,
                "official_scorer": official, "image_encoding": "PIL RGB to lossless PNG; no resizing",
                "audit": audit_samples(samples)}
    write_json(root / "manifest.json", manifest)
    print(json.dumps(manifest["audit"], indent=2))
    print(f"Prepared data: {root}")


def load_prepared(root: Path) -> tuple[dict, list[dict]]:
    m = json.loads((root / "manifest.json").read_text())
    if sha_file(root / "samples.jsonl") != m["samples_sha256"]:
        raise RuntimeError("Dataset JSONL checksum mismatch.")
    for name, digest in m["official_scorer"]["sha256"].items():
        if sha_file(root / "official_mmmu" / name) != digest:
            raise RuntimeError(f"Official file changed: {name}")
    samples = read_jsonl(root / "samples.jsonl")
    if len(samples) != 900 or len({s["id"] for s in samples}) != 900:
        raise RuntimeError("Prepared dataset must have 900 unique samples.")
    return m, samples


def choose_samples(samples: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.subject:
        selected = [s for s in samples if s["subject"] in args.subject]
        if set(args.subject) - {s["subject"] for s in selected}:
            raise ValueError("Unknown subject name. Use the exact official subject name.")
    else:
        selected = list(samples)
    if args.smoke:
        # Deterministic coverage test: maximum image count, longest text, open QA, all domains.
        ranked = [max(selected, key=lambda s: sum(p["type"] == "local_image" for p in content_plan(s))),
                  max(selected, key=lambda s: len(prompt_text(s)))]
        ranked += [s for s in selected if s["question_type"] == "open"][:1]
        for domain in sorted({s["category"] for s in selected}):
            ranked.append(next(s for s in selected if s["category"] == domain))
        ranked += selected
        seen, unique = set(), []
        for s in ranked:
            if s["id"] not in seen:
                seen.add(s["id"])
                unique.append(s)
        selected = unique[:args.smoke]
    if args.limit:
        selected = selected[:args.limit]
    return selected


class LocalAPI:
    def __init__(self, base_url: str, timeout: float):
        u = urllib.parse.urlsplit(base_url)
        if u.scheme != "http" or u.hostname not in ("127.0.0.1", "localhost", "::1") or u.username or u.password:
            raise ValueError("This client intentionally accepts LOCAL http loopback endpoints only.")
        self.base = base_url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        self.timeout = timeout
        # Do not send local inference through HTTP_PROXY/HTTPS_PROXY.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, route: str, payload: dict | None = None) -> dict:
        body = None if payload is None else jd(payload).encode("utf-8")
        req = urllib.request.Request(self.base + route, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:5000]
            raise RuntimeError(f"HTTP {exc.code} {route}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            raise RuntimeError(f"Local server request failed at {route}: {exc}") from exc
        result = json.loads(raw) if raw else {}
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(f"Server error: {result['error']}")
        return result


def import_scorer(root: Path):
    module_path = root / "official_mmmu/eval_utils.py"
    spec = importlib.util.spec_from_file_location("mmdl_official_mmmu_eval", module_path)
    module = importlib.util.module_from_spec(spec)
    state = random.getstate()
    try:
        spec.loader.exec_module(module)
    finally:
        random.setstate(state)
    return module


def score_rows(samples: list[dict], latest: dict, scorer) -> list[dict]:
    """Call official functions unmodified. Track official random-choice fallback."""
    rows = []
    state = random.getstate()
    random.seed(42)  # upstream's default; fixed canonical order makes final rescoring stable
    try:
        for s in sorted(samples, key=lambda x: (x["subject"], int(x["id"].rsplit('_', 1)[1]))):
            r = latest.get(s["id"], {})
            ok = r.get("status") == "ok"
            parsed, correct, fallback = None, False, False
            if ok:
                if s["question_type"] == "multiple-choice":
                    index2ans = {chr(65+i): opt for i, opt in enumerate(s["options"])}
                    before = random.getstate()
                    parsed = scorer.parse_multi_choice_response(r["raw_response"], list(index2ans), index2ans)
                    fallback = random.getstate() != before
                    correct = bool(scorer.eval_multi_choice(s["answer"], parsed))
                else:
                    parsed = scorer.parse_open_response(r["raw_response"])
                    correct = bool(scorer.eval_open(s["answer"], parsed))
                    parsed = sorted(parsed, key=lambda x: (type(x).__name__, str(x)))
            rows.append({"id": s["id"], "subject": s["subject"], "category": s["category"],
                         "question_type": s["question_type"], "status": r.get("status", "pending"),
                         "question": s["question"], "options": s["options"], "answer": s["answer"],
                         "raw_response": r.get("raw_response"), "parsed_prediction": parsed,
                         "correct": correct, "parser_random_fallback": fallback,
                         "finish_reason": r.get("finish_reason"),
                         "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                         "completion_tokens": r.get("usage", {}).get("completion_tokens"),
                         "api_elapsed_sec": r.get("api_elapsed_sec", 0.0),
                         "sample_elapsed_sec": r.get("sample_elapsed_sec", 0.0),
                         "image_count": len(s["images"]), "error": r.get("error")})
    finally:
        random.setstate(state)
    return rows


def stats(rows: list[dict]) -> dict:
    successes = [r for r in rows if r["status"] == "ok"]
    total, n = len(rows), len(successes)
    correct = sum(r["correct"] for r in successes)
    return {"expected": total, "completed": n, "errors": sum(r["status"] == "error" for r in rows),
            "pending": total - n - sum(r["status"] == "error" for r in rows),
            "correct": correct,
            "accuracy": correct / total if total and n == total else None,
            "partial_accuracy_on_completed": correct / n if n else None,
            "parser_random_fallbacks": sum(r["parser_random_fallback"] for r in successes),
            "accuracy_without_random_credit": sum(r["correct"] and not r["parser_random_fallback"] for r in successes) / total if total and n == total else None,
            "length_stops": sum(r["finish_reason"] == "length" for r in successes),
            "sum_api_elapsed_sec": sum(r["api_elapsed_sec"] for r in successes),
            "sum_sample_elapsed_sec": sum(r["sample_elapsed_sec"] for r in successes)}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: jd(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def summarize(out: Path, samples: list[dict], latest: dict, scorer) -> dict:
    rows = score_rows(samples, latest, scorer)
    overall = stats(rows)
    summary = {"run_complete": overall["completed"] == len(samples),
               "full_validation_complete": len(samples) == 900 and overall["completed"] == 900,
               "overall": overall, "sampling": SAMPLING, "prompt_version": PROMPT_VERSION,
               "scoring": "Unmodified official MMMU functions; random.seed(42) in canonical subject/numeric-ID order",
               "quality_caveat": "Answer accuracy does not establish explanation correctness.",
               "timing_note": "Serial client wall times; model load/download/preflight excluded from per-sample inference times."}
    for group, name in [("subject", "subject_results.csv"), ("category", "category_results.csv"),
                        ("question_type", "question_type_results.csv")]:
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[group]].append(row)
        result = [{group: key, **stats(value)} for key, value in sorted(grouped.items())]
        write_csv(out / name, result)
    write_csv(out / "sample_results.csv", rows)
    write_csv(out / "failures.csv", [r for r in rows if r["status"] == "error"])
    write_json(out / "summary.json", summary)
    return summary


def preflight(api: LocalAPI, samples: list[dict], root: Path, out: Path, settings: dict) -> dict:
    started = time.perf_counter()
    records, errors = [], []
    for i, s in enumerate(samples, 1):
        try:
            result = api.request("/tokenize", {"model": MODEL_ID, "messages": messages(s, root),
                                             "add_generation_prompt": True})
            count, model_len = int(result["count"]), int(result["max_model_len"])
            if model_len != settings["max_model_len"]:
                raise RuntimeError("Running server max_model_len differs from recorded settings.")
            valid = count + SAMPLING["max_tokens"] <= model_len
            record = {"id": s["id"], "prompt_tokens": count, "max_tokens": SAMPLING["max_tokens"],
                      "required_context": count + SAMPLING["max_tokens"], "ok": valid}
            if not valid:
                errors.append(f"{s['id']}: needs {record['required_context']} tokens > {model_len}")
        except Exception as exc:
            record = {"id": s["id"], "ok": False, "error": str(exc)}
            errors.append(f"{s['id']}: {exc}")
            if i == 1:
                records.append(record)
                break  # common server/template failures should not cause 900 repeated calls
        records.append(record)
        if i % 30 == 0 or i == len(samples):
            print(f"Preflight [{i}/{len(samples)}]", flush=True)
    report = {"selected": len(samples), "checked": len(records), "errors": errors,
              "max_required_context": max((r.get("required_context", 0) for r in records), default=0),
              "elapsed_sec": time.perf_counter() - started,
              "passed": len(records) == len(samples) and not errors, "samples": records}
    write_json(out / "preflight.json", report)
    if errors:
        raise RuntimeError("Preflight failed; NO inference started and NO samples omitted. See preflight.json.\n" + "\n".join(errors[:5]))
    return report


def run(args: argparse.Namespace) -> None:
    root, out = Path(args.data_path).expanduser().resolve(), Path(args.out).expanduser().resolve()
    manifest, all_samples = load_prepared(root)
    samples = choose_samples(all_samples, args)
    if not samples:
        raise ValueError("No samples selected.")
    server = json.loads(Path(args.server_config).expanduser().read_text())
    settings = server["settings"]
    if settings.get("load_format") == "dummy":
        raise ValueError("Refusing to evaluate dummy weights.")
    if settings.get("seed") != SAMPLING["seed"]:
        raise ValueError("Use the supplied launcher to set the engine seed to 3407.")
    api = LocalAPI(args.base_url, args.timeout)
    api.request("/health")
    model_list = api.request("/v1/models")["data"]
    live = next((m for m in model_list if m["id"] == MODEL_ID), None)
    if live is None or str(Path(live["root"]).resolve()) != server["model_path"]:
        raise RuntimeError("Served model root/name does not match server record. Stop the other server and use start_server.sh.")
    if int(live["max_model_len"]) != settings["max_model_len"]:
        raise RuntimeError("Served context length differs from server record.")
    identity = {"tool_version": TOOL_VERSION, "script_sha256": sha_file(Path(__file__)),
                "data_manifest_sha256": sha_file(root / "manifest.json"),
                "ids": [s["id"] for s in samples], "sampling": SAMPLING,
                "prompt_version": PROMPT_VERSION, "server_settings": settings,
                "model_hashes": server["model_hashes"], "server_vllm_version": server["vllm_version"],
                "torch_version": server["torch_version"], "transformers_version": server["transformers_version"],
                "sampler_override": server.get("sampler_override"),
                "parser_seed": 42, "parser_order": "subject then numeric ID",
                "client_concurrency": 1}
    out.mkdir(parents=True, exist_ok=True)
    config_file = out / "config.json"
    if config_file.exists():
        old = json.loads(config_file.read_text())
        if not args.resume:
            raise RuntimeError("Output already exists. Use --resume or a new --out. Nothing overwritten.")
        if old["identity"] != identity:
            raise RuntimeError("Run settings changed. Do not mix results; use a NEW --out directory.")
    else:
        if any(out.iterdir()):
            raise RuntimeError("Output directory is nonempty without a config. Choose a new --out.")
        write_json(config_file, {"created_at": utc(), "identity": identity,
                                 "data_manifest": manifest, "server_record": server})
        write_json(out / "environment.json", environment())
    scorer = import_scorer(root)
    attempts_file = out / "predictions.jsonl"
    latest = {r["id"]: r for r in read_jsonl(attempts_file)}
    session_start, wall_start = utc(), time.perf_counter()
    loop_sec, prep_sec = 0.0, 0.0
    stop_error = None
    try:
        pre = preflight(api, samples, root, out, settings)
        prep_sec = pre["elapsed_sec"]
        if args.preflight_only:
            print(f"Preflight passed: {len(samples)} samples; required context <= {pre['max_required_context']}.")
            return
        loop_start = time.perf_counter()
        try:
            for i, sample in enumerate(samples, 1):
                if latest.get(sample["id"], {}).get("status") == "ok":
                    continue
                t = time.perf_counter()
                record = {"id": sample["id"], "timestamp": utc(),
                          "prompt_text": prompt_text(sample), "content_plan": content_plan(sample)}
                try:
                    payload = {"model": MODEL_ID, "messages": messages(sample, root), **SAMPLING}
                    before_api = time.perf_counter()
                    response = api.request("/v1/chat/completions", payload)
                    record["api_elapsed_sec"] = time.perf_counter() - before_api
                    choice = response["choices"][0]
                    raw = choice["message"].get("content")
                    if not isinstance(raw, str) or not raw.strip():
                        record["response"] = response
                        raise RuntimeError("Empty model response. Not scored as a random guessed answer.")
                    record.update(status="ok", raw_response=raw, response=response,
                                  finish_reason=choice.get("finish_reason"), usage=response.get("usage", {}))
                except Exception as exc:
                    record.update(status="error", error=str(exc))
                    stop_error = exc
                record["sample_elapsed_sec"] = time.perf_counter() - t
                append_json(attempts_file, record)
                latest[sample["id"]] = record
                print(f"[{i}/{len(samples)}] {sample['id']} {record['status']} "
                      f"{record['sample_elapsed_sec']:.2f}s finish={record.get('finish_reason')}", flush=True)
                if i % 30 == 0 or stop_error or i == len(samples):
                    summarize(out, samples, latest, scorer)
                if stop_error:
                    raise RuntimeError(f"Stopped at {sample['id']}: {stop_error}. Fix the cause, then --resume with unchanged settings.") from stop_error
        finally:
            loop_sec = time.perf_counter() - loop_start
    finally:
        summary = summarize(out, samples, latest, scorer)
        append_json(out / "timing_sessions.jsonl", {"started_at": session_start, "ended_at": utc(),
                    "session_wall_sec": time.perf_counter() - wall_start,
                    "preflight_sec": prep_sec, "inference_loop_wall_sec": loop_sec,
                    "completed_total": summary["overall"]["completed"]})
        timing_sessions = read_jsonl(out / "timing_sessions.jsonl")
        write_json(out / "timing.json", {"sessions": timing_sessions,
            "total_session_wall_sec": sum(x["session_wall_sec"] for x in timing_sessions),
            "total_inference_loop_wall_sec": sum(x["inference_loop_wall_sec"] for x in timing_sessions),
            "note": "Offline gaps between resumed sessions, dataset download, and model startup are excluded."})
        print(json.dumps(summary["overall"], indent=2))
        print(f"Saved: {out}")


def server_record(args: argparse.Namespace) -> None:
    root = Path(args.data_path).expanduser().resolve()
    manifest, _ = load_prepared(root)
    model = Path(args.model_path).expanduser().resolve()
    index_file = model / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())
    weight_files = sorted(set(index["weight_map"].values()))
    tracked = sorted(set(weight_files + [p.name for p in model.iterdir()
                                        if p.is_file() and p.suffix in (".json", ".jinja", ".txt")]))
    print("Fingerprinting model files for experiment provenance (not loading the GPU)...", flush=True)
    hashes = {}
    for name in tracked:
        p = model / name
        if not p.resolve().is_relative_to(model):
            raise RuntimeError("Model weight path outside model directory.")
        hashes[name] = {"sha256": sha_file(p), "bytes": p.stat().st_size}
    settings = {"dtype": "bfloat16", "max_model_len": args.max_model_len,
                "gpu_memory_utilization": 0.90, "max_num_seqs": 1,
                "limit_mm_per_prompt": {"image": max(1, manifest["audit"]["max_request_images"]), "video": 0},
                "min_pixels": 1003520, "max_pixels": 4014080, "mm_processor_cache_gb": 0,
                "safetensors_load_strategy": "eager", "enforce_eager": True, "seed": 3407,
                "load_format": "auto", "generation_config": "vllm"}
    record = {"created_at": utc(), "model_path": str(model), "served_model_name": MODEL_ID,
              "settings": settings, "model_hashes": hashes,
              "vllm_version": metadata.version("vllm"),
              "torch_version": metadata.version("torch"),
              "transformers_version": metadata.version("transformers"),
              "sampler_override": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
              "cuda_home": os.environ.get("CUDA_HOME")}
    write_json(Path(args.out).expanduser(), record)
    print(f"Server settings recorded in {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Download validation only; record revisions and image hashes")
    p.add_argument("--data-path", required=True)
    p.add_argument("--dataset-revision")
    p.add_argument("--parser-revision")
    p.set_defaults(func=prepare)
    p = sub.add_parser("run", help="Preflight, infer sequentially, save raw outputs and official scores")
    p.add_argument("--data-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--server-config", default="logs/baseline_server.json")
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--subject", nargs="+")
    p.add_argument("--smoke", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    p.set_defaults(func=run)
    p = sub.add_parser("server-record", help="Internal: snapshot model hashes and launch settings")
    p.add_argument("--data-path", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--max-model-len", type=int, default=9048)
    p.add_argument("--out", required=True)
    p.set_defaults(func=server_record)
    args = parser.parse_args()
    if getattr(args, "smoke", 0) < 0 or getattr(args, "limit", 0) < 0:
        parser.error("--smoke/--limit must be nonnegative.")
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed rows remain saved. Resume only with unchanged settings.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
