"""Validate the frozen evaluator, checkpoint, resume identity and completed ID set.

This wrapper does not alter prompts, generation, parsing, judge or scoring.
"""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
UNSET = object()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def reference():
    return read_json(ROOT / "protocol_reference.json")


def source_check():
    for name, expected in reference()["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen evaluator changed: {name}")


def records_check(path):
    expected = reference()
    if sha(path) != expected["records_sha256"]:
        raise ValueError("Test records SHA256 differs from the AIX-408-06 reference. Do not evaluate until resolved.")
    data = read_json(path)
    rows = data["records"]
    ids = {r["id"] for r in rows}
    if len(rows) != expected["total_items"] or len(ids) != len(rows):
        raise ValueError("Missing or duplicate Test records")
    if collections.Counter(r["question_type"] for r in rows) != expected["question_type_counts"]:
        raise ValueError("Question type counts differ")
    settings = data["evaluation"]["settings"]
    if settings["dataset"]["split"] != "test" or settings["runtime"]["max_images_per_prompt"] != 7:
        raise ValueError("Expected Test split and up to seven images")
    return data


def runtime_info():
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Python 3.12 is required")
    versions = {}
    for line in (ROOT / "env/requirements_eval.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        actual = importlib.metadata.version(name)
        if actual.split("+")[0] != expected:
            raise ValueError(f"Version mismatch: {name}={actual}, expected {expected}")
        versions[name] = actual
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("A BF16-capable NVIDIA GPU is required")
    gpu = torch.cuda.get_device_properties(0)
    info = {"python": sys.version, "packages": versions, "gpu": gpu.name,
            "gpu_bytes": gpu.total_memory, "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True)
    info["nvidia_smi"] = result.stdout.strip()
    return info


def model_fingerprint(path):
    path = Path(path).resolve(strict=True)
    if ":" in str(path) or "\n" in str(path):
        raise ValueError("Use a Linux checkpoint path without ':' or newlines")
    config = read_json(path / "config.json")
    if config.get("model_type") != "qwen3_vl" or config.get("quantization_config"):
        raise ValueError("Expected an unquantized, merged Qwen3-VL checkpoint")
    if (path / "adapter_config.json").exists():
        raise ValueError("Use merged weights (including ERA/DeepStack gates), not a LoRA-only directory")
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        if not (path / name).is_file():
            raise ValueError(f"Missing model/processor file: {name}")
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        shards = set(read_json(index_path)["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    if not shards:
        raise ValueError("No model weight shards")
    for name in shards:
        if not isinstance(name, str) or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("Weight shards must be filenames within the model directory")
        file = (path / name).resolve(strict=True)
        # Official HF snapshots link flat shard names to the repository's blob cache.
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Invalid weight shard: {name}")
    # Hash weights and tokenizer/processor metadata.
    files = sorted(set(shards) | {p.name for p in path.iterdir()
                                 if p.is_file() and p.suffix in {".json", ".txt", ".jinja"}})
    print("Hashing model weights and processor files ...", flush=True)
    return {"path": str(path), "files": {name: sha(path / name) for name in files}}


def validate_tag(tag):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", tag):
        raise ValueError("Tag must be 1-100 letters/digits/underscores/hyphens, starting with a letter/digit")


def validate_seed(seed):
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 4294967295")


def base_model_path():
    from huggingface_hub import snapshot_download
    cfg = read_json(ROOT / "eval_settings.json")["settings"]["model"]
    return str(Path(snapshot_download(cfg["repo"], revision=cfg["revision"], local_files_only=True)).resolve())


def read_rows(path, allowed, seed=None, model=UNSET, complete=False):
    rows = {}
    if not path.exists():
        if complete:
            raise ValueError(f"Missing result file: {path}")
        return rows
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid/truncated JSON at {path}:{number}; preserve and repair before resuming") from error
            rid = row["id"]
            if rid not in allowed or rid in rows:
                raise ValueError(f"Unexpected or duplicate ID: {rid} in {path}")
            if seed is not None and row.get("seed") != seed:
                raise ValueError(f"Mixed generation seeds in {path}")
            if model is not UNSET and ("model_path" not in row or row["model_path"] != model):
                raise ValueError(f"Mixed model paths in {path}")
            rows[rid] = row
    if complete and set(rows) != allowed:
        raise ValueError(f"Incomplete result file {path}: {len(rows)}/{len(allowed)} unique IDs")
    return rows


def begin(model, tag, seed=42):
    validate_tag(tag)
    validate_seed(seed)
    source_check()
    data = records_check(ROOT / "records/mmmu_test.json")
    runtime = runtime_info()
    fingerprint = model_fingerprint(model)
    frozen = reference()
    profile = frozen.get("profile", "dpo_e")
    if profile not in {"baseline", "dpo_e"}:
        raise ValueError("Unknown evaluation profile")
    if profile == "baseline" and fingerprint["path"] != base_model_path():
        raise ValueError("Baseline must use the pinned official checkpoint cache")
    prediction_model = None if profile == "baseline" else fingerprint["path"]
    effective = dict(frozen["evaluation"], generation_seed=seed)
    identity = {"model": fingerprint, "records_sha256": frozen["records_sha256"],
                "source_sha256": frozen["source_sha256"], "evaluation": effective, "tag": tag,
                "profile": profile, "prediction_model_path": prediction_model}
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    manifest = out / f"{tag}.manifest.json"
    if manifest.exists():
        previous = read_json(manifest)
        if previous["identity"] != identity:
            raise ValueError("This tag belongs to a different checkpoint/protocol. Use a new tag.")
        if previous["runtime"] != runtime:
            raise ValueError("Runtime changed during resume. Restore the same environment/GPU.")
    else:
        if any(out.glob(f"{tag}.*")):
            raise ValueError("Existing outputs have no identity manifest. Use a new tag; do not mix old responses.")
        write_json(manifest, {"identity": identity, "runtime": runtime,
                              "record_settings": data["evaluation"]["settings"],
                              "note": f"CLI overrides: generation seed {seed}, seqs 16, max tokens 32768, loop-stop; judge seed 3407 unchanged."})
    ids = {r["id"] for r in data["records"]}
    predictions = read_rows(out / f"{tag}.jsonl", ids, seed=seed, model=prediction_model)
    if any(row.get("run_tag") != tag for row in predictions.values()):
        raise ValueError("Mixed run tags in predictions")
    graded = out / f"{tag}.graded.jsonl"
    if graded.exists() and graded.stat().st_size:
        # The archived launcher skips a nonempty graded file; refuse a partial one.
        read_rows(graded, ids, seed=seed, complete=True)
        read_rows(out / f"{tag}.jsonl", ids, seed=seed, model=prediction_model, complete=True)
    print(f"Protocol verified: profile={profile}, model={fingerprint['path']}, generation seed={seed}, judge seed=3407, saved={len(predictions)}/10500")


def verify(tag):
    validate_tag(tag)
    source_check()
    data = records_check(ROOT / "records/mmmu_test.json")
    ids = {r["id"] for r in data["records"]}
    out = ROOT / "outputs"
    manifest = read_json(out / f"{tag}.manifest.json")
    identity = manifest["identity"]
    frozen = reference()
    seed = identity["evaluation"]["generation_seed"]
    validate_seed(seed)
    if (identity["records_sha256"] != frozen["records_sha256"] or
            identity["source_sha256"] != frozen["source_sha256"] or
            identity["evaluation"] != dict(frozen["evaluation"], generation_seed=seed) or
            identity["profile"] != frozen.get("profile", "dpo_e") or identity["tag"] != tag):
        raise ValueError("Saved identity does not match this evaluator/profile")
    predictions = read_rows(out / f"{tag}.jsonl", ids, seed=seed,
                            model=identity["prediction_model_path"], complete=True)
    if any(row.get("run_tag") != tag for row in predictions.values()):
        raise ValueError("Mixed run tags in predictions")
    graded = read_rows(out / f"{tag}.graded.jsonl", ids, seed=seed, complete=True)
    judges = read_rows(out / f"{tag}.judge.jsonl", ids)
    sys.path.insert(0, str(ROOT / "scripts/diag"))
    from common import grade_prediction
    needed = set()
    for rec in data["records"]:
        rid = rec["id"]
        if grade_prediction(predictions[rid], rec)["verdict"] == "needs_judge":
            needed.add(rid)
        actual = graded[rid]
        expected = grade_prediction(predictions[rid], rec, judges.get(rid))
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Regrade mismatch: {rid}")
        for key in ("output_tokens", "generation_status"):
            if actual.get(key) != predictions[rid]["model_response"].get(key):
                raise ValueError(f"Grading metadata mismatch: {rid} / {key}")
        if actual["verdict"] not in {"correct", "incorrect"}:
            raise ValueError(f"Unresolved grading: {rid}")
    if needed != set(judges):
        raise ValueError("Judge output IDs must match exactly the responses requiring judge")
    correct = sum(r["verdict"] == "correct" for r in graded.values())
    result = {"unique_predictions": len(predictions), "unique_graded": len(graded),
              "missing": 0, "duplicates": 0, "regrade_mismatches": 0, "correct": correct,
              "accuracy_percent": 100 * correct / len(ids), "generation_seed": seed,
              "judge_seed": 3407, "status": "PASS", "profile": identity["profile"],
              "scope": "Stored-judge grading replay; not semantic correctness or new judge inference.",
              "generation_status": dict(collections.Counter(r["generation_status"] for r in graded.values()))}
    write_json(out / f"{tag}.verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("source-check")
    sub.add_parser("runtime-check")
    sub.add_parser("base-path")
    record_parser = sub.add_parser("records-check")
    record_parser.add_argument("path")
    begin_parser = sub.add_parser("begin")
    begin_parser.add_argument("model")
    begin_parser.add_argument("tag")
    begin_parser.add_argument("--seed", type=int, default=42)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("tag")
    args = parser.parse_args()
    if args.command == "source-check":
        source_check()
    elif args.command == "runtime-check":
        print(json.dumps(runtime_info(), ensure_ascii=False, indent=2))
    elif args.command == "records-check":
        source_check()
        records_check(Path(args.path))
    elif args.command == "begin":
        begin(args.model, args.tag, args.seed)
    elif args.command == "verify":
        verify(args.tag)
    elif args.command == "base-path":
        print(base_model_path())
        return
    print("PASS", args.command)


if __name__ == "__main__":
    main()
