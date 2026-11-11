"""Export a completed LoRA onto its fresh pinned BF16 base; plan by default."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import uuid

from .common import config_digest, load_config, sha256_file, write_json
from .train import (ARM_NAMES, MODEL_REPO, MODEL_REVISION, discover_decoder_targets,
                    file_lock, file_manifest, output_directory, resolve_model_snapshot,
                    state_digest, utc_now)


def validate_adapter_run(config: dict, arm: str) -> tuple[Path, dict]:
    output = output_directory(config, arm).resolve()
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"No training run manifest exists: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("identity", {})
    if manifest.get("status") != "complete":
        raise RuntimeError("Only successfully completed fixed-final-epoch runs can be merged")
    if (identity.get("arm") != arm or identity.get("config_sha256") != config_digest(config)
            or identity.get("base_repo") != MODEL_REPO or identity.get("base_revision") != MODEL_REVISION):
        raise RuntimeError("Training arm, configuration or frozen base differs from the requested export")
    adapter = output / "adapter"
    expected = manifest.get("adapter_files_sha256")
    if not expected or expected != file_manifest(adapter):
        raise RuntimeError("Final adapter files do not match the completed training manifest")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError("A safetensors adapter is required")
    return output, manifest


def processor_files(snapshot: Path) -> list[Path]:
    """Copy the original serialization, including chat templates, without re-saving."""
    suffixes = {".json", ".txt", ".model", ".jinja", ".tiktoken"}
    return sorted(path for path in snapshot.rglob("*") if path.is_file()
                  and path.suffix in suffixes and not path.name.endswith(".safetensors.index.json")
                  and path.name != "adapter_config.json")


def copy_processor_files(snapshot: Path, destination: Path) -> dict[str, str]:
    paths = processor_files(snapshot)
    required = {"config.json", "tokenizer_config.json", "preprocessor_config.json"}
    found = {p.relative_to(snapshot).as_posix() for p in paths}
    if not required.issubset(found):
        raise RuntimeError(f"Base snapshot lacks original processor artifacts: {sorted(required - found)}")
    hashes = {}
    for path in paths:
        relative = path.relative_to(snapshot)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        expected = sha256_file(path)
        if sha256_file(target) != expected:
            raise RuntimeError(f"Byte-for-byte processor copy failed: {relative}")
        hashes[relative.as_posix()] = expected
    return hashes


def tensor_hashes(model) -> dict[str, str]:
    return {name: state_digest({name: tensor}) for name, tensor in model.state_dict().items()}


def tied_state_groups(model) -> list[list[str]]:
    """Safetensors intentionally stores only one name for exactly tied weights."""
    groups = {}
    for name, tensor in model.state_dict().items():
        key = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), str(tensor.device))
        groups.setdefault(key, []).append(name)
    return [sorted(names) for names in groups.values() if len(names) > 1]


def validate_merged_state(before: dict[str, str], after: dict[str, str], targets: list[str]) -> dict:
    if before.keys() != after.keys():
        raise RuntimeError("Merged model state keys differ from the original baseline architecture")
    permitted = {name + ".weight" for name in targets}
    if not permitted.issubset(before):
        raise RuntimeError("Decoder target weights are missing from the baseline state")
    changed = sorted(name for name in before if before[name] != after[name])
    unexpected = set(changed) - permitted
    if unexpected:
        raise RuntimeError(f"Weights outside decoder LoRA targets changed: {sorted(unexpected)}")
    if not changed:
        raise RuntimeError("Merged weights are identical to baseline; the learned update may be missing")
    return {"state_keys_identical": True, "all_non_target_tensors_identical": True,
            "changed_target_weights": changed, "changed_target_count": len(changed)}


def validate_saved_weights(directory: Path, expected_keys: set[str], tied_groups: list[list[str]] | None = None) -> dict:
    from safetensors import safe_open
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        weight_map = None
        shards = ["model.safetensors"]
    found = set()
    for shard in shards:
        path = directory / shard
        if not path.is_file():
            raise RuntimeError(f"Missing exported weight shard: {shard}")
        with safe_open(str(path), framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in found:
                    raise RuntimeError(f"Duplicate exported tensor: {name}")
                found.add(name)
                if "lora_" in name:
                    raise RuntimeError("Standalone export still contains LoRA modules")
                if weight_map is not None and weight_map.get(name) != shard:
                    raise RuntimeError(f"Exported index points to the wrong shard: {name}")
                # Qwen3-VL's persistent state comprises floating point model weights.
                if stream.get_slice(name).get_dtype() != "BF16":
                    raise RuntimeError(f"Exported tensor is not BF16: {name}")
    allowed_missing = set()
    for group in tied_groups or []:
        if not set(group).issubset(expected_keys):
            raise RuntimeError("Unknown key in tied-weight structure")
        if found.intersection(group):
            allowed_missing.update(group)
    missing = expected_keys - found
    if found - expected_keys or missing - allowed_missing or (weight_map is not None and set(weight_map) != found):
        raise RuntimeError("Exported safetensors state differs from the merged model state")
    return {"shards": shards, "tensor_count": len(found), "all_weights_bfloat16": True,
            "omitted_tied_aliases": sorted(missing)}


def make_plan(config: dict, arm: str) -> dict:
    output = output_directory(config, arm).resolve()
    return {"action": "plan_only", "merge_started": False, "arm": arm,
            "adapter": str(output / "adapter"), "destination": str(output / "merged"),
            "base_repo": MODEL_REPO, "base_revision": MODEL_REVISION,
            "precision": "bfloat16", "processor_policy": "copy frozen base files byte-for-byte",
            "config_sha256": config_digest(config),
            "execute_instruction": "Pass --execute after this arm completes training."}


def execute_merge(config: dict, arm: str, allow_download: bool = False) -> dict:
    output = output_directory(config, arm).resolve()
    with file_lock(output.parent / f".{arm}.lock"):
        return _execute_merge_locked(config, arm, allow_download)


def _execute_merge_locked(config: dict, arm: str, allow_download: bool = False) -> dict:
    output, training_manifest = validate_adapter_run(config, arm)
    destination = output / "merged"
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing merged checkpoint: {destination}")
    versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "safetensors")}
    for package, expected in {"torch": "2.8.0", "transformers": "4.57.0", "peft": "0.17.1"}.items():
        if versions[package].split("+")[0] != expected:
            raise RuntimeError(f"{package}: expected {expected}, found {versions[package]}")
    import torch
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration

    snapshot = resolve_model_snapshot(config, allow_download)
    actual_base = {p.name: sha256_file(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    if actual_base != training_manifest.get("base_files_sha256"):
        raise RuntimeError("Fresh base snapshot files differ from the base used for training")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        str(snapshot), dtype=torch.bfloat16, attn_implementation=config["training"]["attn_implementation"],
        local_files_only=True, low_cpu_mem_usage=True, use_safetensors=True)
    targets = discover_decoder_targets(base)
    adapter_config = json.loads((output / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
    expected = config["lora"]
    # PEFT 0.17 minimizes full target paths to suffixes when saving. Resolve
    # those suffixes against the fresh base and require the same exact modules.
    saved_targets = adapter_config.get("target_modules", [])
    if not isinstance(saved_targets, list):
        raise RuntimeError("Expected an explicit saved LoRA target list")
    resolved_targets = {name for name, _ in base.named_modules()
                        if any(name == target or name.endswith('.' + target) for target in saved_targets)}
    if (resolved_targets != set(targets)
            or adapter_config.get("r") != expected["r"]
            or adapter_config.get("lora_alpha") != expected["alpha"]
            or adapter_config.get("lora_dropout") != expected["dropout"]
            or adapter_config.get("bias") != "none" or adapter_config.get("modules_to_save")):
        raise RuntimeError("Adapter structure is not the configured decoder-only experiment")
    before = tensor_hashes(base)
    model = PeftModel.from_pretrained(base, str(output / "adapter"), is_trainable=False)
    model.eval()
    merged = model.merge_and_unload(safe_merge=True)
    merged.to(dtype=torch.bfloat16)
    merged.config.use_cache = True
    if any("lora_" in name for name, _ in merged.named_modules()):
        raise RuntimeError("LoRA modules remain after merge")
    structure = validate_merged_state(before, tensor_hashes(merged), targets)
    tied_groups = tied_state_groups(merged)
    # A failed export remains in its uniquely named incomplete directory for diagnosis.
    staging = output / f"merged.incomplete-{uuid.uuid4().hex[:12]}"
    staging.mkdir()
    merged.save_pretrained(staging, safe_serialization=True, max_shard_size="4GB")
    copied = copy_processor_files(snapshot, staging)
    serialized = validate_saved_weights(staging, set(before), tied_groups)
    if (staging / "adapter_config.json").exists():
        raise RuntimeError("Merged export must not be recognized as a standalone adapter")
    manifest = {"created_at": utc_now(), "arm": arm, "destination": str(destination),
                "training_identity": training_manifest["identity"],
                "base_snapshot": str(snapshot), "versions": versions,
                "adapter_files_sha256": training_manifest["adapter_files_sha256"],
                "processor_files_sha256": copied, "structure_validation": structure,
                "serialized_validation": serialized, "artifact_files_sha256": file_manifest(staging),
                "merge_precision": "bfloat16 base; PEFT adapter dtype preserved, final model bfloat16",
                "baseline_evaluator_compatible": True,
                "numerical_note": "Merge rounding can change outputs relative to unmerged inference; compare both arms using these exports."}
    write_json(staging / "artifact_manifest.json", manifest)
    staging.rename(destination)
    write_json(output / "merge_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True, choices=ARM_NAMES)
    parser.add_argument("--execute", action="store_true", help="Actually load and merge the completed adapter")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    result = execute_merge(config, args.arm, args.allow_download) if args.execute else make_plan(config, args.arm)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
