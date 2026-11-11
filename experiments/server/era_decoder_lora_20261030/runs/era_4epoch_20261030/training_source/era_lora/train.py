"""Auditable decoder-only paired LoRA training. Planning is the default action."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any

from .common import canonical_json_hash, config_digest, load_config, resolve_path, sha256_file, write_json

ARM_NAMES = ("plain", "era")
MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
BASE_CONFIG_SHA256 = "edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d"
TARGET_PATTERN = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$")
TRAINABLE_PATTERN = re.compile(
    r"(?:^|\.)model\.language_model\.layers\.\d+\.self_attn\."
    r"(?:q_proj|k_proj|v_proj|o_proj)\.lora_[AB]\.default\.weight$"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def discover_decoder_targets(model: Any, linear_type: type | None = None) -> list[str]:
    """Enumerate exact language paths; suffix matching must never touch vision."""
    if linear_type is None:
        from torch.nn import Linear
        linear_type = Linear
    count = int(model.config.text_config.num_hidden_layers)
    expected = {
        f"model.language_model.layers.{i}.self_attn.{projection}"
        for i in range(count) for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
    }
    actual = {name for name, module in model.named_modules()
              if TARGET_PATTERN.fullmatch(name) and isinstance(module, linear_type)}
    if count < 1 or actual != expected:
        raise RuntimeError(f"Decoder target mismatch: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    return sorted(actual)


def assert_trainables(model: Any, targets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not TRAINABLE_PATTERN.search(name):
            raise RuntimeError(f"Unexpected trainable parameter: {name}")
        rows.append({"name": name, "shape": list(parameter.shape),
                     "numel": parameter.numel(), "dtype": str(parameter.dtype)})
    if len(rows) != 2 * len(targets):
        raise RuntimeError(f"Expected two LoRA matrices per target, got {len(rows)} for {len(targets)} targets")
    for target in targets:
        for matrix in ("A", "B"):
            suffix = f"{target}.lora_{matrix}.default.weight"
            if sum(row["name"].endswith(suffix) for row in rows) != 1:
                raise RuntimeError(f"Missing or duplicate trainable matrix: {suffix}")
    return rows


def tensor_bytes(tensor: Any) -> bytes:
    import torch
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def state_digest(state: dict[str, Any], encode=tensor_bytes) -> str:
    """Hash tensor names, shape, dtype and bytes, independent of save-file order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        metadata = json.dumps([name, list(tensor.shape), str(tensor.dtype)], separators=(",", ":")).encode()
        raw = encode(tensor)
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def parameter_digest(model: Any, trainable: bool) -> str:
    return state_digest({n: p for n, p in model.named_parameters() if p.requires_grad == trainable})


def source_hashes() -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def file_manifest(directory: Path, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Output artifacts must not contain symlinks: {path}")
        if path.is_file() and path.name not in exclude:
            result[path.relative_to(directory).as_posix()] = sha256_file(path)
    return result


def validate_base_weight_index(snapshot: Path, shards: set[str]) -> None:
    """A valid filename alone must not let an altered index select other weights."""
    from safetensors import safe_open
    if {p.name for p in snapshot.glob("*.safetensors")} != shards:
        raise RuntimeError("Unexpected additional or missing safetensors in the base snapshot")
    actual = {}
    for shard in sorted(shards):
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as stream:
            for name in stream.keys():
                if name in actual:
                    raise RuntimeError(f"Duplicate tensor in frozen base shards: {name}")
                actual[name] = shard
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        declared = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        if declared != actual:
            raise RuntimeError("Base weight index does not describe the verified frozen shards")
    elif len(shards) != 1:
        raise RuntimeError("Sharded base weights require their complete index")


def resolve_model_snapshot(config: dict, allow_download: bool = False) -> Path:
    spec = config["model"]
    if spec["repo"] != MODEL_REPO or spec["revision"] != MODEL_REVISION:
        raise ValueError("Training requires the exact frozen baseline model and revision")
    if spec.get("local_path"):
        path = resolve_path(config, spec["local_path"])
    else:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(spec["repo"], revision=spec["revision"],
                                      local_files_only=not allow_download))
    path = Path(path).resolve()
    if not path.is_dir() or path.name != spec["revision"]:
        raise RuntimeError("Use a complete Hugging Face snapshot directory whose basename is the pinned revision")
    model_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if sha256_file(path / "config.json") != BASE_CONFIG_SHA256:
        raise RuntimeError("Base model config.json differs from the executed baseline")
    if model_config.get("model_type") != "qwen3_vl" or model_config.get("quantization_config"):
        raise RuntimeError("Expected unquantized Qwen3-VL base snapshot")
    if (path / "adapter_config.json").exists():
        raise RuntimeError("A trained adapter cannot initialize either experimental arm")
    index_path = path / "model.safetensors.index.json"
    shards = (set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values())
              if index_path.exists() else {"model.safetensors"})
    if not shards or any(not (path / name).is_file() for name in shards):
        raise RuntimeError("Pinned base snapshot has missing weight shards")
    from .compare import BASE_WEIGHTS
    from .evaluate import AUX_HASHES
    if shards != set(BASE_WEIGHTS):
        raise RuntimeError("Base snapshot shard names differ from the executed baseline")
    for name, expected in {**BASE_WEIGHTS, **AUX_HASHES}.items():
        if not (path / name).is_file() or sha256_file(path / name) != expected:
            raise RuntimeError(f"Base snapshot differs from the executed baseline: {name}")
    validate_base_weight_index(path, shards)
    if (path / "chat_template.jinja").exists():
        raise RuntimeError("Unexpected chat-template override in the pinned base snapshot")
    return path


def runtime_check() -> dict:
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Execution requires the isolated Linux Python 3.12 training environment")
    packages = ("torch", "transformers", "peft", "accelerate", "qwen-vl-utils", "safetensors", "huggingface-hub", "Pillow")
    versions = {name: importlib.metadata.version(name) for name in packages}
    for name, expected in {"torch": "2.8.0", "transformers": "4.57.0", "peft": "0.17.1",
                           "accelerate": "1.10.1", "qwen-vl-utils": "0.0.14",
                           "huggingface-hub": "0.36.2", "Pillow": "11.3.0"}.items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"{name}: expected {expected}, found {versions[name]}")
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU supporting BF16 is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("The pilot protocol is single-process, single-GPU training")
    return {"python": sys.version, "platform": platform.platform(), "versions": versions,
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda}


def output_directory(config: dict, arm: str) -> Path:
    if arm not in ARM_NAMES:
        raise ValueError(f"Unknown arm: {arm}")
    return Path(resolve_path(config, config["training"]["output_root"])) / arm


@contextmanager
def file_lock(path: Path):
    """An OS lock survives no process crash; no stale-lock override is needed."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another operation is already using this run: {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_pair_identity(parent: Path, identity: dict, init_digest: str) -> None:
    paired = {k: v for k, v in identity.items() if k != "arm"}
    paired["initial_adapter_sha256"] = init_digest
    path = parent / "pair_identity.json"
    with file_lock(parent / ".pair_identity.lock"):
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != paired:
                raise RuntimeError("Paired runs do not share configuration, data, code and initial LoRA tensors")
        else:
            write_json(path, paired)


def run_identity(config: dict, arm: str, data_manifest: dict) -> dict:
    return {"arm": arm, "config_sha256": config_digest(config),
            "data_sha256": canonical_json_hash(data_manifest), "source_sha256": source_hashes(),
            "base_repo": MODEL_REPO, "base_revision": MODEL_REVISION}


def check_resume(output: Path, resume: str | None, identity: dict) -> tuple[Path | None, dict | None]:
    output = output.resolve()
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Refusing to overwrite nonempty run directory: {output}; use an explicit valid --resume")
        return None, None
    checkpoint = Path(resume).expanduser().resolve()
    if checkpoint.parent != output or not re.fullmatch(r"checkpoint-\d+", checkpoint.name):
        raise RuntimeError("Resume checkpoint must belong directly to this arm's output directory")
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Resume requires the original run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != identity:
        raise RuntimeError("Resume arm, configuration, dataset, code or pinned model does not match")
    if manifest.get("status") == "complete" or (output / "adapter").exists():
        raise RuntimeError("Completed or partially exported final adapters cannot be overwritten")
    required = ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
                "rng_state.pth", "trainer_state.json", "run_identity.json", "checkpoint_integrity.json")
    if any(not (checkpoint / name).is_file() for name in required):
        raise RuntimeError("Resume checkpoint is incomplete: optimizer, scheduler, RNG, adapter and identity are required")
    if json.loads((checkpoint / "run_identity.json").read_text(encoding="utf-8")) != identity:
        raise RuntimeError("Checkpoint identity does not match this arm")
    expected = json.loads((checkpoint / "checkpoint_integrity.json").read_text(encoding="utf-8"))
    actual = file_manifest(checkpoint, exclude=("checkpoint_integrity.json",))
    if expected != actual:
        raise RuntimeError("Resume checkpoint failed file-integrity verification")
    return checkpoint, manifest


def validate_gradient_ownership(model: Any) -> dict:
    import torch
    missing, frozen, nonfinite, nonzero = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                frozen.append(name)
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not bool(torch.isfinite(parameter.grad).all()):
            nonfinite.append(name)
        elif bool(torch.count_nonzero(parameter.grad)):
            nonzero.append(name)
    if missing or frozen or nonfinite or not nonzero:
        raise RuntimeError(f"Invalid first gradients: missing={missing}, frozen={frozen}, nonfinite={nonfinite}, nonzero={len(nonzero)}")
    return {"all_trainables_have_gradient": True, "frozen_parameters_have_no_gradient": True,
            "all_gradients_finite": True, "nonzero_gradient_names": nonzero}


def make_audit_callback(output: Path, identity: dict, frozen_digest: str):
    from transformers import TrainerCallback

    class AuditCallback(TrainerCallback):
        def __init__(self):
            self.checked = False
            self.before = None
            self.gradient_report = None

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            self.before = parameter_digest(model, True)

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if not self.checked:
                self.gradient_report = validate_gradient_ownership(model)

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if self.checked:
                return
            if self.gradient_report is None:
                raise RuntimeError("First-gradient callback was not invoked; unsupported Trainer lifecycle")
            after = parameter_digest(model, True)
            if parameter_digest(model, False) != frozen_digest:
                raise RuntimeError("Frozen base weights changed during the first optimizer step")
            # The first warmup update may legitimately have learning_rate=0.
            write_json(output / "first_step_audit.json", {
                **self.gradient_report, "global_step": state.global_step,
                "frozen_weights_unchanged": True, "adapter_before_sha256": self.before,
                "adapter_after_sha256": after, "adapter_changed": after != self.before,
                "note": "A zero-learning-rate warmup step may leave the adapter unchanged."})
            self.checked = True

        def on_save(self, args, state, control, **kwargs):
            checkpoint = output / f"checkpoint-{state.global_step}"
            write_json(checkpoint / "run_identity.json", identity)
            write_json(checkpoint / "checkpoint_integrity.json",
                       file_manifest(checkpoint, exclude=("checkpoint_integrity.json",)))

    return AuditCallback()


def make_plan(config: dict, arm: str, records: list, data_manifest: dict) -> dict:
    training = config["training"]
    return {"action": "plan_only", "training_started": False, "arm": arm,
            "output": str(output_directory(config, arm)), "items": len(records),
            "effective_batch_size": training["micro_batch_size"] * training["gradient_accumulation_steps"],
            "epochs": training["epochs"], "checkpoint_selection": "fixed final epoch; no MMMU selection",
            "identity": run_identity(config, arm, data_manifest),
            "training": training, "lora": config["lora"],
            "execute_instruction": "Pass --execute only when ready to run actual GPU training."}


def execute_training(config: dict, arm: str, records: list, data_manifest: dict,
                     resume: str | None = None, allow_download: bool = False) -> dict:
    # Keep the lock outside the arm directory, whose emptiness is an overwrite guard.
    output = output_directory(config, arm).resolve()
    with file_lock(output.parent / f".{arm}.lock"):
        return _execute_training_locked(config, arm, records, data_manifest, resume, allow_download)


def _execute_training_locked(config: dict, arm: str, records: list, data_manifest: dict,
                             resume: str | None = None, allow_download: bool = False) -> dict:
    from .data import AssistantOnlyCollator, PairedSFTDataset, load_processor, validate_encoded_pair
    environment = runtime_check()
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, set_seed

    training = config["training"]
    if training["micro_batch_size"] != 1 or training.get("dataloader_num_workers", 0) != 0:
        raise ValueError("The audited pilot requires microbatch 1 and dataloader_num_workers 0")
    if training["seed"] != training["data_seed"]:
        raise ValueError("Training and data seeds must match for this pilot")
    output = output_directory(config, arm).resolve()
    identity = run_identity(config, arm, data_manifest)
    checkpoint, previous = check_resume(output, resume, identity)
    snapshot = resolve_model_snapshot(config, allow_download)
    processor_config = dict(config, model=dict(config["model"], local_path=str(snapshot)))
    processor = load_processor(processor_config)
    # Verify every pair, including the longest images and targets, before loading the GPU model.
    encoded_report = [validate_encoded_pair(row, processor, config) for row in records]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "encoded_pair_preflight.json", encoded_report)
    set_seed(training["seed"])
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(snapshot), dtype=torch.bfloat16, attn_implementation=training["attn_implementation"],
        local_files_only=True, low_cpu_mem_usage=True, use_safetensors=True)
    model.requires_grad_(False)
    targets = discover_decoder_targets(model)
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules=targets, r=lora["r"],
        lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
        modules_to_save=None, init_lora_weights=True))
    trainables = assert_trainables(model, targets)
    model.config.use_cache = False
    if training["gradient_checkpointing"]:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    init_digest = parameter_digest(model, True)
    frozen_digest = parameter_digest(model, False)
    if previous and previous.get("initial_adapter_sha256") != init_digest:
        raise RuntimeError("Fresh adapter initialization differs from the original run")
    ensure_pair_identity(output.parent, identity, init_digest)
    other_arm = "era" if arm == "plain" else "plain"
    sibling_path = output.parent / other_arm / "run_manifest.json"
    if sibling_path.exists():
        sibling = json.loads(sibling_path.read_text(encoding="utf-8"))
        peer_identity = dict(identity, arm=other_arm)
        if sibling.get("identity") != peer_identity or sibling.get("initial_adapter_sha256") != init_digest:
            raise RuntimeError("The other arm does not share identical initialization, data, configuration and code")
    write_json(output / "trainable_parameters.json", {"targets": targets, "parameters": trainables,
               "trainable_numel": sum(row["numel"] for row in trainables)})
    if checkpoint is None:
        model.save_pretrained(output / "initial_adapter", safe_serialization=True)
    manifest = {"identity": identity, "status": "running", "started_at": utc_now(),
                "environment": environment, "base_snapshot": str(snapshot),
                "base_files_sha256": {p.name: sha256_file(p) for p in sorted(snapshot.iterdir()) if p.is_file()},
                "initial_adapter_sha256": init_digest, "frozen_parameters_sha256": frozen_digest,
                "data_manifest": data_manifest, "training": training, "lora": lora,
                "processor_provenance": processor._era_provenance,
                "loss_normalization": "mean assistant-token CE per example, then mean over actual accumulation group; microbatch=1",
                "resume_checkpoint": str(checkpoint) if checkpoint else None,
                "prior_attempts": (previous.get("prior_attempts", []) + [{
                    "started_at": previous.get("started_at"), "status": previous.get("status"),
                    "resume_checkpoint": previous.get("resume_checkpoint")}]) if previous else [],
                "checkpoint_selection": "fixed_final_epoch", "final_training_release": False}
    write_json(output / "run_manifest.json", manifest)
    arguments = TrainingArguments(
        output_dir=str(output), overwrite_output_dir=False,
        num_train_epochs=training["epochs"], per_device_train_batch_size=1,
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"], weight_decay=training["weight_decay"],
        warmup_ratio=training["warmup_ratio"], lr_scheduler_type=training["lr_scheduler_type"],
        max_grad_norm=training["max_grad_norm"], adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
        optim="adamw_torch", bf16=True, fp16=False,
        gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=training["seed"], data_seed=training["data_seed"],
        dataloader_num_workers=0, dataloader_drop_last=False,
        remove_unused_columns=False, save_strategy="epoch", save_safetensors=True,
        save_only_model=False, eval_strategy="no", logging_strategy="steps", logging_steps=1,
        report_to=[], load_best_model_at_end=False, disable_tqdm=False)
    trainer = Trainer(model=model, args=arguments, train_dataset=PairedSFTDataset(records, arm),
                      data_collator=AssistantOnlyCollator(processor, config),
                      callbacks=[make_audit_callback(output, identity, frozen_digest)])
    # Do not let **kwargs on Qwen.forward silently opt into token-weighted batch
    # normalization. With microbatch 1, this preserves equal weight per paired
    # question; Trainer 4.57 divides by the actual accumulation group size.
    trainer.model_accepts_loss_kwargs = False
    try:
        torch.cuda.reset_peak_memory_stats()
        result = trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
        if parameter_digest(model, False) != frozen_digest:
            raise RuntimeError("Frozen base weights changed during training")
        final_digest = parameter_digest(model, True)
        if final_digest == init_digest:
            raise RuntimeError("Adapter parameters did not change over the full training run")
        model.config.use_cache = True
        model.save_pretrained(output / "adapter", safe_serialization=True)
        trainer.save_state()
        trainer.save_metrics("train", result.metrics)
        manifest.update(status="complete", completed_at=utc_now(),
                        final_adapter_sha256=final_digest, actual_optimizer_steps=trainer.state.global_step,
                        train_metrics=result.metrics, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
                        adapter_files_sha256=file_manifest(output / "adapter"),
                        frozen_weights_unchanged=True)
        write_json(output / "run_manifest.json", manifest)
        return manifest
    except BaseException as error:
        manifest.update(status="failed", failed_at=utc_now(), failure=f"{type(error).__name__}: {error}")
        write_json(output / "run_manifest.json", manifest)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--execute", action="store_true", help="Start actual GPU training; absent means plan only")
    parser.add_argument("--resume", help="Explicit complete checkpoint directory from this same arm")
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading the pinned base snapshot")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    from .data import load_paired_records
    records, data_manifest = load_paired_records(config)
    result = (execute_training(config, args.arm, records, data_manifest, args.resume, args.allow_download)
              if args.execute else make_plan(config, args.arm, records, data_manifest))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
