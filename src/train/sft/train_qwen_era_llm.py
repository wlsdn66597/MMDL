#!/usr/bin/env python3
"""Train Qwen3-VL-4B on original + ERA answers and evaluate every epoch on MMMU."""

import argparse
import hashlib
import io
import json
import logging
import math
import os
import random
import re
import shutil
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(os.environ.get("QWEN_BASELINE_ROOT", "/shared_dir/qwen_baseline")).expanduser().resolve()
DATA_DIR = ROOT / "data/era_2000"
BASELINE_REPO = Path(__file__).resolve().parents[3]
HF_CACHE = ROOT / "cache/huggingface"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MODEL_DIR = HF_CACHE / "hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots" / REVISION
RUN_DIR = ROOT / "training/original_plus_era_llm"
# The released trainer before evaluation was added used the same data and optimizer logic.
TRAINING_ONLY_SCRIPT_SHA256 = "874f395bc036042c509ebdb5def90d197c14e937e6d1a384e8140bd1f201dca9"
LANGUAGE_ATTENTION_PROJECTIONS = (
    ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj",
)
SETTINGS = {
    "examples": 2000,
    "epochs": 5,
    "seed": 3407,
    "micro_batch": 1,
    "accumulation": 16,
    "max_tokens": 4096,
    "min_pixels": 262144,
    "max_pixels": 1048576,
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "clip_grad": 1.0,
    "warmup_ratio": 0.03,
    "dtype": "bfloat16",
    "quantization": None,
    "gradient_checkpointing": True,
    "packing": False,
    "lora_targets": list(LANGUAGE_ATTENTION_PROJECTIONS),
    "vision_encoder_frozen": True,
}
ERA_PATTERN = re.compile(
    r"\AEvidence:\s*(.+?)\r?\nRule:\s*(.+?)\r?\nApplication:\s*(.+?)\r?\nFinal answer:\s*(.+)\Z",
    re.DOTALL,
)
CHAT_TOKENS = re.compile(r"<\|(?:im_start|im_end|endoftext|vision_start|vision_end|image_pad|video_pad)\|>")
ANSWER_HEADERS = (
    "", "\n\nTo summarize the reasoning in Evidence, Rule, and Application form:\nEvidence: ",
    "\nRule: ", "\nApplication: ",
)
LOG = logging.getLogger("qwen_era")


@dataclass
class Example:
    id: str
    image_path: Path
    image_size: tuple[int, int]
    prompt: str
    sections: tuple[str, ...]
    final_answer: str
    era_answer_differs: bool


@dataclass
class EncodedExample:
    example: Example
    image_size: tuple[int, int]
    image_tokens: int
    prefix_ids: list[int]
    target_ids: list[int]
    original_tokens: int
    era_tokens: int
    original_trimmed: int
    era_trimmed: int

    @property
    def length(self):
        return len(self.prefix_ids) + len(self.target_ids)


def baseline_inputs(repo_dir):
    source = repo_dir / "src"
    if not (source / "evaluate.py").is_file():
        raise FileNotFoundError(f"Baseline evaluator was not found at {source}.")
    sys.path.insert(0, str(source))
    try:
        from evaluate import prompt_for
        from prepare import config
    finally:
        sys.path.pop(0)
    baseline = config(source / "config.json")
    digest = hashlib.sha256()
    for filename in ("evaluate.py", "config.json"):
        digest.update((source / filename).read_bytes())
    return partial(prompt_for, c=baseline), baseline["vision"], digest.hexdigest()


def load_examples(data_dir, prompt_for):
    data_dir = data_dir.resolve()
    raw = (data_dir / "data.json").read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != SETTINGS["examples"]:
        raise ValueError(f"Expected {SETTINGS['examples']} rows in {data_dir / 'data.json'}.")

    examples = []
    seen_ids = set()
    fingerprint = hashlib.sha256(raw)
    for row in rows:
        row_id = row["id"]
        if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
            raise ValueError(f"Missing or duplicate sample ID: {row_id!r}")
        seen_ids.add(row_id)
        image_fields = [key for key in row if re.fullmatch(r"image_\d+", key) and row[key]]
        if image_fields != ["image_1"]:
            raise ValueError(f"{row_id}: expected exactly one image in image_1.")
        image_path = (data_dir / row["image_1"]).resolve()
        if not image_path.is_relative_to(data_dir):
            raise ValueError(f"{row_id}: image path is outside the data directory.")
        image_bytes = image_path.read_bytes()
        fingerprint.update(hashlib.sha256(image_bytes).digest())
        with Image.open(io.BytesIO(image_bytes)) as image:
            width, height = image.size
            if image.getexif().get(274, 1) in (5, 6, 7, 8):
                width, height = height, width
            # Load each file once here so a corrupt image fails before training.
            image.load()

        question = row["question"].strip()
        options = row["options"]
        if row["question_type"] == "multiple-choice":
            if not isinstance(options, list) or not 2 <= len(options) <= 26:
                raise ValueError(f"{row_id}: invalid multiple-choice options.")
        elif row["question_type"] != "open" or options:
            raise ValueError(f"{row_id}: invalid open-ended question.")
        prompt = prompt_for(row)

        # Keep one final-answer field and omit markup that wrapped the source reasoning.
        original = re.sub(r"</?think>", "", row["explanation"], flags=re.IGNORECASE)
        original = re.sub(r"(?im)^[ \t]*Final answer:[^\r\n]*$", "", original).strip()
        era = ERA_PATTERN.fullmatch(row["explanation_era"].strip())
        if not era:
            raise ValueError(f"{row_id}: expected Evidence, Rule, Application and Final answer.")
        evidence, rule, application, answer = (part.strip() for part in era.groups())
        answer = " ".join(answer.split())
        if row["question_type"] == "multiple-choice" and (
            len(answer) != 1 or answer not in string.ascii_uppercase[:len(options)]
        ):
            raise ValueError(f"{row_id}: ERA final answer must be one available option letter.")
        sections = (original, evidence, rule, application)
        if not all((question, answer, *sections)) or any(
            CHAT_TOKENS.search(text) for text in (prompt, answer, *sections)
        ):
            raise ValueError(f"{row_id}: missing text or unexpected chat control token.")

        examples.append(Example(
            row_id, image_path, (width, height), prompt, sections, answer,
            answer != " ".join(row["answer"].split()),
        ))
    return examples, fingerprint.hexdigest()


def fit_reasoning(lengths, available):
    """Prioritize complete ERA, retaining part of the original when space is tight."""
    if sum(lengths) <= available:
        return lengths
    if available < len(lengths):
        raise ValueError("The question, image and final answer leave no room for reasoning.")
    original_reserve = min(lengths[0], 256, max(1, available // 4))
    era_budget = min(sum(lengths[1:]), available - original_reserve)
    keep = [min(lengths[0], available - era_budget), 0, 0, 0]
    era_order = sorted(range(1, 4), key=lambda index: lengths[index])
    for index, section in enumerate(era_order):
        keep[section] = min(lengths[section], era_budget // (len(era_order) - index))
        era_budget -= keep[section]
    return keep


def encode_examples(examples, processor, resize, max_tokens, vision):
    tokenizer = processor.tokenizer
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    headers = [encode(text) for text in ANSWER_HEADERS]
    eos = encode("<|im_end|>")
    if eos != [tokenizer.eos_token_id]:
        raise ValueError("The cached tokenizer does not use the expected Qwen end token.")

    encoded = []
    for example in examples:
        width, height = example.image_size
        height, width = resize(
            height, width, factor=factor,
            min_pixels=vision["min_pixels"], max_pixels=vision["max_pixels"],
        )
        if width % factor or height % factor or not (
            vision["min_pixels"] <= width * height <= vision["max_pixels"]
        ):
            raise ValueError(f"{example.id}: invalid resized image dimensions {width}x{height}.")
        image_tokens = (width // factor) * (height // factor)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": str(example.image_path)},
            {"type": "text", "text": example.prompt},
        ]}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if prefix.count(processor.image_token) != 1:
            raise ValueError(f"{example.id}: expected one image placeholder in the Qwen template.")
        prefix_ids = encode(prefix.replace(processor.image_token, processor.image_token * image_tokens, 1))
        if prefix_ids.count(processor.image_token_id) != image_tokens:
            raise ValueError(f"{example.id}: image token count differs from the resized image grid.")

        sections = [encode(text) for text in example.sections]
        final = encode(f"\nFinal answer: {example.final_answer}") + eos
        available = max_tokens - len(prefix_ids) - len(final) - sum(map(len, headers))
        lengths = [len(section) for section in sections]
        try:
            keep = fit_reasoning(lengths, available)
        except ValueError as exc:
            raise ValueError(f"{example.id}: {exc}") from exc
        target_ids = []
        for header, section, count in zip(headers, sections, keep, strict=True):
            target_ids.extend(header)
            target_ids.extend(section[:count])
        target_ids.extend(final)
        item = EncodedExample(
            example, (width, height), image_tokens, prefix_ids, target_ids,
            lengths[0], sum(lengths[1:]), lengths[0] - keep[0], sum(lengths[1:]) - sum(keep[1:]),
        )
        if item.length > max_tokens or min(keep) < 1:
            raise ValueError(f"{example.id}: invalid combined training sequence.")
        encoded.append(item)
    return encoded


def batch_for(item, processor):
    import torch

    with Image.open(item.example.image_path) as image:
        picture = ImageOps.exif_transpose(image).convert("RGB")
        picture = picture.resize(item.image_size, Image.Resampling.BICUBIC)
    vision = processor.image_processor(images=[picture], do_resize=False, return_tensors="pt")
    merge_size = processor.image_processor.merge_size
    if vision["image_grid_thw"].prod().item() // merge_size**2 != item.image_tokens:
        raise RuntimeError(f"{item.example.id}: image grid differs from tokenization.")
    ids = torch.tensor([item.prefix_ids + item.target_ids], dtype=torch.long, device="cuda:0")
    labels = ids.clone()
    labels[:, :len(item.prefix_ids)] = -100
    return {
        "input_ids": ids,
        "labels": labels,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": vision["pixel_values"].to("cuda:0", dtype=torch.bfloat16),
        "image_grid_thw": vision["image_grid_thw"].to("cuda:0"),
    }


def is_language_attention(name):
    return ".language_model.layers." in name and name.endswith(LANGUAGE_ATTENTION_PROJECTIONS)


def adapter_targets(model, torch):
    targets = [name for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and is_language_attention(name)]
    expected = model.config.text_config.num_hidden_layers * len(LANGUAGE_ATTENTION_PROJECTIONS)
    if not targets or len(targets) != expected:
        raise RuntimeError(f"Expected {expected} language attention projections, found {len(targets)}.")
    return targets


def trainable_language_adapters(model):
    parameters = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            module, marker, _ = name.partition(".lora_")
            if not marker or not is_language_attention(module):
                raise RuntimeError(f"Unexpected trainable parameter outside language LoRA: {name}")
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError("No trainable language attention adapters were found.")
    return parameters


def adapter_pattern(targets):
    # An exact regex keeps PEFT from shortening a long list of module names on save.
    return "(?:" + "|".join(re.escape(name) for name in targets) + ")"


def checkpoint_config(checkpoint, target_pattern, config_type):
    config = config_type.from_pretrained(checkpoint)
    if config.target_modules != target_pattern:
        raise RuntimeError(f"Checkpoint does not match language-only LoRA: {checkpoint}")
    return config


def write_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def latest_checkpoint(run_dir):
    checkpoints = [run_dir / f"epoch-{epoch:02d}" for epoch in range(1, SETTINGS["epochs"] + 1)]
    present = [path for path in checkpoints if path.exists()]
    if present != checkpoints[:len(present)]:
        raise RuntimeError("Checkpoint folders must be contiguous from epoch-01.")
    for path in present:
        for name in ("adapter_config.json", "adapter_model.safetensors", "training_state.pt", "metrics.json"):
            if not (path / name).is_file():
                raise RuntimeError(f"Incomplete checkpoint: {path / name}")
    return len(present), present[-1] if present else None


def save_epoch(run_dir, epoch, step, mean_loss, model, optimizer, scheduler):
    import torch

    destination = run_dir / f"epoch-{epoch:02d}"
    with tempfile.TemporaryDirectory(prefix=".writing-", dir=run_dir) as tmp:
        stage = Path(tmp) / "checkpoint"
        model.save_pretrained(stage, safe_serialization=True)
        torch.save({
            "epoch": epoch,
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(0),
        }, stage / "training_state.pt")
        write_json(stage / "metrics.json", {"epoch": epoch, "step": step, "mean_loss": mean_loss})
        stage.rename(destination)
    LOG.info("Saved epoch %d/%d, step %d, loss %.5f: %s",
             epoch, SETTINGS["epochs"], step, mean_loss, destination)


def train(run_dir, examples, data_hash, baseline_hash, max_tokens, vision, preflight_only):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from qwen_vl_utils.vision_process import smart_resize
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup

    if not (MODEL_DIR / "config.json").is_file():
        raise FileNotFoundError(f"Cached Qwen3-VL-4B snapshot was not found: {MODEL_DIR}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Run inside the GPU container with a BF16-capable CUDA GPU.")
    torch.set_num_threads(4)
    torch.manual_seed(SETTINGS["seed"])
    torch.cuda.manual_seed_all(SETTINGS["seed"])

    steps_per_epoch, remainder = divmod(len(examples), SETTINGS["accumulation"])
    if remainder:
        raise ValueError("The dataset size must be divisible by gradient accumulation.")
    total_steps = steps_per_epoch * SETTINGS["epochs"]
    warmup_steps = math.ceil(total_steps * SETTINGS["warmup_ratio"])
    done_epochs, checkpoint = (0, None) if preflight_only else latest_checkpoint(run_dir)
    config_path = run_dir / "run.json"
    config = {
        **SETTINGS, **vision, "max_tokens": max_tokens,
        "model_revision": REVISION, "data_sha256": data_hash, "baseline_sha256": baseline_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup_steps,
    }
    if checkpoint:
        if not config_path.is_file():
            raise RuntimeError(f"Checkpoint exists without its run configuration: {config_path}")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config.get("script_sha256") == TRAINING_ONLY_SCRIPT_SHA256:
            config["script_sha256"] = TRAINING_ONLY_SCRIPT_SHA256
        if saved_config != config:
            raise RuntimeError(f"Data, script or training settings changed. Existing run: {run_dir}")
    elif not preflight_only:
        write_json(config_path, config)
    if done_epochs == SETTINGS["epochs"]:
        LOG.info("Training already complete: %s", checkpoint)
        return

    LOG.info("GPU: %s. %d examples, %d updates/epoch, %d updates total, %d warmup.",
             torch.cuda.get_device_name(0), len(examples), steps_per_epoch, total_steps, warmup_steps)
    LOG.info("Tokenizing questions, images, original explanations and ERA sections.")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    encoded = encode_examples(examples, processor, smart_resize, max_tokens, vision)
    if not preflight_only:
        write_json(run_dir / "data_lengths.json", [
            {"id": item.example.id, "total_tokens": item.length, "image_tokens": item.image_tokens,
             "image_width": item.image_size[0], "image_height": item.image_size[1],
             "original_tokens_before_trim": item.original_tokens, "era_tokens_before_trim": item.era_tokens,
             "original_tokens_trimmed": item.original_trimmed, "era_tokens_trimmed": item.era_trimmed,
             "era_answer_differs_from_original": item.example.era_answer_differs}
            for item in encoded
        ])
    LOG.info("Maximum sequence: %d tokens. Trimmed original in %d rows, ERA in %d rows.",
             max(item.length for item in encoded),
             sum(item.original_trimmed > 0 for item in encoded),
             sum(item.era_trimmed > 0 for item in encoded))

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": "cuda:0"}, local_files_only=True,
    )
    vision_encoder = base.model.visual
    vision_encoder.requires_grad_(False)
    targets = adapter_targets(base, torch)
    target_pattern = adapter_pattern(targets)
    if checkpoint:
        adapter_config = checkpoint_config(checkpoint, target_pattern, LoraConfig)
        model = PeftModel.from_pretrained(base, checkpoint, config=adapter_config, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(
            r=SETTINGS["rank"], lora_alpha=SETTINGS["alpha"], lora_dropout=SETTINGS["dropout"],
            bias="none", target_modules=target_pattern, task_type="CAUSAL_LM",
        ))
    language_model = base.model.language_model
    language_model.config.use_cache = False
    language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    language_model.enable_input_require_grads()
    model.train()
    vision_encoder.eval()
    trainable = trainable_language_adapters(model)
    LOG.info("Vision encoder frozen and in eval mode. Language attention modules: %d, trainable parameters: %d.",
             len(targets), sum(parameter.numel() for parameter in trainable))
    optimizer = torch.optim.AdamW(
        trainable, lr=SETTINGS["lr"], weight_decay=SETTINGS["weight_decay"], foreach=False,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    step = done_epochs * steps_per_epoch
    if checkpoint:
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        if state["epoch"] != done_epochs or state["step"] != step:
            raise RuntimeError(f"Checkpoint step does not match the saved epoch: {checkpoint}")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state(state["rng_cuda"], device=0)
        del state
        LOG.info("Resuming after epoch %d, step %d.", done_epochs, step)
    else:
        longest = max(encoded, key=lambda item: (item.length, item.image_tokens))
        LOG.info("Testing forward/backward on the longest sample: %s (%d tokens).",
                 longest.example.id, longest.length)
        torch.cuda.reset_peak_memory_stats()
        with torch.random.fork_rng(devices=[0]):
            loss = model(**batch_for(longest, processor), use_cache=False).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite loss in the preflight sample.")
            loss.backward()
            del loss
            torch.nn.utils.clip_grad_norm_(trainable, SETTINGS["clip_grad"], error_if_nonfinite=True)
        optimizer.zero_grad(set_to_none=True)
        LOG.info("GPU preflight passed. Peak reserved CUDA memory: %.2f GiB.",
                 torch.cuda.max_memory_reserved() / 1024**3)
        if preflight_only:
            return
        LOG.info("Starting epoch 1.")

    accumulation = SETTINGS["accumulation"]
    for epoch in range(done_epochs + 1, SETTINGS["epochs"] + 1):
        order = list(range(len(encoded)))
        random.Random(SETTINGS["seed"] + epoch - 1).shuffle(order)
        epoch_loss, epoch_tokens = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        window_loss, window_tokens = 0.0, 0
        for index, row_index in enumerate(order, start=1):
            item = encoded[row_index]
            target_tokens = len(item.target_ids)
            loss = model(**batch_for(item, processor), use_cache=False).loss
            (loss * target_tokens).backward()
            window_loss = window_loss + loss.detach() * target_tokens
            window_tokens += target_tokens
            del loss
            if index % accumulation:
                continue

            # Normalize across supervised answer tokens in all 16 microbatches.
            for parameter in trainable:
                if parameter.grad is not None:
                    parameter.grad.div_(window_tokens)
            torch.nn.utils.clip_grad_norm_(trainable, SETTINGS["clip_grad"], error_if_nonfinite=True)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            window_sum = window_loss.item()
            epoch_loss += window_sum
            epoch_tokens += window_tokens
            LOG.info("Epoch %d/%d | step %d/%d | loss %.5f | lr %.3g",
                     epoch, SETTINGS["epochs"], step, total_steps, window_sum / window_tokens, lr)
            window_loss, window_tokens = 0.0, 0

        save_epoch(run_dir, epoch, step, epoch_loss / epoch_tokens, model, optimizer, scheduler)
    LOG.info("Training complete: %s", run_dir)


def sha256_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def adapter_identity(checkpoint):
    return {
        "base_revision": REVISION,
        "adapter_config_sha256": sha256_file(checkpoint / "adapter_config.json"),
        "adapter_weights_sha256": sha256_file(checkpoint / "adapter_model.safetensors"),
    }


def merge_epoch(run_dir, epoch):
    import torch
    from peft import LoraConfig, PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    baseline_inputs(BASELINE_REPO)
    from prepare import validate_weights

    epoch_name = f"epoch-{epoch:02d}"
    checkpoint = run_dir / epoch_name
    destination = run_dir / "evaluation" / epoch_name / "merged_model"
    identity = adapter_identity(checkpoint)
    LOG.info("Merging %s LoRA into the original BF16 model for MMMU.", epoch_name)
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map={"": "cuda:0"}, local_files_only=True,
    )
    pattern = adapter_pattern(adapter_targets(base, torch))
    adapter_config = checkpoint_config(checkpoint, pattern, LoraConfig)
    model = PeftModel.from_pretrained(base, checkpoint, config=adapter_config, is_trainable=False)
    with torch.no_grad():
        merged = model.merge_and_unload(safe_merge=True)
    if any(".lora_" in name for name, _ in merged.named_parameters()):
        raise RuntimeError(f"LoRA modules remain after merging {checkpoint}.")

    with tempfile.TemporaryDirectory(prefix=".merging-", dir=destination.parent) as temp:
        stage = Path(temp) / "model"
        merged.save_pretrained(stage, safe_serialization=True, max_shard_size="5GB")
        for path in MODEL_DIR.iterdir():
            # Preserve the exact tokenizer, processor and chat template from the baseline.
            if path.is_file() and path.suffix != ".safetensors" and path.name not in {
                "config.json", "model.safetensors.index.json",
            }:
                shutil.copyfile(path, stage / path.name)
        validate_weights(stage, "qwen3_vl")
        AutoProcessor.from_pretrained(stage, local_files_only=True)
        write_json(stage / "adapter_source.json", identity)
        stage.rename(destination)
    LOG.info("Merged checkpoint ready: %s", destination)


def completed_score(folder):
    if not all((folder / name).is_file() for name in ("summary.json", "results.json", "scores.md")):
        return None
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["total"] != 900 or summary["graded"] != 900 or not 0 <= summary["correct"] <= 900:
        raise RuntimeError(f"Invalid MMMU summary in {folder}.")
    if not math.isclose(summary["accuracy_percent"], summary["correct"] / 9, abs_tol=1e-9):
        raise RuntimeError(f"MMMU accuracy does not match the number correct in {folder}.")
    return summary


def write_epoch_scores(eval_dir, scores):
    lines = [
        "| Epoch | Optimizer step | Correct | MMMU accuracy | Details |",
        "|---:|---:|---:|---:|---|",
    ]
    for epoch, step, score in scores:
        lines.append(
            f"| {epoch} | {step} | {score['correct']}/900 | {score['accuracy_percent']:.2f}% "
            f"| [Per-subject scores](epoch-{epoch:02d}/scores.md) |"
        )
    temporary = eval_dir / "accuracy_by_epoch.md.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(eval_dir / "accuracy_by_epoch.md")


def evaluate_epochs(run_dir, lock_fd):
    done_epochs, _ = latest_checkpoint(run_dir)
    if done_epochs != SETTINGS["epochs"]:
        raise RuntimeError(f"Expected {SETTINGS['epochs']} saved epochs, found {done_epochs}.")
    eval_dir = run_dir / "evaluation"
    evaluator = BASELINE_REPO / "src/evaluate.py"
    baseline_code = {p.name: sha256_file(p) for p in sorted(evaluator.parent.glob("*")) if p.is_file()}
    scores = []
    for epoch in range(1, done_epochs + 1):
        epoch_name = f"epoch-{epoch:02d}"
        checkpoint = run_dir / epoch_name
        folder = eval_dir / epoch_name
        folder.mkdir(parents=True, exist_ok=True)
        identity = adapter_identity(checkpoint)
        binding = {"epoch": epoch, "adapter": identity, "baseline_code_sha256": baseline_code}
        binding_path = folder / "evaluation_source.json"
        if binding_path.exists():
            if json.loads(binding_path.read_text(encoding="utf-8")) != binding:
                raise RuntimeError(f"Saved evaluation is for different inputs: {folder}")
        else:
            if any((folder / name).exists() for name in (
                "manifest.json", "predictions.jsonl", "results.json", "summary.json", "merged_model",
            )):
                raise RuntimeError(f"Found evaluation files without a source manifest: {folder}")
            write_json(binding_path, binding)

        summary = completed_score(folder)
        newly_measured = summary is None
        if newly_measured:
            merged_model = folder / "merged_model"
            if not merged_model.exists():
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--run-dir", str(run_dir),
                     "--_merge-epoch", str(epoch)],
                    check=True, pass_fds=(lock_fd,),
                )
            merged_source = json.loads((merged_model / "adapter_source.json").read_text(encoding="utf-8"))
            if merged_source != identity:
                raise RuntimeError(f"Merged model does not match this epoch's adapter: {merged_model}")
            LOG.info("Evaluating %s on the original 900 MMMU validation questions.", epoch_name)
            subprocess.run(
                [sys.executable, str(evaluator), "--checkpoint", str(merged_model),
                 "--run-name", epoch_name, "--output-root", str(eval_dir),
                 "--model-cache", str(HF_CACHE / "hub"), "--dataset-cache", str(HF_CACHE / "datasets")],
                check=True, pass_fds=(lock_fd,),
            )
            summary = completed_score(folder)
            if summary is None:
                raise RuntimeError(f"Evaluation exited before producing all results: {folder}")
        else:
            LOG.info("Reusing completed MMMU evaluation for %s.", epoch_name)

        step = json.loads((checkpoint / "metrics.json").read_text(encoding="utf-8"))["step"]
        scores.append((epoch, step, summary))
        if newly_measured:
            write_epoch_scores(eval_dir, scores)
        LOG.info("Epoch %d, step %d: %d/900 (%.2f%%).",
                 epoch, step, summary["correct"], summary["accuracy_percent"])
    write_epoch_scores(eval_dir, scores)
    LOG.info("All epoch evaluations complete: %s", eval_dir / "accuracy_by_epoch.md")


def run_training(args):
    prompt_for, baseline_vision, baseline_hash = baseline_inputs(BASELINE_REPO)
    vision = baseline_vision if args.baseline_vision else {
        key: SETTINGS[key] for key in ("min_pixels", "max_pixels")
    }
    LOG.info("Reading and checking %s", args.data_dir)
    examples, data_hash = load_examples(args.data_dir, prompt_for)
    train(args.run_dir, examples, data_hash, baseline_hash, args.max_tokens, vision, args.preflight_only)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--max-tokens", type=int, default=SETTINGS["max_tokens"],
                        help="Maximum total sequence length, including the image, question and answer.")
    parser.add_argument("--baseline-vision", action="store_true",
                        help="Use the evaluation image-size bounds instead of the training bounds.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check the data and GPU forward/backward without updating any weights.")
    parser.add_argument("--check-data", action="store_true", help="Validate JSON and images without loading the model.")
    parser.add_argument("--_train-stage", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_merge-epoch", type=int, choices=range(1, SETTINGS["epochs"] + 1), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive.")
    if args.check_data:
        prompt_for, _, _ = baseline_inputs(BASELINE_REPO)
        examples, data_hash = load_examples(args.data_dir, prompt_for)
        print(f"Verified {len(examples)} original+ERA rows and images. SHA-256: {data_hash}")
        return

    import fcntl

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["HF_HOME"] = str(HF_CACHE)
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.run_dir / "train.log", encoding="utf-8")],
    )
    # Internal subprocesses inherit the lock from the parent until their GPU stage exits.
    if args._train_stage:
        run_training(args)
        return
    if args._merge_epoch is not None:
        merge_epoch(args.run_dir, args._merge_epoch)
        return

    with (args.run_dir / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Training or evaluation is already running in {args.run_dir}.") from None
        try:
            if args.preflight_only:
                run_training(args)
                return
            # Separate processes release CUDA memory between training, merging, and evaluation.
            command = [sys.executable, str(Path(__file__).resolve()), "--_train-stage",
                       "--run-dir", str(args.run_dir), "--data-dir", str(args.data_dir.resolve()),
                       "--max-tokens", str(args.max_tokens)]
            if args.baseline_vision:
                command.append("--baseline-vision")
            subprocess.run(command, check=True, pass_fds=(lock.fileno(),))
            evaluate_epochs(args.run_dir, lock.fileno())
        except Exception:
            LOG.exception("Pipeline stopped. Completed checkpoints and evaluations remain in %s.", args.run_dir)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
