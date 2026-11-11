"""Paired ERA data and baseline-faithful, assistant-only multimodal SFT inputs.

Importing this module does not import torch/transformers or load model weights.
The preflight command loads only the pinned processor and runs on CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from .common import (canonical_json_hash, config_digest, load_config,
                     reject_unpinned_processor_overrides, resolve_path, sha256_file, write_json)


_FINAL = re.compile(r"^Final answer: ([A-Z])\s*$", re.MULTILINE)
_ERA_INSTRUCTION = re.compile(
    r"\b(?:use|follow|write|answer|respond|solve|output|format)\b[^\n]{0,100}\bERA\b"
    r"|\bEvidence\s*(?:[:→>]|->)[^\n]{0,120}\bRule\s*(?:[:→>]|->)"
    r"|\bEvidence\s*,\s*Rule\s*(?:,|and)\s*(?:Application|Apply)\b",
    re.IGNORECASE,
)
_CHAT_CONTROL = ("<|im_start|>", "<|im_end|>", "<|vision_start|>",
                 "<|vision_end|>", "<|image_pad|>", "<|video_pad|>")


def _inside(root: Path, value: str, description: str) -> Path:
    """Resolve a package-relative path, including symlinks, without escaping it."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description}: expected a nonempty relative path")
    # Reject Windows drive/UNC paths even when the training host is Linux.
    if Path(value).is_absolute() or re.match(r"^[A-Za-z]:", value) or value.startswith(("\\", "/")):
        raise ValueError(f"{description}: absolute paths are not allowed: {value}")
    candidate = (root / value.replace("\\", "/")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{description}: path escapes dataset root: {value}") from exc
    if not candidate.is_file():
        raise ValueError(f"{description}: missing file: {candidate}")
    return candidate


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def _required_text(row: dict[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: missing/non-string {key}")
    return value


def _target_answer(response: str, prompt: str, where: str) -> str:
    if any(token in response for token in _CHAT_CONTROL):
        raise ValueError(f"{where}: response contains a reserved chat/image token")
    matches = list(_FINAL.finditer(response))
    if len(matches) != 1 or response[matches[0].end():].strip():
        raise ValueError(f"{where}: expected exactly one terminal Final answer: LETTER")
    answer = matches[0].group(1)
    options = re.findall(r"^([A-Z])\. ", prompt, re.MULTILINE)
    if not options or answer not in options:
        raise ValueError(f"{where}: answer {answer} is outside the supplied options")
    if not response[:matches[0].start()].strip():
        raise ValueError(f"{where}: response has no explanation")
    return answer


def load_paired_records(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate the ordered pair of flat HF JSONL exports; retain original prompts.

    Release/review status is recorded, never silently converted into approval.
    A candidate status does not block the non-training CPU preflight.
    """
    settings = config["dataset"]
    root = Path(resolve_path(config, settings["root"])).resolve()
    plain_path = _inside(root, settings["plain_jsonl"], "plain_jsonl")
    era_path = _inside(root, settings["era_jsonl"], "era_jsonl")
    plain, era = _jsonl(plain_path), _jsonl(era_path)
    expected = int(settings.get("expected_items", 200))
    if expected < 1 or len(plain) != expected or len(era) != expected:
        raise ValueError(f"Expected {expected} paired items; plain={len(plain)}, era={len(era)}")
    records, fingerprints, input_fingerprints, seen = [], [], [], set()
    for index, (left, right) in enumerate(zip(plain, era)):
        where = f"row {index + 1}"
        sample_id = _required_text(left, "sample_id", where)
        if sample_id != right.get("sample_id"):
            raise ValueError(f"{where}: ordered sample_id mismatch")
        if sample_id in seen:
            raise ValueError(f"{where}: duplicate sample_id {sample_id}")
        seen.add(sample_id)
        for key in ("input_prompt", "visual_group_id"):
            _required_text(left, key, sample_id)
            if left[key] != right.get(key):
                raise ValueError(f"{sample_id}: {key} differs between arms")
        prompt = left["input_prompt"]
        if _ERA_INSTRUCTION.search(prompt):
            raise ValueError(f"{sample_id}: user input contains an ERA instruction")
        if any(token in prompt for token in _CHAT_CONTROL):
            raise ValueError(f"{sample_id}: user input contains a reserved chat/image token")
        image_left = _inside(root, left.get("image_path"), f"{sample_id} plain image")
        image_right = _inside(root, right.get("image_path"), f"{sample_id} era image")
        if image_left != image_right:
            raise ValueError(f"{sample_id}: image paths differ between arms")
        image_hash = sha256_file(image_left)
        if image_hash != sha256_file(image_right):
            raise ValueError(f"{sample_id}: image bytes differ between arms")
        for arm, row in (("plain", left), ("era", right)):
            if row.get("image_sha256") and row["image_sha256"] != image_hash:
                raise ValueError(f"{sample_id}/{arm}: declared image hash mismatch")
        response_plain = _required_text(left, "response", sample_id)
        response_era = _required_text(right, "response", sample_id)
        answer_plain = _target_answer(response_plain, prompt, f"{sample_id}/plain")
        answer_era = _target_answer(response_era, prompt, f"{sample_id}/era")
        if answer_plain != answer_era:
            raise ValueError(f"{sample_id}: final answers differ between arms")
        if response_plain == response_era:
            raise ValueError(f"{sample_id}: identical target responses do not define two arms")
        headings = [response_era.find(f"{name}:") for name in ("Evidence", "Rule", "Application")]
        if min(headings) < 0 or headings != sorted(headings):
            raise ValueError(f"{sample_id}: ERA target must contain Evidence, Rule, Application in order")
        shared = dict(sample_id=sample_id, visual_group_id=left["visual_group_id"],
                      input_prompt=prompt, image_sha256=image_hash)
        record = dict(shared, image_path=str(image_left),
                      image_relative_path=image_left.relative_to(root).as_posix(),
                      response_plain=response_plain, response_era=response_era,
                      answer_letter=answer_plain)
        records.append(record)
        input_fingerprints.append(shared)
        fingerprints.append(dict(shared, response_plain=response_plain, response_era=response_era))
    status_path = _inside(root, settings["validation_status"], "validation_status")
    with status_path.open(encoding="utf-8-sig") as handle:
        status = json.load(handle)
    if not isinstance(status, dict):
        raise ValueError("validation_status must be a JSON object")
    declared_items = status.get("dataset_items")
    if declared_items is not None and declared_items != expected:
        raise ValueError("validation_status dataset_items disagrees with paired exports")
    manifest = {
        "dataset_root": str(root), "items": len(records),
        "sample_ids": [r["sample_id"] for r in records],
        "visual_groups": len({r["visual_group_id"] for r in records}),
        "ordered_input_sha256": canonical_json_hash(input_fingerprints),
        "ordered_pair_sha256": canonical_json_hash(fingerprints),
        "files": {
            settings["plain_jsonl"]: sha256_file(plain_path),
            settings["era_jsonl"]: sha256_file(era_path),
            settings["validation_status"]: sha256_file(status_path),
        },
        "source_validation_status": status,
        "checks": {"same_ordered_ids": True, "same_input_text": True, "same_image_bytes": True,
                   "same_final_answers": True, "no_era_input_instruction": True,
                   "image_paths_contained": True},
        "scope": "Input/pair integrity only; does not certify target correctness, human review or training success.",
    }
    return records, manifest


class PairedSFTDataset:
    """Minimal map-style dataset; does not pretokenize or hold GPU tensors."""

    def __init__(self, records: list[dict[str, Any]], arm: str):
        if arm not in ("plain", "era"):
            raise ValueError("arm must be plain or era")
        self.records, self.arm = records, arm

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return dict(record, response=record[f"response_{self.arm}"], arm=self.arm)


def _load_image(path: str) -> Any:
    from PIL import Image
    with Image.open(path) as image:
        return image.convert("RGB")


def _plain_list(value: Any) -> list[Any]:
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _one_row(value: Any) -> list[int]:
    values = _plain_list(value)
    if len(values) != 1 or not isinstance(values[0], (list, tuple)):
        raise ValueError("Expected exactly one tensor row (micro batch must be 1)")
    return list(values[0])


def assistant_label_ids(full_ids: list[int], prefix_ids: list[int], attention_mask: list[int],
                        assistant_end_id: int, forbidden_ids: set[int] | None = None) -> list[int]:
    """Mask all input/padding and trailing template whitespace, retain assistant end."""
    if len(full_ids) != len(attention_mask) or full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Assistant generation prefix is not an exact token prefix of the SFT example")
    suffix = full_ids[len(prefix_ids):]
    positions = [i + len(prefix_ids) for i, token in enumerate(suffix) if token == assistant_end_id]
    if len(positions) != 1:
        raise ValueError("Expected one normal assistant end token after the prompt prefix")
    end = positions[0]
    if end <= len(prefix_ids):
        raise ValueError("Assistant target has no body tokens")
    if any(token in (forbidden_ids or set()) for token in full_ids[len(prefix_ids):end]):
        raise ValueError("Assistant target unexpectedly contains an image/chat-control token")
    if any(not attention_mask[i] for i in range(len(prefix_ids), end + 1)):
        raise ValueError("Assistant body/end was marked as padding")
    return [token if len(prefix_ids) <= i <= end and attention_mask[i] else -100
            for i, token in enumerate(full_ids)]


class AssistantOnlyCollator:
    """Encode one example using exactly the baseline vision path; no truncation."""

    def __init__(self, processor: Any, config: dict[str, Any], *,
                 vision_processor: Callable[..., Any] | None = None,
                 image_loader: Callable[[str], Any] | None = None):
        self.processor, self.config = processor, config
        self.settings = config["training"]
        if self.settings.get("micro_batch_size", 1) != 1:
            raise ValueError("This collator requires micro_batch_size=1")
        self.vision_processor = vision_processor
        self.image_loader = image_loader or _load_image

    def encode(self, feature: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        processor, settings = self.processor, self.settings
        sample_id = feature["sample_id"]
        response = _required_text(feature, "response", sample_id)
        if any(token in response for token in _CHAT_CONTROL):
            raise ValueError(f"{sample_id}: reserved token in response")
        image = self.image_loader(feature["image_path"])
        vision = {key: int(settings[key]) for key in ("min_pixels", "max_pixels")}
        messages = [{"role": "user", "content": [
            dict(type="image", image=image, **vision),
            {"type": "text", "text": feature["input_prompt"]},
        ]}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(messages + [{"role": "assistant", "content": response}],
                                             tokenize=False, add_generation_prompt=False)
        if not full.startswith(prefix):
            raise ValueError(f"{sample_id}: chat template changed the inference prefix")
        process_vision_info = self.vision_processor
        if process_vision_info is None:
            from qwen_vl_utils import process_vision_info
        images, videos, kwargs = process_vision_info(
            messages, image_patch_size=processor.image_processor.patch_size,
            return_video_kwargs=True, return_video_metadata=True)
        if videos is not None:
            raise ValueError(f"{sample_id}: unexpected video input")
        kwargs = dict(kwargs, do_resize=False)
        encoding_kwargs = dict(images=images, return_tensors="pt", padding=False, truncation=False, **kwargs)
        prefix_encoded = processor(text=[prefix], **encoding_kwargs)
        encoded = processor(text=[full], **encoding_kwargs)
        prefix_ids, ids = _one_row(prefix_encoded["input_ids"]), _one_row(encoded["input_ids"])
        if len(ids) > int(settings["max_seq_length"]):
            raise ValueError(f"{sample_id}: {len(ids)} tokens exceeds max_seq_length={settings['max_seq_length']}; no truncation")
        grid = _plain_list(encoded["image_grid_thw"])
        prefix_grid = _plain_list(prefix_encoded["image_grid_thw"])
        if grid != prefix_grid:
            raise ValueError(f"{sample_id}: image grid differs between inference prefix and SFT sequence")
        tokenizer = processor.tokenizer
        end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(end_id, int) or end_id == getattr(tokenizer, "unk_token_id", None):
            raise ValueError("Pinned tokenizer does not expose a normal <|im_end|> token")
        forbidden = {tokenizer.convert_tokens_to_ids(token) for token in _CHAT_CONTROL if token != "<|im_end|>"}
        forbidden.discard(None)
        forbidden.discard(getattr(tokenizer, "unk_token_id", None))
        attention = _one_row(encoded["attention_mask"])
        labels = assistant_label_ids(ids, prefix_ids, attention, end_id, forbidden)
        # Some templates append a newline after im_end. It must not become target loss.
        last_target = max(i for i, value in enumerate(labels) if value != -100)
        if tokenizer.decode(ids[last_target + 1:], skip_special_tokens=False).strip():
            raise ValueError(f"{sample_id}: non-whitespace template content follows assistant end")
        encoded["labels"] = encoded["input_ids"].new_tensor([labels])
        # Trainer must receive only model tensors, never metadata, paths, or targets.
        batch = {key: value for key, value in encoded.items()
                 if key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels")}
        report = {
            "sample_id": sample_id, "arm": feature.get("arm"),
            "input_prompt_sha256": canonical_json_hash(feature["input_prompt"]),
            "rendered_prefix_sha256": canonical_json_hash(prefix),
            "prefix_token_ids_sha256": canonical_json_hash(prefix_ids),
            "input_tokens": len(prefix_ids), "sequence_tokens": len(ids),
            "supervised_tokens": sum(value != -100 for value in labels),
            "assistant_end_token_id": end_id, "assistant_end_in_loss": labels[last_target] == end_id,
            "image_grid_thw": grid,
            "image_sha256": feature.get("image_sha256"),
            "processed_image_sizes_wh": [list(im.size) for im in images],
        }
        return batch, report

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise ValueError("Exactly one example per micro batch is required")
        return self.encode(features[0])[0]


def validate_encoded_pair(record: dict[str, Any], processor: Any, config: dict[str, Any],
                          *, collator: AssistantOnlyCollator | None = None) -> dict[str, Any]:
    collator = collator or AssistantOnlyCollator(processor, config)
    reports = {}
    for arm in ("plain", "era"):
        _, reports[arm] = collator.encode(dict(record, response=record[f"response_{arm}"], arm=arm))
    for field in ("input_prompt_sha256", "rendered_prefix_sha256", "prefix_token_ids_sha256",
                  "input_tokens", "image_grid_thw", "image_sha256", "processed_image_sizes_wh"):
        if reports["plain"][field] != reports["era"][field]:
            raise ValueError(f"{record['sample_id']}: encoded pair differs in {field}")
    return {"sample_id": record["sample_id"], "paired_prefix_and_image_grid_equal": True, **reports}


def validate_processor_files(path: str | Path) -> dict[str, str]:
    """Return hashes only for the exact frozen processor and model metadata."""
    from .evaluate import AUX_HASHES, BASE_CONFIG_SHA256
    source = Path(path)
    reject_unpinned_processor_overrides(source)
    actual = {}
    for filename, expected in {**AUX_HASHES, "config.json": BASE_CONFIG_SHA256}.items():
        path = source / filename
        if not path.is_file():
            raise ValueError(f"Missing frozen baseline processor file: {filename}")
        actual[filename] = sha256_file(path)
        if actual[filename] != expected:
            raise ValueError(f"Processor/tokenizer differs from the frozen baseline: {filename}")
    return actual


def processor_source(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Verify the exact tokenizer/template/vision files from the baseline run."""
    from .evaluate import AUX_HASHES
    model = config["model"]
    local_path = model.get("local_path")
    if local_path:
        source = Path(resolve_path(config, local_path))
        if not source.is_dir():
            raise ValueError(f"Pinned model/processor local_path does not exist: {source}")
    else:
        from huggingface_hub import snapshot_download
        # Explicit file allowlist: this CPU preflight must never download weights.
        source = Path(snapshot_download(repo_id=model["repo"], revision=model["revision"],
                                        allow_patterns=list(AUX_HASHES) + ["config.json"]))
    actual = validate_processor_files(source)
    return source, {"verified_against_baseline": True, "repo": model["repo"],
                    "revision": model["revision"], "source": str(source), "sha256": actual,
                    "artifact_set_sha256": canonical_json_hash(actual)}


def load_processor(config: dict[str, Any]) -> Any:
    source, provenance = processor_source(config)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(str(source), trust_remote_code=False, local_files_only=True)
    processor._era_provenance = provenance
    return processor


def run_preflight(config: dict[str, Any], output: str | Path,
                  *, processor: Any | None = None) -> dict[str, Any]:
    records, manifest = load_paired_records(config)
    processor = processor if processor is not None else load_processor(config)
    pairs = [validate_encoded_pair(record, processor, config) for record in records]
    summary = {}
    for arm in ("plain", "era"):
        summary[arm] = {}
        for field in ("input_tokens", "sequence_tokens", "supervised_tokens"):
            values = sorted(pair[arm][field] for pair in pairs)
            summary[arm][field] = {"min": min(values), "max": max(values),
                                   "mean": sum(values) / len(values),
                                   "p95": values[max(0, math.ceil(len(values) * .95) - 1)],
                                   "total": sum(values)}
    result = {"schema_version": 1, "status": "pass", "operation": "cpu_data_preflight",
              "model_weights_loaded": False, "gpu_forward_executed": False,
              "config_sha256": config_digest(config), "model": config.get("model"),
              "processor": getattr(processor, "_era_provenance", {
                  "verified_against_baseline": False, "reason": "Externally supplied processor; source hashes not verified by load_processor"}),
              "dataset": manifest, "vision": {k: config["training"][k] for k in ("min_pixels", "max_pixels")},
              "max_seq_length": config["training"]["max_seq_length"],
              "summary": summary, "pairs": pairs,
              "scope": "Actual processor/chat template, image grids, token prefix and assistant/end loss masks only; no model training or evaluation."}
    write_json(Path(output), result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU-only paired SFT processor preflight (no model weights/GPU forward)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output = Path(resolve_path(config, args.output))
    result = run_preflight(config, output)
    print(json.dumps({"status": result["status"], "items": result["dataset"]["items"],
                      "report": str(output), "summary": result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
