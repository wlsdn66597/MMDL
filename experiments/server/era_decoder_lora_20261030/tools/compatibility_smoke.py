"""CPU-only API/serialization smoke check using a tiny random Qwen3-VL.

This never reads the pilot dataset or downloads a model, performs no backward
pass or optimizer step, and never invokes Trainer.train(). It is distinct from
the real model's processor preflight and a GPU memory check.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import tempfile

# These must be set before importing torch, Transformers or PEFT.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from era_lora.common import write_json
from era_lora.merge import tied_state_groups, tensor_hashes, validate_saved_weights
from era_lora.train import (assert_trainables, discover_decoder_targets,
                            make_audit_callback, parameter_digest, utc_now)


def run(output: Path) -> dict:
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "accelerate", "safetensors")}
    for name, expected in {"torch": "2.8.0", "transformers": "4.57.0",
                           "peft": "0.17.1", "accelerate": "1.10.1"}.items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {versions[name]}")
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import (Qwen3VLConfig, Qwen3VLForConditionalGeneration,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)

    torch.set_num_threads(1)
    configuration = Qwen3VLConfig(
        text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "head_dim": 8, "max_position_embeddings": 128,
                     "tie_word_embeddings": True, "bos_token_id": 1, "eos_token_id": 2,
                     "pad_token_id": 0,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2],
                                      "mrope_interleaved": True}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "in_channels": 3, "patch_size": 16,
                       "spatial_merge_size": 2, "temporal_patch_size": 2,
                       "out_hidden_size": 32, "num_position_embeddings": 16,
                       "deepstack_visual_indexes": [0, 1]},
        image_token_id=120, video_token_id=121, vision_start_token_id=122,
        vision_end_token_id=123, tie_word_embeddings=True)
    configuration._attn_implementation = "sdpa"
    set_seed(3407)
    model = Qwen3VLForConditionalGeneration(configuration).to(dtype=torch.bfloat16, device="cpu")
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count >= 1_000_000:
        raise RuntimeError("Tiny compatibility model unexpectedly exceeds one million parameters")
    baseline_hashes = tensor_hashes(model)
    model.requires_grad_(False)
    targets = discover_decoder_targets(model)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, target_modules=targets,
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        modules_to_save=None, init_lora_weights=True))
    trainables = assert_trainables(model, targets)
    if any(p.device.type != "cpu" for p in model.parameters()):
        raise RuntimeError("Compatibility smoke must remain entirely on CPU")
    initial_digest = parameter_digest(model, True)
    zero_delta_count = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, "get_delta_weight"):
                if torch.count_nonzero(module.get_delta_weight("default")).item() != 0:
                    raise RuntimeError(f"Default LoRA initialization is not a zero update: {name}")
                zero_delta_count += 1
    if zero_delta_count != len(targets):
        raise RuntimeError("Not every decoder target has a verified zero-initialized LoRA update")
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if not hasattr(TrainerCallback, "on_pre_optimizer_step"):
        raise RuntimeError("Installed Trainer lacks the required first-gradient audit callback")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tiny-compatibility-", dir=output.parent) as temporary:
        work = Path(temporary)
        arguments = TrainingArguments(
            output_dir=str(work / "trainer"), use_cpu=True, bf16=False, fp16=False,
            num_train_epochs=2, per_device_train_batch_size=1, gradient_accumulation_steps=16,
            learning_rate=1e-4, weight_decay=0.01, warmup_ratio=0.03,
            lr_scheduler_type="cosine", max_grad_norm=1.0, adam_beta1=0.9,
            adam_beta2=0.999, adam_epsilon=1e-8, optim="adamw_torch",
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            seed=3407, data_seed=3407, dataloader_num_workers=0, dataloader_drop_last=False,
            remove_unused_columns=False, save_strategy="epoch", save_safetensors=True,
            save_only_model=False, eval_strategy="no", logging_strategy="steps",
            logging_steps=1, report_to=[], load_best_model_at_end=False,
            dataloader_pin_memory=False)

        def never_collate(features):
            raise AssertionError("This smoke check must not iterate a training dataset")

        trainer = Trainer(model=model, args=arguments, train_dataset=[{"synthetic_unused": True}],
                          data_collator=never_collate,
                          callbacks=[make_audit_callback(work / "trainer", {"smoke": True},
                                                         parameter_digest(model, False))])
        trainer.model_accepts_loss_kwargs = False
        if trainer.optimizer is not None or trainer.state.global_step != 0:
            raise RuntimeError("Trainer unexpectedly created an optimizer or advanced training")
        if parameter_digest(model, True) != initial_digest:
            raise RuntimeError("Trainer construction changed the initial adapter")
        adapter = work / "adapter"
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        set_seed(3407)
        fresh_configuration = Qwen3VLConfig.from_dict(configuration.to_dict())
        fresh_configuration._attn_implementation = "sdpa"
        fresh = Qwen3VLForConditionalGeneration(fresh_configuration).to(dtype=torch.bfloat16, device="cpu")
        if tensor_hashes(fresh) != baseline_hashes:
            raise RuntimeError("Same-seed fresh tiny base did not reproduce its original tensor state")
        loaded = PeftModel.from_pretrained(fresh, str(adapter), is_trainable=False)
        merged = loaded.merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
        merged.config.use_cache = True
        if tensor_hashes(merged) != baseline_hashes:
            raise RuntimeError("Merging a zero-update adapter changed the base tensors")
        merged_path = work / "merged"
        merged.save_pretrained(merged_path, safe_serialization=True)
        serialized = validate_saved_weights(merged_path, set(baseline_hashes), tied_state_groups(merged))
        restored = Qwen3VLForConditionalGeneration.from_pretrained(
            str(merged_path), dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True, use_safetensors=True)
        if tensor_hashes(restored) != baseline_hashes:
            raise RuntimeError("Tiny BF16 checkpoint changed after safe serialization and reload")
        report = {"checked_at": utc_now(), "passed": True, "versions": versions,
                  "scope": "tiny random CPU architecture/API/zero-merge serialization check only",
                  "actual_pilot_training_started": False, "optimizer_created": False,
                  "optimizer_steps": 0, "backward_passes": 0, "real_model_downloaded": False,
                  "cpu_only": True, "base_parameter_count": parameter_count,
                  "decoder_targets": targets, "trainable_parameter_count": sum(p["numel"] for p in trainables),
                  "initial_adapter_sha256": initial_digest,
                  "zero_delta_modules": zero_delta_count,
                  "zero_merge_preserves_all_base_tensors": True,
                  "serialization_roundtrip_exact": True, "serialized_validation": serialized,
                  "limitations": ["No full 4B model load, actual multimodal input, backward pass or VRAM check.",
                                  "Does not prove actual fine-tuning or evaluation succeeds."]}
    write_json(output, report)
    return report


def main() -> int:
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "compatibility_smoke.json")
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
