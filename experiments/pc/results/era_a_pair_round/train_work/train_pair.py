"""Matched positive ERA SFT, with or without an Application-only preference loss."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_work"
OUT = ROOT / "models"
PREVIOUS = Path.home() / "era_single_call_campaign"
sys.path.insert(0, str(PREVIOUS))
import train_retention as donor
import train_round as common
sys.path.insert(0, str(common.PROJECT))
from era_lora.common import canonical_json_hash, write_json
from era_lora.data import AssistantOnlyCollator, load_processor
from era_lora.train import file_lock

LR = 1e-5
PAIR_WEIGHT = 0.5
PAIR_SCALE = 2.0
PAIR_MARGIN = 1.0


def train_rows():
    receipt = common.read(DATA / "receipt.json")
    path = DATA / "train.jsonl"
    assert receipt["verified"] is True and common.sha(path) == receipt["train_sha256"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    assert len(rows) == len({row["id"] for row in rows}) == receipt["n"] and rows
    for row in rows:
        for field in ("evidence", "rule", "chosen_application", "rejected_application", "final_answer"):
            assert isinstance(row[field], str) and row[field] == row[field].strip() and row[field] and "\n" not in row[field], (row["id"], field)
        assert row["chosen_application"] != row["rejected_application"]
        image = (DATA / row["image"]).resolve()
        assert image.is_relative_to((DATA / "images").resolve())
        assert common.sha(image) == row["image_sha256"]
    return rows


def encode(row, application, processor, collator):
    """Exclude boundary-crossing BPEs so preference never scores a header or final."""
    import torch

    prefix = f"Evidence: {row['evidence']}\nRule: {row['rule']}\nApplication:"
    end = len(prefix) + 1 + len(application)
    response = f"{prefix} {application}\nFinal answer: {row['final_answer']}"
    feature = dict(sample_id=row["id"], image_path=str(DATA / row["image"]),
                   image_sha256=row["image_sha256"], input_prompt=row["prompt"], response=response)
    batch, report = collator.encode(feature)
    tokens = processor.tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    start = report["input_tokens"]
    pos = batch.pop("labels")[0].ne(-100).nonzero().flatten()
    assert pos.tolist() == list(range(start, start + len(tokens.input_ids) + 1))
    assert batch["input_ids"][0, pos[:-1]].tolist() == tokens.input_ids
    assert batch["input_ids"][0, pos[-1]].item() == report["assistant_end_token_id"]
    selected = [i for i, (a, b) in enumerate(tokens.offset_mapping) if len(prefix) <= a < b <= end]
    assert selected and selected == list(range(selected[0], selected[-1] + 1)), row["id"]
    app_indices = torch.tensor(selected, dtype=torch.long)
    return batch, pos, app_indices


def check_pair(positive, negative, sample_id):
    import torch

    pb, pp, pi = positive
    nb, np, ni = negative
    # Compare the literal model inputs preceding A in addition to the source strings.
    assert torch.equal(pb["input_ids"][0, :pp[pi[0]]], nb["input_ids"][0, :np[ni[0]]]), sample_id
    assert torch.equal(pb["image_grid_thw"], nb["image_grid_thw"]) and torch.equal(pb["pixel_values"], nb["pixel_values"]), sample_id
    assert not torch.equal(pb["input_ids"][0, pp[pi]], nb["input_ids"][0, np[ni]]), sample_id


def token_losses(model, accelerator, batch, positions):
    import torch.nn.functional as F

    labels = batch["input_ids"][0, positions].to(accelerator.device)
    inputs = {key: value.to(accelerator.device) for key, value in batch.items()}
    with accelerator.autocast():
        logits = model(**inputs, logits_to_keep=(positions - 1).to(accelerator.device)).logits[0]
        assert logits.shape[0] == len(labels)
        return F.cross_entropy(logits.float(), labels, reduction="none")


def pair_objective(pos_nll, neg_nll):
    import torch.nn.functional as F

    # Signed margin in log likelihood, normalized for different lengths.
    gap = neg_nll.mean() - pos_nll.mean()
    return F.softplus(PAIR_MARGIN - PAIR_SCALE * gap), gap


def preflight(rows, processor, collator):
    import torch

    pos = torch.tensor([1., 1.], requires_grad=True)
    neg = torch.tensor([2., 2., 2.], requires_grad=True)
    pref, gap = pair_objective(pos, neg)
    pref.backward()
    assert torch.all(pos.grad > 0) and torch.all(neg.grad < 0) and gap.item() == 1.
    audit = []
    for row in rows:
        chosen = encode(row, row["chosen_application"], processor, collator)
        rejected = encode(row, row["rejected_application"], processor, collator)
        check_pair(chosen, rejected, row["id"])
        audit.append([row["id"], len(chosen[1]), len(chosen[2]), len(rejected[2])])
    result = dict(verified=True, train_sha256=common.sha(DATA / "train.jsonl"), code_sha256=common.sha(Path(__file__)),
                  n=len(rows), prefix_and_image_equal=True, pair_targets_differ=True, analytic_gradient_sign_checked=True,
                  columns=["id", "supervised_positive_tokens", "chosen_A_tokens", "rejected_A_tokens"], rows=audit)
    write_json(ROOT / "train_work/preflight.json", result)
    print(f"Verified {len(audit)} tokenized pairs, including images, prefixes, A-only score masks and EOS.", flush=True)


def train(arm, cfg, base, donor_record, rows, processor, collator):
    import torch
    from accelerate import Accelerator
    from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import Qwen3VLForConditionalGeneration, get_scheduler, set_seed
    from era_lora.train import (assert_trainables, discover_decoder_targets, parameter_digest,
                                runtime_check, validate_gradient_ownership)

    pre = common.read(ROOT / "train_work/preflight.json")
    assert pre["verified"] and pre["n"] == len(rows) and pre["train_sha256"] == common.sha(DATA / "train.jsonl")
    assert pre["code_sha256"] == common.sha(Path(__file__))
    helper = common.trainer_module()
    out = OUT / arm
    runtime = runtime_check()
    settings = cfg["training"]
    seed, accumulation = settings["seed"], settings["gradient_accumulation_steps"]
    assert seed == 3407 and accumulation == 16
    indices = helper.sample_schedule(len(rows), 1, seed)[0]
    order = [rows[index]["id"] for index in indices]
    steps = math.ceil(len(rows) / accumulation)
    identity = dict(arm=arm, starting_weights=common.HASHES, donor=donor_record,
                    data_sha256=pre["train_sha256"], receipt_sha256=common.sha(DATA / "receipt.json"),
                    preflight_sha256=common.sha(ROOT / "train_work/preflight.json"),
                    code_sha256=common.sha(Path(__file__)), helper_sha256=common.sha(Path(helper.__file__)),
                    config_sha256=common.sha(common.CFG), sequence_sha256=canonical_json_hash(order),
                    seed=seed, epochs=1, steps=steps, accumulation=accumulation, lr=LR,
                    pair_weight=PAIR_WEIGHT if arm == "candidate" else 0.,
                    pair_margin=PAIR_MARGIN, pair_scale=PAIR_SCALE, software=runtime["versions"])
    if (out / "identity.json").exists():
        assert common.read(out / "identity.json") == identity
    else:
        write_json(out / "identity.json", identity)
    if (out / "manifest.json").exists():
        manifest = common.read(out / "manifest.json")
        assert manifest["complete"] and manifest["identity"] == identity
        assert common.sha(out / "adapter/adapter_model.safetensors") == manifest["adapter_file_sha256"]
        return

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    set_seed(seed)
    accel = Accelerator(mixed_precision="bf16")
    assert accel.num_processes == 1 and accel.scaler is None
    backbone = Qwen3VLForConditionalGeneration.from_pretrained(
        str(base), dtype=torch.bfloat16, attn_implementation=settings["attn_implementation"],
        local_files_only=True, low_cpu_mem_usage=True, use_safetensors=True)
    backbone.requires_grad_(False)
    targets = discover_decoder_targets(backbone)
    model = PeftModel.from_pretrained(backbone, str(donor.DONOR), is_trainable=True)
    allowed = assert_trainables(model, targets)
    saved = load_file(str(donor.DONOR / "adapter_model.safetensors"), device="cpu")
    installed = get_peft_model_state_dict(model)
    assert set(saved) == set(installed) and len(saved) == 288
    assert all(torch.equal(tensor, installed[name].detach().cpu()) for name, tensor in saved.items())
    del saved, installed
    frozen, initial = parameter_digest(model, False), parameter_digest(model, True)
    matched = dict(initial_adapter=initial, frozen=frozen, sequence=identity["sequence_sha256"],
                   train=identity["data_sha256"], seed=seed, steps=steps, lr=LR)
    with file_lock(OUT / ".initial.lock"):
        path = OUT / "matched_init.json"
        if path.exists():
            assert common.read(path) == matched
        else:
            write_json(path, matched)
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=LR, weight_decay=settings["weight_decay"])
    scheduler = get_scheduler(settings["lr_scheduler_type"], optimizer,
                              num_warmup_steps=math.ceil(steps * settings["warmup_ratio"]), num_training_steps=steps)
    model, optimizer = accel.prepare(model, optimizer)
    raw = accel.unwrap_model(model)
    folder = out / "checkpoints"
    latest = helper.select_checkpoint(folder, identity)
    step, prior, seen = 0, 0., []
    totals = dict(positive_tokens=0, chosen_A_tokens=0, rejected_A_tokens=0, compared=0, preferred=0, gap_sum=0.)
    if latest:
        cursor = common.read(latest / "cursor.json")
        step, prior, seen, totals = cursor["step"], cursor["training_seconds"], cursor["seen_ids"], cursor["totals"]
        assert seen == order[:min(step * accumulation, len(order))]
        restored = set_peft_model_state_dict(raw, load_file(str(latest / "adapter_model.safetensors")))
        assert not restored.unexpected_keys and parameter_digest(raw, True) == cursor["adapter_digest"]
        optimizer.load_state_dict(torch.load(latest / "optimizer.pt", map_location="cpu", weights_only=True))
        scheduler.load_state_dict(torch.load(latest / "scheduler.pt", map_location="cpu", weights_only=True))
        assert scheduler.last_epoch == step
        helper.restore_rng(torch.load(latest / "rng.pt", map_location="cpu", weights_only=False))
    tic = time.monotonic()
    elapsed = prior
    while step < steps:
        group = indices[step * accumulation:(step + 1) * accumulation]
        optimizer.zero_grad(set_to_none=True)
        for idx in group:
            row = rows[idx]
            positive = encode(row, row["chosen_application"], processor, collator)
            batch, positions, app = positive
            ce = token_losses(model, accel, batch, positions)
            loss = ce.mean()
            totals["positive_tokens"] += len(positions)
            if arm == "candidate":
                negative = encode(row, row["rejected_application"], processor, collator)
                check_pair(positive, negative, row["id"])
                neg_batch, neg_positions, neg_app = negative
                # Preserve the positive-only arm's RNG stream for subsequent chosen forwards.
                with torch.random.fork_rng(devices=[accel.device]):
                    neg_ce = token_losses(model, accel, neg_batch, neg_positions[neg_app])
                pref, gap = pair_objective(ce[app.to(accel.device)], neg_ce)
                loss = loss + PAIR_WEIGHT * pref
                totals["chosen_A_tokens"] += len(app)
                totals["rejected_A_tokens"] += len(neg_app)
                totals["compared"] += 1
                totals["gap_sum"] += gap.detach().item()
                totals["preferred"] += int(gap.detach().item() > 0)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss for {row['id']}")
            accel.backward(loss / len(group))
            seen.append(row["id"])
        if step == 0:
            write_json(out / "gradient_audit.json", validate_gradient_ownership(raw))
        accel.clip_grad_norm_(model.parameters(), settings["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        step += 1
        elapsed = prior + time.monotonic() - tic
        progress = dict(step=step, steps=steps, seen=len(seen), training_seconds=elapsed, eta_seconds=elapsed / step * (steps-step))
        write_json(out / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        if step % 5 == 0 or step == steps:
            cursor = dict(identity=identity, step=step, seen_ids=seen, training_seconds=elapsed,
                          totals=totals, adapter_digest=parameter_digest(raw, True))
            helper.checkpoint(folder, raw, optimizer, scheduler, cursor)
    assert seen == order and parameter_digest(raw, False) == frozen and parameter_digest(raw, True) != initial
    assert totals["compared"] == (len(rows) if arm == "candidate" else 0)
    helper.save_final(out, helper.select_checkpoint(folder, identity))
    write_json(out / "seen_ids.json", seen)
    write_json(out / "manifest.json", dict(complete=True, arm=arm, identity=identity, actual_steps=step,
               seen=len(seen), sequence_sha256=identity["sequence_sha256"], initial_adapter=initial,
               final_adapter=parameter_digest(raw, True), adapter_file_sha256=common.sha(out / "adapter/adapter_model.safetensors"),
               trainable=sum(entry["numel"] for entry in allowed), frozen_verified=True,
               objective_tokens=totals, training_seconds=elapsed, environment=runtime))


def merge(out, base, cfg, donor_record):
    import torch
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    from era_lora.merge import copy_processor_files

    manifest = common.read(out / "manifest.json")
    assert manifest["complete"] and manifest["identity"]["donor"] == donor_record
    assert manifest["identity"]["code_sha256"] == common.sha(Path(__file__))
    adapter_sha = common.sha(out / "adapter/adapter_model.safetensors")
    assert adapter_sha == manifest["adapter_file_sha256"]
    dest = out / "merged"
    if not dest.exists():
        original = Qwen3VLForConditionalGeneration.from_pretrained(
            str(base), dtype=torch.bfloat16, attn_implementation=cfg["training"]["attn_implementation"],
            local_files_only=True, low_cpu_mem_usage=True, use_safetensors=True)
        model = PeftModel.from_pretrained(original, str(out / "adapter"), is_trainable=False)
        merged = model.eval().merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
        assert not any("lora_" in name for name, _ in merged.named_parameters())
        merged.config.use_cache = True
        temp = out / (".merged." + uuid.uuid4().hex + ".partial")
        merged.save_pretrained(temp, safe_serialization=True, max_shard_size="4GB")
        copy_processor_files(base, temp)
        weights = {p.name: common.sha(p) for p in temp.glob("*.safetensors")}
        assert weights
        write_json(temp / "artifact_manifest.json", dict(weights=weights, parent=common.HASHES, adapter=adapter_sha))
        temp.rename(dest)
    meta = common.read(dest / "artifact_manifest.json")
    assert meta["parent"] == common.HASHES and meta["adapter"] == adapter_sha
    assert meta["weights"] == {p.name: common.sha(p) for p in dest.glob("*.safetensors")}
    write_json(out / "merge.json", meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "candidate"))
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.preflight:
        assert not args.arm and not args.merge
        cfg = common.read(common.CFG)
        cfg["_config_dir"] = str(common.PROJECT)
        processor = load_processor(cfg)
        preflight(train_rows(), processor, AssistantOnlyCollator(processor, cfg))
        return
    assert args.arm
    out = OUT / args.arm
    out.mkdir(parents=True, exist_ok=True)
    with file_lock(out / ".writer.lock"):
        cfg, base = common.baseline()
        verified = donor.verify_donor()
        if args.merge:
            merge(out, base, cfg, verified)
        else:
            cfg["_config_dir"] = str(common.PROJECT)
            processor = load_processor(cfg)
            train(args.arm, cfg, base, verified, train_rows(), processor, AssistantOnlyCollator(processor, cfg))


if __name__ == "__main__":
    main()
