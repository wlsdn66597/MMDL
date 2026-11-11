"""Export one R9 decoder LoRA at strength 1/2 using the original PEFT BF16 merge."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
R9 = ROOT.parent / "era_a_only_round"
PROJECT = Path.home() / "era_decoder_lora_20261030"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
BASE = Path.home() / "mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots" / REVISION
BASE_HASHES = {
    "model-00001-of-00002.safetensors": "30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9",
    "model-00002-of-00002.safetensors": "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6",
}
TRAIN_RESULT_HASH = "956c9ebbee19343903d45de2a39d75adc09a1e3f1e1ade6c590e4921f966b052"
ARMS = ("ordinary", "enhanced")


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert read(path) == obj, path
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf8")
    tmp.replace(path)


def weights(folder):
    return {p.name: sha(p) for p in sorted(folder.glob("*.safetensors"))}


def source(arm):
    path = R9 / "train/training_result.json"
    assert sha(path) == TRAIN_RESULT_HASH
    tr = read(path)
    assert tr["verified"] and tr["base_revision"] == REVISION and tr["arms"][arm]["updates"] == 38
    folder = R9 / "train/models/r9" / arm
    proof = read(folder / "merge.json")
    assert proof["weights"] == tr["arms"][arm]["weights"] and proof["parent"] == BASE_HASHES
    assert sha(folder / "adapter/adapter_model.safetensors") == proof["adapter"] == read(folder/"manifest.json")["adapter_file_sha256"]
    return folder


def tensordigest(tensor):
    import torch
    view = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(view).cast("B")).hexdigest()


def export(arm, folder, staging):
    import torch
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    sys.path.insert(0, str(PROJECT))
    from era_lora.train import discover_decoder_targets
    from era_lora.merge import copy_processor_files, validate_merged_state

    cfg = read(PROJECT / "configs/era_4epoch_20261030.json")
    assert weights(BASE) == BASE_HASHES
    model = Qwen3VLForConditionalGeneration.from_pretrained(str(BASE), dtype=torch.bfloat16,
        attn_implementation=cfg["training"]["attn_implementation"], local_files_only=True,
        low_cpu_mem_usage=True, use_safetensors=True)
    targets = discover_decoder_targets(model)
    expected = set(targets)
    before = {k:tensordigest(v) for k,v in model.state_dict().items()}
    adapter_cfg = read(folder / "adapter/adapter_config.json")
    configured = set(adapter_cfg["target_modules"])
    # PEFT can save q_proj etc. as suffixes even when configured with full paths.
    # Verify all actual attached modules against the complete decoder paths below.
    assert configured and all(any(n == t or n.endswith("." + t) for t in configured) for n in expected)
    assert adapter_cfg["bias"] == "none"
    assert not adapter_cfg.get("use_dora") and not adapter_cfg.get("modules_to_save")
    adapted = PeftModel.from_pretrained(model, str(folder / "adapter"), is_trainable=False).eval()
    modules = dict(adapted.get_base_model().named_modules())
    assert {n for n,m in modules.items() if hasattr(m,"lora_A")} == expected
    after_expected = {}
    max_delta = 0.0
    with torch.inference_mode():
        for name in targets:
            module = modules[name]
            scale = module.scaling["default"]
            assert scale > 0
            full = module.get_delta_weight("default")
            module.scaling["default"] = scale * 0.5
            half = module.get_delta_weight("default")
            assert half.dtype == torch.float32 and torch.equal(half, full * 0.5), name
            peak = half.abs().amax().item()
            assert peak > 0 and torch.isfinite(half).all(), name
            max_delta = max(max_delta, peak)
            w = module.get_base_layer().weight
            # PEFT versions may cast the update to BF16 before the add, or cast after it.
            after_expected[name+".weight"] = {
                tensordigest((w.float() + half).to(w.dtype)), tensordigest(w + half.to(w.dtype))}
        del half, full
        merged = adapted.merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
        merged.config.use_cache = True
        after = {k:tensordigest(v) for k,v in merged.state_dict().items()}
        structure = validate_merged_state(before, after, targets)
        assert len(structure["changed_target_weights"]) == len(targets)
        assert all(after[n] in matches for n,matches in after_expected.items())
        merged.save_pretrained(staging, safe_serialization=True, max_shard_size="4GB")
    processor = copy_processor_files(BASE, staging)
    return dict(arm=arm, factor=0.5, base_revision=REVISION, parent=BASE_HASHES,
        r9_training_result_sha256=TRAIN_RESULT_HASH, adapter=sha(folder/"adapter/adapter_model.safetensors"),
        adapter_config=sha(folder/"adapter/adapter_config.json"), processor=processor, weights=weights(staging),
        pre_bf16_exact_half_targets=len(targets), bf16_export_checked_targets=len(after_expected),
        max_abs_float32_half_delta=max_delta, unchanged_non_target_tensors=len(before)-len(targets),
        model_tensor_count=len(before))


def remove_artifacts(folder, arm, complete):
    expected_parent = ROOT/"models"/arm
    assert folder.parent == expected_parent and folder.name in (".building","merged")
    assert not folder.is_symlink() and folder.resolve() == folder
    assert read(folder / ".half-export-owner.json") == dict(arm=arm, path=str(expected_parent), factor=0.5)
    files = [p for p in folder.rglob("*") if p.is_file()]
    assert all(not p.is_symlink() and p.resolve().is_relative_to(folder) for p in folder.rglob("*"))
    if complete:
        proof = read(folder/"half-export.json")
        assert proof == read(ROOT/"scale_receipts"/(arm+".json")) and weights(folder) == proof["weights"]
        allowed = set(proof["processor"]) | set(proof["weights"]) | {
            ".half-export-owner.json","half-export.json","model.safetensors.index.json","generation_config.json"}
        assert all(p.relative_to(folder).as_posix() in allowed for p in files)
    # This exact directory was created by the exporter; R9 adapters and models are outside it.
    assert files
    shutil.rmtree(folder)


def main(arm, retire=False, stage="diagnostic"):
    parent = ROOT/"models"/arm
    parent.mkdir(exist_ok=True, parents=True)
    destination = parent/"merged"
    staging = parent/".building"
    receipt = ROOT/"scale_receipts"/(arm+".json")
    with (ROOT/"models/.half-export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if retire:
            runroot = ROOT/"confirm" if stage=="confirm" else ROOT
            run = runroot/"eval/runs"/arm
            journal = runroot/"eval/fresh/runs"/("half-"+arm)
            proof = read(receipt)
            assert read(run/"receipt.json")["scale_receipt_sha256"] == sha(receipt)
            assert read(run/"receipt.json")["sha256"] == sha(run/"predictions.jsonl")
            assert read(journal/"done.json")["n"] == 192 and read(journal/"manifest.json")["weights"] == proof["weights"]
            if destination.exists():
                remove_artifacts(destination,arm,True)
            return
        folder = source(arm)
        if destination.exists():
            proof = read(destination/"half-export.json")
            assert proof["factor"] == 0.5 and proof["arm"]==arm and weights(destination)==proof["weights"]
            freeze(receipt, proof)
            return
        others = [ROOT/"models"/other/name for other in ARMS if other!=arm for name in ("merged",".building")]
        assert not any(p.exists() for p in others), "Retire the other half-strength export first."
        if staging.exists():
            remove_artifacts(staging,arm,False)
        required = sum(p.stat().st_size for p in BASE.glob("*.safetensors")) + 200_000_000
        if shutil.disk_usage(parent).free < required:
            raise RuntimeError(f"Export needs {required} bytes in {parent}")
        staging.mkdir()
        freeze(staging/".half-export-owner.json",dict(arm=arm,path=str(parent),factor=0.5))
        proof = export(arm, folder, staging)
        if receipt.exists():
            assert read(receipt) == proof, "Recreated checkpoint disagrees with the original receipt"
        freeze(staging/"half-export.json",proof)
        staging.rename(destination)
        freeze(receipt,proof)
        print(json.dumps(proof),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("arm",choices=ARMS)
    parser.add_argument("--retire",action="store_true")
    parser.add_argument("--stage",choices=("diagnostic","confirm"),default="diagnostic")
    args=parser.parse_args()
    main(args.arm,args.retire,args.stage)
