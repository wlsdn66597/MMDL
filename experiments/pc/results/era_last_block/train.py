"""Run the archived R9 ordinary trainer with exactly four trainable adapter targets."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
ARCHIVE = Path.home() / "era_a_only_round/train"
PROJECT = Path.home() / "era_decoder_lora_20261030"
sys.path[:0] = [str(ARCHIVE), str(PROJECT)]


def inventory():
    from safetensors import safe_open
    import re
    rows = []
    candidates = []
    for folder, dirs, files in os.walk(Path.home()):
        dirs[:] = [d for d in dirs if d not in {".venv", ".git", "node_modules", "__pycache__"}]
        if "adapter_config.json" not in files or ROOT in Path(folder).parents:
            continue
        file = Path(folder) / "adapter_model.safetensors"
        if not file.is_file():
            candidates.append(str(folder))
            continue
        with safe_open(file, framework="pt", device="cpu") as handle:
            names = handle.keys()
        blocks = sorted({int(i) for name in names for i in re.findall(r"\.layers\.(\d+)\.", name)})
        rows.append(dict(path=folder, blocks=blocks, tensors=len(names)))
        if blocks == [35] or not blocks:
            candidates.append(str(folder))
    (ROOT / "inventory.json").write_text(json.dumps(dict(adapters=rows, inspect=candidates), indent=2))
    print("Inventory:", len(rows), "adapters,", len(candidates), "to inspect", flush=True)
    if candidates:
        raise RuntimeError(f"Inspect candidate provenance before training: {candidates}")


def main():
    import torch
    import peft
    import train_a as r9
    from era_lora import train as layers
    from copy import deepcopy

    ref = r9.engine.read(ARCHIVE / "training_result.json")
    assert ref["verified"] and ref["n"] == 296 and ref["arms"]["ordinary"]["updates"] == 38
    orig_targets = layers.discover_decoder_targets
    orig_peft = peft.get_peft_model
    def targets(model):
        all_names = orig_targets(model)
        assert len(all_names) == 144 and model.config.text_config.num_hidden_layers == 36
        last = [name for name in all_names if ".layers.35." in name]
        assert len(last) == 4
        return last

    def matched_initialization(base, config):
        full_cfg = deepcopy(config)
        full_cfg.target_modules = set(orig_targets(base))
        full = orig_peft(base, full_cfg)
        prior = r9.engine.read(ARCHIVE / "models/r9/ordinary/manifest.json")
        assert layers.parameter_digest(full, True) == prior["initial_adapter"]
        initial = {n: p.detach().clone() for n,p in full.named_parameters()
                   if p.requires_grad and ".layers.35." in n}
        assert len(initial) == 8
        rng = torch.get_rng_state()
        core = full.unload()
        model = orig_peft(core, config)
        with torch.no_grad():
            owned = {n:p for n,p in model.named_parameters() if p.requires_grad}
            assert owned.keys() == initial.keys()
            for n,p in owned.items():
                p.copy_(initial[n])
        torch.set_rng_state(rng)
        return model

    (ROOT / "data").mkdir(exist_ok=True)
    shutil.copyfile(r9.PREFLIGHT, ROOT / "data/receipt.json")
    with patch.object(layers, "discover_decoder_targets", targets), patch.object(peft,"get_peft_model",matched_initialization), patch.object(r9, "HERE", ROOT), patch.object(sys,"argv",["train_a.py","--arm","ordinary"]):
        r9.main()
    out = ROOT / "models/r9/ordinary"
    man = r9.engine.read(out / "manifest.json")
    old = r9.engine.read(ARCHIVE / "models/r9/ordinary/manifest.json")
    assert man["actual_steps"] == old["actual_steps"] == 38 and man["seen"] == old["seen"] == 592
    assert man["frozen_verified"] and man["trainable"] * 36 == old["trainable"]
    assert man["objective_tokens"] == old["objective_tokens"] and man["sequence_sha256"] == old["sequence_sha256"]
    for name in ("parent","train_config","seed","accumulation","steps","versions","hyperparameters"):
        assert man["identity"][name] == old["identity"][name], name
    print("Verified all 38 updates and unchanged native weights", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["inventory"]:
        inventory()
    else:
        main()
