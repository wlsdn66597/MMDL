"""Create a training-only dependency overlay; never install into the baseline venv.

The default is a plan. --execute creates .venv and installs the two pinned extras.
No model, CUDA context, optimizer step or inference is started by this script.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from era_lora.common import load_config, resolve_path, write_json

INVENTORY = "import importlib.metadata as m,json; names=sorted({d.metadata['Name'] for d in m.distributions()}); print(json.dumps({n.lower():m.version(n) for n in names},sort_keys=True))"
SITE = "import sysconfig; print(sysconfig.get_path('purelib'))"


def capture(command: list[str], env: dict) -> str:
    return subprocess.check_output(command, text=True, env=env).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.json"))
    parser.add_argument("--uv", default=str(Path.home() / ".local/bin/uv"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    baseline_python = resolve_path(config, config["evaluation"]["python"])
    venv = ROOT / ".venv"
    new_python = venv / "bin/python"
    plan = {"baseline_python": str(baseline_python), "training_venv": str(venv),
            "extras": ["peft==0.17.1", "accelerate==1.10.1"],
            "baseline_modified": False, "model_loaded": False, "training_started": False}
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    if sys.platform == "win32":
        raise RuntimeError("Run this setup on the requested Linux SSH host")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", CUDA_VISIBLE_DEVICES="")
    before = json.loads(capture([str(baseline_python), "-c", INVENTORY], env))
    expected = {"torch": "2.8.0", "transformers": "4.57.0", "vllm": "0.11.0", "qwen-vl-utils": "0.0.14"}
    if any(before.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Baseline package versions do not match the executed reference")
    baseline_site = capture([str(baseline_python), "-c", SITE], env)
    if venv.exists():
        marker = venv / "era_overlay.json"
        if not marker.is_file() or json.loads(marker.read_text())["baseline_site"] != baseline_site:
            raise RuntimeError("Refusing to modify an existing unrecognized training environment")
    else:
        subprocess.run([str(baseline_python), "-m", "venv", "--without-pip", str(venv)], check=True, env=env)
        write_json(venv / "era_overlay.json", {"baseline_site": baseline_site})
    training_site = Path(capture([str(new_python), "-c", SITE], env))
    (training_site / "baseline_readonly.pth").write_text(
        "import sys; sys.dont_write_bytecode = True\n" + baseline_site + "\n", encoding="utf-8")
    # --no-deps prevents an accidental replacement of the existing torch/transformers stack.
    subprocess.run([args.uv, "pip", "install", "--python", str(new_python), "--no-deps",
                    "-r", str(ROOT / "requirements-training.txt")], check=True, env=env)
    training = json.loads(capture([str(new_python), "-c", INVENTORY], env))
    for key, value in {**expected, "peft": "0.17.1", "accelerate": "1.10.1"}.items():
        if training.get(key) != value:
            raise RuntimeError(f"Unexpected training version: {key}={training.get(key)}")
    capture([str(new_python), "-c", "import peft,accelerate; from transformers import Trainer,Qwen3VLForConditionalGeneration; print('imports_ok')"], env)
    after = json.loads(capture([str(baseline_python), "-c", INVENTORY], env))
    if before != after:
        raise RuntimeError("Baseline inventory changed during setup")
    report = {**plan, "setup_complete": True, "baseline_inventory_before": before,
              "baseline_inventory_after": after, "training_inventory": training,
              "baseline_inventory_unchanged": True}
    write_json(ROOT / "artifacts/environment.json", report)
    print(json.dumps({**plan, "setup_complete": True, "baseline_inventory_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
