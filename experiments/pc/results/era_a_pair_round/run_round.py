"""Run the fixed comparison to completion or its shared 12-hour reporting boundary."""
import hashlib
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
OLD = Path.home() / "era_single_call_campaign"
EVAL = Path.home() / "mmdl_test/.venv/bin/python"
TRAIN = Path.home() / "era_decoder_lora_20261030/.venv/bin/python"
BASE = Path.home() / "mmdl-cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
sys.path.insert(0, str(OLD))
import campaign

campaign.ROOT = ROOT
from analysis_round import make_report


def run(controller):
    trainer = ROOT / "train_work/train_pair.py"
    diagnostic = ROOT / "connection_work/run_diagnostic.py"
    official = OLD / "mmmu_eval.py"
    if hashlib.sha256(official.read_bytes()).hexdigest() != "ce72d13a3ac569fa2e67aafab3d54d5d2665f6960d356ea7212be0dfb54e3c3e":
        raise ValueError("Official evaluation wrapper changed")
    arms = ("control", "candidate")
    models = {arm: ROOT / "models" / arm / "merged" for arm in arms}
    for arm in arms:
        controller.step(f"{arm}-train", TRAIN, trainer, "--arm", arm)
        controller.step(f"{arm}-merge", TRAIN, trainer, "--arm", arm, "--merge")
    for name, model in (("control", models["control"]), ("paired", models["candidate"]), ("base", BASE)):
        controller.step(f"{name}-connection", EVAL, diagnostic, "--checkpoint", model, "--name", name)
    controller.step("score-connection", EVAL, ROOT / "connection_work/score.py", "--arms", "control", "paired", "base")

    for scope, arm in (("screen","control"), ("screen","candidate"), ("full","candidate"), ("full","control")):
        # Both arms are fixed before results, and each has its own immutable nomination receipt.
        controller.step(f"{scope}-{arm}", EVAL, official, "--name", arm, "--scope", scope,
                        "--checkpoint", models[arm], "--output-root", ROOT / "mmmu" / arm)
        make_report(ROOT, OLD, "running")
        if scope == "screen" and arm == "candidate":
            controller.step("blind-mmmu-packet", EVAL, ROOT / "blind_mmmu.py")


def main():
    for name in ("logs", "steps"):
        (ROOT / name).mkdir(exist_ok=True)
    if not (ROOT/"started.json").exists():
        t = time.time()
        campaign.save(ROOT/"started.json", dict(start_time=t,start_kst=campaign.timestamp(t),deadline=t+campaign.CAP,
                      deadline_kst=campaign.timestamp(t+campaign.CAP),max_rounds=1))
    controller = campaign.Controller()
    state, detail = "completed", ""
    try:
        run(controller)
    except campaign.EndBudget as e:
        state, detail = "budget_limit", str(e)
    except BaseException as e:
        traceback.print_exc()
        state, detail = "failed", str(e)
    try:
        make_report(ROOT, OLD, state)
    except Exception:
        traceback.print_exc()
        if state == "completed":
            state, detail = "failed", "Analysis failed; solver ledgers preserved"
    if (ROOT/"steps/full-control.json").exists():
        controller.rounds = [4]
    controller.sync(state, detail)
    print(state, detail, flush=True)
    if state == "failed":
        sys.exit(1)


if __name__ == "__main__":
    main()
