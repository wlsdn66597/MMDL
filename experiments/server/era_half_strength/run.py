"""Run the single-GPU LoRA-strength comparison from independently sealed inputs."""
import argparse
import fcntl
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "era_a_only_round"
sys.path.insert(0,str(ORIGINAL))
import campaign as jobs

jobs.ROOT = ROOT


def signals(stage):
    if stage == "solver":
        files = ["protocol.md", "run.py", "infer.py", "scale.py"]
    elif stage == "grade":
        files = ["eval/" + p.name for p in sorted((ROOT/"eval").glob("*.py"))]
        assert "eval/diagnostic.py" in files
    elif stage == "manual":
        files = [f"eval/review/manual_{i}.jsonl" for i in range(4)]
    elif stage == "confirm":
        files = ["confirm_run.py", "confirm/eval/spec/receipt.json", "confirm/eval/spec/inputs.jsonl",
                 "confirm/eval/spec/gold.jsonl"]
    else:
        raise ValueError(stage)
    hashes = {f:jobs.sha(ROOT/f) for f in files}
    target = ROOT/"signals"/(stage+".json")
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        assert jobs.read(target)["hashes"] == hashes
    else:
        jobs.write(target,dict(hashes=hashes,time_kst=jobs.stamp()))


def main():
    for name in ["steps","signals","logs"]:
        (ROOT/name).mkdir(exist_ok=True)
    with (ROOT/".run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = ROOT/"started.json"
        if not start.exists():
            jobs.write(start,dict(started=jobs.stamp(),deadline=time.time()+12*3600,
                                  protocol=jobs.sha(ROOT/"protocol.md")))
        receipt = jobs.read(start)
        deadline = receipt["deadline"]
        assert receipt["protocol"] == jobs.sha(ROOT/"protocol.md")
        try:
            jobs.waiting("solver",deadline)
            assert jobs.sha(ORIGINAL/"eval/spec/receipt.json") == "d1622bbf7adfa8e8e782600262aa9573839637dcbb505f17df397d4dd42e4c7b"
            assert jobs.sha(ORIGINAL/"eval/review/summary.json") == "683bdc873df9b67a106d4e7d0dc7fe6ca71454bd54a6fe937ad7482a82d9096f"
            assert jobs.sha(ORIGINAL/"decision.json") == "9d867d9b3ef8d1dff5616fd13cfa7d12ea7b2ce39f5fdfb4cd5cd98e9c280006"
            for arm in ("ordinary","enhanced"):
                jobs.step("scale-"+arm,jobs.TRAIN,ROOT/"scale.py",[arm],deadline)
                jobs.step("infer-"+arm,jobs.EVAL,ROOT/"infer.py",[arm],deadline)
                jobs.step("retire-"+arm,jobs.TRAIN,ROOT/"scale.py",[arm,"--retire"],deadline)
            jobs.waiting("grade",deadline)
            diagnostic = ROOT/"eval/diagnostic.py"
            for mode in ("packet","judge","audit"):
                jobs.step("diagnostic-"+mode,jobs.EVAL,diagnostic,[mode],deadline)
            jobs.waiting("manual",deadline)
            for mode in ("merge","score"):
                jobs.step("diagnostic-"+mode,jobs.EVAL,diagnostic,[mode],deadline)
            summary = jobs.read(ROOT/"eval/review/summary.json")
            passed = summary["advance"]
            assert type(passed) is bool
            if passed:
                jobs.waiting("confirm",deadline)
                jobs.step("confirmation",jobs.EVAL,ROOT/"confirm_run.py",[],deadline)
            jobs.status("completed","final",diagnostic_advanced=passed)
        except Exception as exc:
            stage = jobs.read(ROOT/"status.json")["stage"] if (ROOT/"status.json").exists() else "setup"
            jobs.status("time_limit" if isinstance(exc,TimeoutError) else "failed",stage,error=str(exc))
            raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--seal", choices=("solver","grade","manual","confirm"))
    args = cli.parse_args()
    if args.seal:
        signals(args.seal)
    else:
        main()
