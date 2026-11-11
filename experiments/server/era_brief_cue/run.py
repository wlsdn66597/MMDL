"""Run one solver generation per (arm, item), followed by blinded content review."""
import argparse
import fcntl
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
R9 = ROOT.parent / "era_a_only_round"
sys.path.insert(0, str(R9))
import campaign as jobs

jobs.ROOT = ROOT


def seal(stage):
    if stage == "solver":
        files = ["protocol.md","run.py","prepare.py","checkpoint.py","infer.py"]
    elif stage == "grade":
        files = [str(p.relative_to(ROOT)) for p in sorted((ROOT/"eval").glob("*.py"))]
        assert "eval/score_cue.py" in files
    elif stage == "manual":
        files = [f"eval/review/manual_{i}.jsonl" for i in range(4)]
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
            jobs.write(start,dict(started=jobs.stamp(),deadline=time.time()+12*3600,protocol=jobs.sha(ROOT/"protocol.md")))
        init = jobs.read(start)
        deadline = init["deadline"]
        assert init["protocol"] == jobs.sha(ROOT/"protocol.md")
        try:
            jobs.waiting("solver",deadline)
            assert jobs.sha(R9/"eval/spec/receipt.json") == "d1622bbf7adfa8e8e782600262aa9573839637dcbb505f17df397d4dd42e4c7b"
            h = ROOT.parent/"era_half_strength/eval/review"
            assert jobs.sha(h/"scored.jsonl") == "85db0dc94b7ac345988a6e69c82230f288a9df21bd1a8b09dc12d8be816a44ca"
            assert jobs.sha(h/"summary.json") == "58f6fb9bb8c3d5855089b1e807504d9a39672ec49302e5e7e81fdcfc30859fad"
            jobs.step("prepare",jobs.EVAL,ROOT/"prepare.py",[],deadline)
            for arm in ("base","ordinary","enhanced"):
                if arm != "base":
                    jobs.step("scale-"+arm,jobs.TRAIN,ROOT/"checkpoint.py",[arm],deadline)
                jobs.step("infer-"+arm,jobs.EVAL,ROOT/"infer.py",[arm],deadline)
                if arm != "base":
                    jobs.step("retire-"+arm,jobs.TRAIN,ROOT/"checkpoint.py",[arm,"--retire"],deadline)
            jobs.waiting("grade",deadline)
            scorer = ROOT/"eval/score_cue.py"
            for mode in ("packet","judge","audit"):
                jobs.step(mode,jobs.EVAL,scorer,[mode],deadline)
            jobs.waiting("manual",deadline)
            for mode in ("merge","score"):
                jobs.step(mode,jobs.EVAL,scorer,[mode],deadline)
            passed = jobs.read(ROOT/"eval/review/summary.json")["advance"]
            assert type(passed) is bool
            jobs.status("completed","pilot-final",pilot_advanced=passed)
        except Exception as exc:
            stage = jobs.read(ROOT/"status.json")["stage"] if (ROOT/"status.json").exists() else "setup"
            jobs.status("time_limit" if isinstance(exc,TimeoutError) else "failed",stage,error=str(exc))
            raise


if __name__=="__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("--seal", choices=("solver","grade","manual"))
    args = cli.parse_args()
    if args.seal:
        seal(args.seal)
    else:
        main()
