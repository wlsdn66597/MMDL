"""Finish the untouched tail of the control run once, preserving the missing in-flight ID."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path.home() / "era_single_call_campaign"))
import mmmu_eval as original

OLD_SHA = "6645f2faf34be449a2c2b55d723d4697a4acdc212a5d7f45b728d5db230631e7"
MANIFEST_SHA = "57062abffafa395766575e279061a77c82866ba5df4d4191729eaa1f3113f173"
FINGERPRINT = "f9df56ed0b3db53999c44f66f975054ba052978c8958efc62e442c39fe3dff4e"
WRAPPER_SHA = "ce72d13a3ac569fa2e67aafab3d54d5d2665f6960d356ea7212be0dfb54e3c3e"
GAP = "validation_Mechanical_Engineering_11"
HARD_TIME = "2026-11-07T21:15:09+09:00"


def run(plan_only):
    source = ROOT / "mmmu/control/control"
    rec = ROOT / "control_recovery"
    solver = rec / "solver"
    solver.mkdir(parents=True, exist_ok=True)
    check = original.require
    check(original.sha(Path(original.__file__))==WRAPPER_SHA, "Original solver wrapper hash changed")
    hard = original.read(ROOT/"started.json")["deadline"] + 4*3600
    extension = original.read(ROOT/"extension.json")
    check(extension["hard_deadline_kst"]==HARD_TIME and int(hard)==1794053709, "Extension deadline changed")
    check(time.time()<hard, "Control-only extension already expired")
    args = SimpleNamespace(project=Path.home()/"era_decoder_lora_20261030",baseline=Path.home()/"mmmu_baseline_20261009",
                           checkpoint=(ROOT/"models/control/merged").resolve(),output_root=source.parent)
    frozen, config, roster, archive = original.plan(args)
    import prepare
    from era_lora.evaluate import checkpoint_preflight
    original_ledger = source/"predictions.jsonl"
    with ExitStack() as stack:
        stack.enter_context(frozen.run_lock(source/".run.lock"))
        stack.enter_context(frozen.run_lock(solver/".run.lock"))
        check(original.sha(original_ledger)==OLD_SHA and original.sha(source/"manifest.json")==MANIFEST_SHA,
              "Original source ledger/identity changed")
        check(original.read(source/"solver_inflight.json")=={"id":GAP}, "Original in-flight marker changed")
        manifest = original.read(source/"manifest.json")
        weights = checkpoint_preflight(args.checkpoint)
        fingerprint = hashlib.sha256(json.dumps(weights, sort_keys=True).encode()).hexdigest()
        check(fingerprint==manifest["fingerprint"]==FINGERPRINT and config==manifest["configuration"] and
              manifest["model_identity"]==dict(path=str(args.checkpoint),hashes=weights) and
              manifest["source_sha256"]==WRAPPER_SHA and manifest["baseline_sha256"]==original.BASELINE_RESULT and
              manifest["pixels_sha256"]==original.PIXEL_FILE and manifest["solver_calls_per_item"]==1,"Identity mismatch")
        old_rows = prepare.read_jsonl(original_ledger)
        ids = [r["id"] for r in old_rows]
        old_screen = set(roster["screen_ids"])
        ordered = roster["screen_ids"] + [rid for rid in roster["full_ids"] if rid not in old_screen]
        check(len(ids)==719 and ids==ordered[:719] and ordered[719]==GAP, "Unexpected old submitted prefix")
        for row in old_rows:
            original.match_inputs(row,archive[row["id"]])
        virgin = ordered[720:]
        check(len(virgin)==180 and not set(virgin).intersection([GAP,*ids]) and
              [rid for rid in roster["full_ids"] if rid in set(virgin)]==virgin, "Tail scope mismatch")
        plan = dict(old_ledger_sha256=OLD_SHA,old_manifest_sha256=MANIFEST_SHA,interrupted_id=GAP,
                    original_ids=ids,virgin_ids=virgin,fingerprint=fingerprint,hard_deadline=hard,hard_deadline_kst=HARD_TIME)
        original.freeze(rec/"plan.json",plan)
        original.freeze(solver/"manifest.json",manifest)
        print(f"Verified 719 saved, 1 irrevocably omitted ({GAP}), 180 virgin IDs",flush=True)
        if plan_only:
            return
        if (rec/"solver_done.json").exists():
            seal = original.read(rec/"solver_done.json")
            check(seal["predictions_sha256"]==original.sha(solver/"predictions.jsonl") and
                  seal["plan_sha256"]==original.sha(rec/"plan.json"), "Supplement seal mismatch")
            return
        # O_EXCL survives parent or host termination. An interrupted invocation can never auto-repeat.
        check(not (solver/"predictions.jsonl").exists() and not (solver/"solver_inflight.json").exists(),
              "Supplement contains submissions; refusing to start another solver invocation")
        with (rec/"solver_started.json").open("x",encoding="utf8") as marker:
            json.dump(dict(pid=os.getpid(),start=time.time(),plan_sha256=original.sha(rec/"plan.json")),marker)
            marker.flush()
            os.fsync(marker.fileno())
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
        os.environ.pop("VLLM_BATCH_INVARIANT",None)
        left=hard-time.time()
        check(left>0, "Extension expired before launch")
        signal.signal(signal.SIGALRM,signal.SIG_DFL)
        signal.setitimer(signal.ITIMER_REAL,left)
        original_save = original.save
        def timed_save(path,value):
            if path==solver/"solver_inflight.json":
                check(time.time()<hard and value["id"] in virgin, "Denied solver submission outside approved scope")
            original_save(path,value)
        with patch.object(original,"save",timed_save):
            original.inference(args,solver,virgin,archive,config,manifest)
        result = prepare.read_jsonl(solver/"predictions.jsonl")
        check([r["id"] for r in result]==virgin and not (solver/"solver_inflight.json").exists(),
              "Supplement incomplete")
        for row in result:
            original.match_inputs(row,archive[row["id"]])
        original.freeze(rec/"solver_done.json",dict(n=len(result),predictions_sha256=original.sha(solver/"predictions.jsonl"),
                                                     plan_sha256=original.sha(rec/"plan.json"),fingerprint=fingerprint))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan-only",action="store_true")
    run(ap.parse_args().plan_only)
