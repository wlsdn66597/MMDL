"""Rebuild and retire the previously verified half-strength export inside this run."""
import argparse
from importlib import util
from pathlib import Path

from prepare import ROOT, read, seal_bytes, sha

HALF = ROOT.parent/"era_half_strength"
HALF_SCRIPT_SHA = "9ee1ee1ae0755ff43c9591fa7ce451633fe70b5bbeb05e98fbb1ffb0bcefe021"
PROOFS = {
    "ordinary": "1fdf6aeaca39c491fe51a5dce15e5ca3a4d2a02a9f6161d13bbf6a5f8de9ca8e",
    "enhanced": "a5564fe01c5ad554b7957d75203cb7d760f12bdaa804b6a27f40cee825813785",
}


def exporter():
    script = HALF/"scale.py"
    assert sha(script)==HALF_SCRIPT_SHA, "Half-strength exporter changed"
    spec = util.spec_from_file_location("archived_half_export", script)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def checkpoint(arm, module):
    if arm=="base":
        assert module.weights(module.BASE)==module.BASE_HASHES
        return module.BASE, None
    receipt = ROOT/"scale_receipts"/f"{arm}.json"
    assert sha(receipt)==PROOFS[arm] and sha(HALF/"scale_receipts"/receipt.name)==PROOFS[arm]
    model = ROOT/"models"/arm/"merged"
    proof = read(receipt)
    assert proof["factor"]==0.5 and proof["arm"]==arm
    assert proof==read(model/"half-export.json") and module.weights(model)==proof["weights"]
    assert read(HALF/"eval/fresh/runs"/("half-"+arm)/"manifest.json")["weights"]==proof["weights"]
    return model, PROOFS[arm]


def main(arm, retire):
    module = exporter()
    receipt = HALF/"scale_receipts"/f"{arm}.json"
    assert sha(receipt)==PROOFS[arm]
    seal_bytes(ROOT/"scale_receipts"/receipt.name, receipt.read_bytes())
    if not retire:
        module.main(arm)
        checkpoint(arm,module)
        return
    import fcntl
    journal = ROOT/"eval/fresh/runs"/("cue-"+arm)
    out = ROOT/"eval/runs"/arm
    with (ROOT/"models/.half-export.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        done, proof = read(journal/"done.json"), read(out/"receipt.json")
        assert proof["arm"]==arm and proof["fresh"]==done["n"]==192
        assert done["output_sha256"]==proof["sha256"]==sha(out/"predictions.jsonl")==sha(journal/"predictions.jsonl")
        assert done["manifest_sha256"]==proof["journal_manifest_sha256"]==sha(journal/"manifest.json")
        assert proof["scale_receipt_sha256"]==PROOFS[arm]
        assert read(journal/"manifest.json")["weights"]==read(receipt)["weights"]
        path = ROOT/"models"/arm/"merged"
        if path.exists():
            module.remove_artifacts(path,arm,True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm",choices=tuple(PROOFS))
    parser.add_argument("--retire",action="store_true")
    args=parser.parse_args()
    main(args.arm,args.retire)
