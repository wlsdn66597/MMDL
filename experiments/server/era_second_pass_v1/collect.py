import hashlib
import json
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def main():
    artifacts=[]
    roots=read(ROOT/"freeze.json")
    assert all(sha(ROOT/k)==v for k,v in roots.items())
    test=read(ROOT/"holdout/inputs.json")
    seen=[]
    prompts=None
    for arm in ("start","control","paired"):
        folder=ROOT/"eval"/arm
        preds=[json.loads(s) for s in (folder/"predictions.jsonl").read_text().splitlines()]
        assert [r["id"] for r in preds]==[r["id"] for r in test] and len(preds)==72
        assert [r["image_sha256"] for r in preds]==[r["image_sha256"] for r in test]
        current=[r["prompt"] for r in preds]
        assert prompts is None or prompts==current
        prompts=current
        artifacts.extend([folder/"manifest.json",folder/"predictions.jsonl"])
        if arm=="start":continue
        model=ROOT/"models"/arm
        m=read(model/"manifest.json")
        assert m["frozen_verified"] and m["complete"] and m["epoch"]==3 and m["actual_steps"]==18
        seen.append((m["initial_adapter"],m["identity"]["start_weights"],read(model/"seen_ids.json")))
        assert len(seen[-1][-1])==288
        assert read(folder/"manifest.json")["weights"]==read(model/"merge.json")["weights"]
        assert read(ROOT/"eval/start/manifest.json")["weights"]==m["identity"]["start_weights"]
        artifacts.extend(model/name for name in ("manifest.json","merge.json","seen_ids.json","preflight.json"))
        artifacts.extend(p for p in (model/"adapter").iterdir() if p.is_file())
        artifacts.extend(ROOT/(arm+s) for s in (".train.log",".merge.log"))
    assert seen[0]==seen[1]
    receipt={str(p.relative_to(ROOT)):sha(p) for p in artifacts}
    file=ROOT/"receipt.json"
    file.write_text(json.dumps(receipt,indent=2)+"\n")
    with tarfile.open(ROOT/"verified.tgz","w:gz") as tar:
        for p in [*artifacts,file]:tar.add(p,arcname=str(p.relative_to(ROOT)))
    print("Verified package:",sha(ROOT/"verified.tgz"),flush=True)


if __name__=="__main__":main()
