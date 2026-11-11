"""Make a bounded remote evaluation bundle from frozen, referenced inputs."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
def main():
    files={"release.py","eval/infer.py","eval/prepare.py","eval/score.py","eval/grade.py",
           "paired/items.jsonl","paired/sources.jsonl","paired/receipt.json","paired/overlap_final.json","paired/review_protocol.md",
           "general/items.jsonl","general/receipt.json","general/build.py"}
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/"eval/spec").glob("*.json*"))
    for line in (ROOT/"eval/spec/inputs.jsonl").read_text(encoding="utf8").splitlines():
        r=json.loads(line);files.add(r["image"])
        assert hashlib.sha256((ROOT/r["image"]).read_bytes()).hexdigest()==r["image_sha256"]
    with tarfile.open(ROOT/"eval_release.tgz","w:gz",compresslevel=2) as out:
        for f in sorted(files):out.add(ROOT/f,arcname=f)
    print(len(files),hashlib.sha256((ROOT/"eval_release.tgz").read_bytes()).hexdigest())


if __name__=="__main__":main()
