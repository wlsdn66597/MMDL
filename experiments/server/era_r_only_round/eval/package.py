"""Transfer only frozen rosters, their images, and executable evaluation code."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    files={"release.py","paired/items.jsonl","paired/sources.jsonl","paired/receipt.json","paired/overlap_final.json",
           "general/items.jsonl","general/receipt.json"}
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/"eval").glob("*.py"))
    files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/"eval/spec").glob("*.json*"))
    for r in map(json.loads,(ROOT/"eval/spec/inputs.jsonl").read_text(encoding="utf8").splitlines()):
        p=ROOT/r["image"]
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r["image_sha256"]
        files.add(r["image"])
    out=ROOT/"eval_release.tgz"
    with tarfile.open(out,"w:gz",compresslevel=2) as tar:
        for f in sorted(files):tar.add(ROOT/f,arcname=f)
    print(len(files),hashlib.sha256(out.read_bytes()).hexdigest())


if __name__=="__main__":main()
