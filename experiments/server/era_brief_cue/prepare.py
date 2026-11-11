"""Insert one predeclared instruction into the sealed R9 solver view."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R9 = ROOT.parent / "era_a_only_round"
SPEC = ROOT / "eval/spec"
CUE = "State the relevant evidence from the image, the rule you use, and how it leads to your choice. Keep each part brief."
LAST = 'On the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'
R9_INPUTS = "ff9f3846e0820e3939dd07488dcdb5c3bc2d800b8d96a81fbea82afe3f8934a5"
R9_GOLD = "5396d9c80697c0577f6657e249f390d12e321575beae4964168e1da8d3829525"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]


def seal_bytes(path, content):
    if path.exists():
        assert path.read_bytes() == content, path
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def prompted(row):
    assert set(row) == {"id", "dataset", "image", "image_sha256", "prompt"}
    old = row["prompt"]
    assert old.endswith("\n"+LAST) and old.count(LAST)==1 and CUE not in old
    new = old.removesuffix(LAST) + CUE + "\n" + LAST
    assert new.replace(CUE+"\n", "", 1) == old and new.count(CUE)==1
    return row | {"prompt": new}


def main():
    source = R9 / "eval/spec"
    assert sha(source/"inputs.jsonl")==R9_INPUTS and sha(source/"gold.jsonl")==R9_GOLD
    assert read(source/"receipt.json")=={"inputs": R9_INPUTS, "gold": R9_GOLD}
    old = rows(source/"inputs.jsonl")
    gold = rows(source/"gold.jsonl")
    assert len(old)==len(gold)==len({r["id"] for r in old})==192
    assert [r["id"] for r in old] == [r["id"] for r in gold]
    assert sum(r["dataset"]=="paired" for r in old)==sum(r["dataset"]=="general" for r in old)==96
    seen = {}
    for r in old:
        path = (R9/r["image"]).resolve()
        assert path.is_relative_to(R9.resolve()) and path.is_file()
        if path not in seen:
            seen[path]=sha(path)
        assert seen[path]==r["image_sha256"]
    assert len(seen)==144
    data = "".join(json.dumps(prompted(r),ensure_ascii=False)+"\n" for r in old).encode("utf8")
    seal_bytes(SPEC/"inputs.jsonl", data)
    seal_bytes(SPEC/"gold.jsonl", (source/"gold.jsonl").read_bytes())
    proof = dict(inputs=sha(SPEC/"inputs.jsonl"), gold=sha(SPEC/"gold.jsonl"),
                 original_inputs=R9_INPUTS, original_gold=R9_GOLD, cue=CUE, n=192,
                 paired=96, general=96, images=len(seen), changed_prompts=192,
                 only_prompt_changed=True, source_image_root="era_a_only_round")
    seal_bytes(SPEC/"receipt.json", (json.dumps(proof,indent=2)+"\n").encode("utf8"))
    print(proof, flush=True)


if __name__=="__main__":
    main()
