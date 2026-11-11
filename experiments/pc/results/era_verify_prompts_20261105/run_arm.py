"""Run the pinned 48-image evaluator, inserting one instruction before the final-line directive."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = Path.home() / "era_row_position_20261105"

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("arm",choices=("generic","boundary"))
    arm=parser.parse_args().arm
    work=ROOT/arm
    work.mkdir(exist_ok=True)
    for name in ("images","inputs.json","eval_config.json"):
        link=work/name
        if not link.exists(): link.symlink_to(PREVIOUS/name)
        assert link.resolve()==(PREVIOUS/name).resolve()
    old=json.loads((PREVIOUS/"run/manifest.json").read_text())
    assert sha(PREVIOUS/"run_eval.py")==old["eval_script_sha256"]
    assert sha(work/"inputs.json")==old["inputs_sha256"]
    sys.path.insert(0,str(PREVIOUS))
    import run_eval as original
    import evaluate as frozen
    original.HERE=work
    base_prompt=frozen.prompt_for
    instruction=json.loads((ROOT/"instructions.json").read_text())[arm]
    def add_instruction(row,config):
        prompt=base_prompt(row,config)
        before,sep,suffix=prompt.rpartition("\n")
        assert sep and suffix==config["prompts"]["suffixes"]["multiple-choice"]
        return "\n".join((before,instruction,suffix))
    frozen.prompt_for=add_instruction
    meta=dict(arm=arm,instruction=instruction,input_sha256=old["inputs_sha256"],
              wrapper_sha256=sha(Path(__file__)),instructions_sha256=sha(ROOT/"instructions.json"))
    ident=work/"arm.json"
    if ident.exists(): assert json.loads(ident.read_text())==meta
    ident.write_text(json.dumps(meta,indent=2)+"\n")
    original.main()

if __name__=="__main__": main()
