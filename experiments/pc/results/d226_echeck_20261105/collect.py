"""Finalize artifact provenance and gather evidence after the server campaign is complete."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def read(path):return json.loads(path.read_text())
def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")

def main():
    assert (ROOT/"complete").exists()
    files=[];receipt={}
    for arm in ("baseline","standard","checked"):
        predictions=ROOT/"eval"/arm/"predictions.jsonl"
        assert len(predictions.read_text().splitlines())==64
        receipt[arm]=sha(predictions)
        files.extend([predictions,predictions.with_name("manifest.json")])
        files.append(ROOT/("baseline.log" if arm=="baseline" else arm+".eval.log"))
        if arm=="baseline":continue
        dest=ROOT/"train"/arm;merge=read(dest/"merge.json")
        assert read(predictions.with_name("manifest.json"))["weights"]==merge["weights"]
        assert {p.name:sha(p) for p in (dest/"merged").glob("*.safetensors")}==merge["weights"]
        adapter_hashes={p.name:sha(p) for p in (dest/"adapter").iterdir() if p.is_file()}
        # The old helper copies all JSON processor files, including the ancestor's manifest.
        # Give this new export its own accurate manifest, without changing inference weights.
        artifact=dict(arm=arm,starting_weights=merge["parent"],export_weights=merge["weights"],
                      adapter_files=adapter_hashes,train_manifest_sha256=sha(dest/"manifest.json"),
                      remote_directory=str(dest/"merged"))
        write(dest/"merged"/"artifact_manifest.json",artifact)
        files.extend([dest/n for n in ("manifest.json","merge.json","first_step_audit.json","seen_ids.json","merged/artifact_manifest.json")])
        files.extend(p for p in (dest/"adapter").iterdir() if p.is_file())
        files.extend([ROOT/(arm+".train.log"),ROOT/(arm+".merge.log")])
    init=read(ROOT/"train/matched_init.json")
    assert init["identity"]["start_weights"]==read(ROOT/"eval/baseline/manifest.json")["weights"]
    assert read(ROOT/"train/standard/seen_ids.json")==read(ROOT/"train/checked/seen_ids.json")
    ms=[read(ROOT/"train"/arm/"manifest.json") for arm in ("standard","checked")]
    assert all(m["identity"]==init["identity"] and m["frozen_verified"] and m["epoch"]==3 and m["actual_steps"]==45 for m in ms)
    assert ms[0]["initial_adapter"]==ms[1]["initial_adapter"] and ms[0]["final_adapter"]!=ms[1]["final_adapter"]
    files.extend([ROOT/"train/matched_init.json",ROOT/"train/paired_preflight.json",ROOT/"campaign.log"])
    receipt["files"]={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    write(ROOT/"server_receipt.json",receipt);files.append(ROOT/"server_receipt.json")
    with tarfile.open(ROOT/"results.tar.gz","w:gz",compresslevel=2) as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT))
    print("Archive:",sha(ROOT/"results.tar.gz"),"bytes",(ROOT/"results.tar.gz").stat().st_size)

if __name__=="__main__":main()
