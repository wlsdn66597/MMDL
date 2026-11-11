"""Export completed runs and adapter checksums; baseline and judged outputs are immutable evidence."""
import hashlib,json,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def main():
    assert (R/"complete").exists() and (R/"judge.ok").exists()
    files=[R/n for n in ("judge.json","target_receipt.json","paired_train.jsonl","draft_audit.json","statement.jsonl","question.jsonl","models/matched_init.json","models/paired_preflight.json","campaign.log")]
    for arm in ("baseline","collect","statement","question"):
        folder=R/"eval"/arm
        files.extend([folder/"predictions.jsonl",folder/"manifest.json"])
        assert len((folder/"predictions.jsonl").read_text().splitlines())==72
        if arm not in ("statement","question"):continue
        t=R/"models"/arm
        meta=read(t/"manifest.json");seq=read(t/"seen_ids.json");m=read(t/"merge.json")
        assert meta["frozen_verified"] and meta["actual_steps"]==27 and meta["epoch"]==3 and len(seq)==429
        assert read(folder/"manifest.json")["weights"]==m["weights"]=={p.name:sha(p) for p in (t/"merged").glob("*.safetensors")}
        files.extend(t/n for n in ("manifest.json","seen_ids.json","merge.json","first_step_audit.json"))
        files.extend(p for p in (t/"adapter").iterdir() if p.is_file())
        files.extend(R/(arm+"."+s+".log") for s in ("train","merge","eval"))
    ms=[read(R/"models"/a/"manifest.json") for a in ("statement","question")]
    assert ms[0]["identity"]==ms[1]["identity"] and ms[0]["initial_adapter"]==ms[1]["initial_adapter"] and ms[0]["sequence_sha256"]==ms[1]["sequence_sha256"]
    assert ms[0]["final_adapter"]!=ms[1]["final_adapter"]
    receipt=R/"server_receipt.json"
    receipt.write_text(json.dumps({str(p.relative_to(R)):sha(p) for p in files},indent=2))
    with tarfile.open(R/"results.tgz","w:gz",compresslevel=2) as tar:
        for p in files+[receipt]:tar.add(p,arcname=p.relative_to(R))
    print("Verified package:",sha(R/"results.tgz"),(R/"results.tgz").stat().st_size)
if __name__=="__main__":main()
