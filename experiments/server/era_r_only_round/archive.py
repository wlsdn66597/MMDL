"""Package sealed results, keeping the much larger model checkpoints on the server."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parent
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    state=json.loads((ROOT/"status.json").read_text())
    assert state["state"]=="completed"
    subprocess.run([sys.executable,str(ROOT/"eval/report.py")],cwd=ROOT,check=True)
    files=["steps","signals","started.json","status.json","decision.json","report.md","train/data/receipt.json",
           "train/training_result.json","train/training_report.md","eval/runs"]
    files.extend("eval/review/"+f for f in ("packet.json","packet.jsonl","judge_identity.json","judge_complete.json","judgments.jsonl","bare.jsonl",
                 "audit_receipt.json","audit_packet.jsonl","manual.jsonl","scored.jsonl","summary.json"))
    for a in ("base","ordinary","enhanced"):
        files.extend(f"eval/fresh/runs/r8-{a}/{f}" for f in ("manifest.json","predictions.jsonl","done.json"))
    if state["promoted"]:files.extend(["mmmu/enhanced","mmmu/roster.json"])
    if (ROOT/"cases.md").is_file():files.append("cases.md")
    if (ROOT/"eval/review/sensitivity.json").is_file():files.append("eval/review/sensitivity.json")
    temp=ROOT/"results.building";dst=ROOT/"results.tgz"
    with tarfile.open(temp,"w:gz",compresslevel=2) as tar:
        for f in files:tar.add(ROOT/f,arcname=f)
    temp.replace(dst)
    proof=dict(sha256=sha(dst),report_sha256=sha(ROOT/"report.md"),summary_sha256=sha(ROOT/"eval/review/summary.json"))
    (ROOT/"archive_receipt.json").write_text(json.dumps(proof)+"\n",encoding="utf8")
    print(proof)


if __name__=="__main__":main()
