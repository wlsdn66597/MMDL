"""Save result artifacts, excluding model weights, after the experiment ends."""
import hashlib
import json
import tarfile
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
def sha(p):
    with p.open("rb") as s:return hashlib.file_digest(s,"sha256").hexdigest()


def main():
    status=json.loads((ROOT/"status.json").read_text())
    assert status["state"]=="completed"
    subprocess.run([sys.executable,str(ROOT/"eval/report.py")],check=True,cwd=ROOT)
    files=["status.json","started.json","decision.json","report.md","steps","signals","attempts","eval/runs",
           "eval/review/packet.json","eval/review/packet.jsonl","eval/review/judgments.jsonl",
           "eval/review/judge_identity.json","eval/review/judge_complete.json","eval/review/audit_packet.jsonl",
           "eval/review/manual.jsonl","eval/review/answer_only.jsonl","eval/review/extra_packet.jsonl","eval/review/extra_manual.jsonl","eval/review/scored.jsonl","eval/review/summary.json",
           "train/training_result.json","train/training_report.md","train/data/receipt.json"]
    for a in ("base","ordinary","enhanced"):
        for f in ("manifest.json","predictions.jsonl","done.json"):
            files.append(f"eval/fresh/runs/r7-{a}/{f}")
    if status["promoted"]:files.extend(["mmmu/enhanced","mmmu/roster.json"])
    for rel in ["cases.md","eval/review/general_extra_cards.jsonl","eval/review/general_extra_labels.jsonl","eval/review/general_sensitivity.json"]:
        if (ROOT/rel).is_file():files.append(rel)
    path=ROOT/"results.tgz";temp=path.with_suffix(".building")
    with tarfile.open(temp,"w:gz",compresslevel=3) as out:
        for f in files:out.add(ROOT/f,arcname=f)
    temp.replace(path)
    proof={"sha256":sha(path),"report_sha256":sha(ROOT/"report.md"),"summary_sha256":sha(ROOT/"eval/review/summary.json")}
    (ROOT/"archive_receipt.json").write_text(json.dumps(proof,indent=2)+"\n",encoding="utf8")
    print(proof)


if __name__=="__main__":main()
