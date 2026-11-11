"""Emit the anonymous manual packet as soon as the sealed third-party review exists."""
import json
import sys
import subprocess
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from campaign import waitfile,read,sha,write

if __name__=="__main__":
    waitfile(ROOT/"steps/blind-grade.json",read(ROOT/"started.json")["deadline"])
    subprocess.run([sys.executable,str(ROOT/"eval/score.py"),"audit"],check=True)
    subprocess.run([sys.executable,str(ROOT/"eval/audit_tasks.py"),"split"],check=True)
    files=["packet.json","packet.jsonl","judgments.jsonl","judge_identity.json","judge_complete.json","audit_packet.jsonl",
           *[f"audit_{i}.jsonl" for i in range(4)]]
    archive=ROOT/"audit_ready.tgz";temp=archive.with_suffix(".building")
    with tarfile.open(temp,"w:gz",compresslevel=2) as tar:
        for name in files:
            rel="eval/review/"+name
            tar.add(ROOT/rel,arcname=rel)
    temp.replace(archive)
    write(ROOT/"audit_ready.json",dict(sha256=sha(archive),n=sum(1 for _ in (ROOT/"eval/review/audit_packet.jsonl").open())))
