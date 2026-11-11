"""Sync the task's compact progress and final evidence to the shared Windows workspace."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
HOST = "rlatjswo@aix408-40"
REMOTE = "era_half_strength_mmmu"
SSH = ["-o","BatchMode=yes","-o","ConnectTimeout=10","-o","ServerAliveInterval=15","-o","ServerAliveCountMax=2"]


def write(path, data):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(data,encoding="utf8")
    tmp.replace(path)


def progress(p):
    if "generated" not in p: return
    write(ROOT/"progress.json",json.dumps(p,ensure_ascii=False,indent=2)+"\n")
    tx=["# MMMU 학습 전 기준과 일반 ERA 50% 비교 진행","","최근 확인 "+p["time_kst"],
        "",f"900문항 중 응답 {p['generated']}개가 저장됐다. 그중 {p['resolved']}개는 채점이 확정됐으며 "
        f"{p['pending_judge']}개는 정답 판정이 남아 있다.",
        "",f"채점이 끝난 같은 {p['resolved']}개에서 후보는 {p['resolved_correct']}개, LoRA 전 모델은 "
        f"{p['resolved_baseline']}개를 맞혔다. 전체 공식 점수는 900개 완료 시 확정된다."]
    if p.get("screen"):
        s=p["screen"]
        tx+=["",f"과목별 3개씩 선정한 처음 90문항은 후보 {s['correct']}/90, 기준 {s['baseline']}/90이다."]
    if p.get("state")=="completed":
        tx+=["","[900문항 최종 결과](report.md)"]
    write(ROOT/"progress.md","\n".join(tx)+"\n")


def main():
    if (ROOT/"verification.json").exists() and json.loads((ROOT/"verification.json").read_bytes()).get("verified"):
        return
    while True:
        try:
            cmd=("cd ~/"+REMOTE+" && if test -f final_ready.json; then cat final_ready.json; "
                 "elif test -f observer_failure.json; then cat observer_failure.json; "
                 "elif test -f progress.json; then cat progress.json; else cat status.json; fi")
            result=subprocess.run(["ssh",*SSH,HOST,cmd],capture_output=True,check=True,timeout=35)
            data=json.loads(result.stdout)
            if data.get("state")=="failed":
                write(ROOT/"remote_failure.json",json.dumps(data,indent=2))
                return
            if "sha256" not in data:
                progress(data)
            else:
                incoming=ROOT/"final.download"
                subprocess.run(["scp",*SSH,"-q",HOST+":"+REMOTE+"/final.tgz",str(incoming)],check=True,timeout=900)
                with incoming.open("rb") as f:
                    assert hashlib.file_digest(f,"sha256").hexdigest()==data["sha256"]
                with tarfile.open(incoming) as archive:
                    for entry in archive.getmembers():
                        assert (entry.isfile() or entry.isdir()) and (ROOT/entry.name).resolve().is_relative_to(ROOT)
                    archive.extractall(ROOT,filter="data")
                incoming.unlink()
                subprocess.run([sys.executable,str(ROOT/"report_results.py")],cwd=ROOT,check=True)
                q=subprocess.run(["ssh",*SSH,HOST,"cat ~/"+REMOTE+"/progress.json"],capture_output=True,timeout=35)
                if q.returncode==0:
                    progress(json.loads(q.stdout))
                write(ROOT/"final_ready.json",json.dumps(data,indent=2))
                return
        except (OSError,subprocess.SubprocessError,ValueError,AssertionError,tarfile.TarError) as exc:
            print(datetime.now().isoformat(timespec="seconds"),str(exc),flush=True)
        time.sleep(55)


if __name__=="__main__":
    main()
