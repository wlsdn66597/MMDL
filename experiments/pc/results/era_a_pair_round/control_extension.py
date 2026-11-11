"""Extend only the live control evaluation, with a 12 h snapshot and a fixed +4 h cap.

The evaluation holds its own process group and durable request ledger. Its original
parent remains its parent and reaps it; this monitor never invokes the evaluator.
"""
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import traceback

from analysis_round import make_report

ROOT = Path(__file__).resolve().parent
OLD = Path.home() / "era_single_call_campaign"
sys.path.insert(0, str(OLD))
from campaign import read, save, timestamp


def wait_pid(fd, until):
    return bool(select.select([fd], [], [], max(0, until-time.time()))[0])


def make_midpoint(deadline):
    make_report(ROOT, OLD, "running_control_extension")
    report = read(ROOT/"mmmu_status.json")
    report.update(original_deadline_kst=timestamp(deadline),captured_kst=timestamp())
    (ROOT/"midpoint.md").write_bytes((ROOT/"server_report.md").read_bytes())
    save(ROOT/"midpoint.json", report)


def main():
    midpoint = read(ROOT/"started.json")["deadline"]
    hard = midpoint + 4*3600
    if time.time() >= midpoint or (ROOT/"extension.json").exists():
        raise RuntimeError("Extension must attach once before the original deadline")
    pid = int(subprocess.check_output(["tmux","display-message","-p","-t","era-a4","#{pane_pid}"]))
    proc = Path(f"/proc/{pid}")
    assert (proc/"cwd").resolve() == ROOT and (proc/"cmdline").read_bytes().split(b"\0")[-2] == b"run_round.py"
    assert read(ROOT/"status.json")["stage"] == "full-control" and (ROOT/"steps/full-candidate.json").exists()
    children = (proc/"task"/str(pid)/"children").read_text().split()
    assert len(children) == 1
    child = int(children[0])
    argv = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
    assert argv[2] == os.fsencode(OLD/"mmmu_eval.py") and argv[3:7] == [b"--name",b"control",b"--scope",b"full"]
    assert os.getpgid(child) == child and os.getsid(child) == child
    parent_fd = os.pidfd_open(pid)
    child_fd = os.pidfd_open(child)
    stopped = cutoff = False
    try:
        signal.pidfd_send_signal(parent_fd, signal.SIGSTOP)
        stopped = True
        for _ in range(150):
            if (proc/"stat").read_text().split(") ")[1][0] == "T":
                break
            time.sleep(.02)
        else:
            raise RuntimeError("Could not pause the control worker's timer")
        expected = dict(stage="full-control", attached_kst=timestamp(), parent=pid, active_evaluator=child,
                        midpoint_kst=timestamp(midpoint), hard_deadline_kst=timestamp(hard),
                        new_solver_submissions_by_monitor=0)
        save(ROOT/"extension.json",expected)
        if not wait_pid(child_fd, midpoint):
            try:
                make_midpoint(midpoint)
            except Exception:
                save(ROOT/"midpoint_error.json", dict(error=traceback.format_exc(),time_kst=timestamp()))
            if not wait_pid(child_fd, hard):
                # The original parent still owns this PID, so the group cannot be recycled.
                cutoff = True
                os.killpg(child, signal.SIGTERM)
                wait_pid(child_fd, time.time()+2)
                try:
                    os.killpg(child, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        signal.pidfd_send_signal(parent_fd, signal.SIGCONT)
        stopped = False
        if not wait_pid(parent_fd, time.time()+120):
            raise RuntimeError("Control evaluator exited; parent did not finalize")
        status = read(ROOT/"status.json")
        if cutoff:
            status.update(state="budget_limit", stage="full-control",detail="Approved +4h control-only limit reached",
                          time_kst=timestamp(), deadline_kst=timestamp(hard),seconds_left=0)
            save(ROOT/"status.json",status)
            make_report(ROOT,OLD,"budget_limit")
        else:
            status.update(deadline_kst=timestamp(hard),seconds_left=max(0,int(hard-time.time())))
            if status["state"] == "budget_limit" and (ROOT/"mmmu/control/control/full/complete.json").is_file():
                # The original timer can expire just as the old parent resumes.
                # Validate the official seal and publish without restarting any worker.
                make_report(ROOT,OLD,"completed")
                status.update(state="completed", detail="Control full score sealed under the extension",rounds_completed=1)
            save(ROOT/"status.json",status)
            if status["state"] == "completed":
                assert (ROOT/"mmmu/control/control/full/complete.json").is_file()
        save(ROOT/"extension_done.json",dict(state=status["state"], completed_kst=timestamp(),
                                            hard_deadline_kst=timestamp(hard), control_received_full=(ROOT/"mmmu/control/control/full/complete.json").exists()))
    finally:
        if stopped:
            signal.pidfd_send_signal(parent_fd, signal.SIGCONT)
        os.close(child_fd)
        os.close(parent_fd)


if __name__ == "__main__":
    main()
