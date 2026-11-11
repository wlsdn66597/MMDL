"""Wait for this run's existing tmux process exit without polling or making a model call."""
import json
import os
from pathlib import Path
import select
import subprocess

root = Path(__file__).resolve().parent
session = subprocess.run(["tmux","display-message","-p","-t","era-a4-control","#{pane_pid}"],capture_output=True,text=True)
if session.returncode == 0 and session.stdout.strip():
    try:
        fd = os.pidfd_open(int(session.stdout.strip()))
    except ProcessLookupError:
        pass
    else:
        select.select([fd],[],[])
        os.close(fd)
print((root/"control_recovery/status.json").read_text(),flush=True)
