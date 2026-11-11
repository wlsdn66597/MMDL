"""Wait for the last dev inference and publish the masked assessment cards using CPU."""
import os
import subprocess
from server_run import EVAL, ROOT, read, wait_file

if __name__=="__main__":
    terminal=lambda:read(ROOT/"status.json")["state"] in ("failed","completed")
    if wait_file(ROOT/"steps/preserved-science.json",terminal):
        env={**os.environ,"CUDA_VISIBLE_DEVICES":""}
        subprocess.run([EVAL,"-u",ROOT/"eval/score.py","packet"],cwd=ROOT,env=env,check=True)
