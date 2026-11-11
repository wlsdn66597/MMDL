"""Wait without occupying the GPU or polling until the reviewer packet is ready."""
import sys
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from campaign import read,waitfile
if __name__=="__main__":
    waitfile(ROOT/"steps/audit-packet.json",read(ROOT/"started.json")["deadline"])
    subprocess.run([sys.executable,"eval/audit_files.py","split"],cwd=ROOT,check=True)
