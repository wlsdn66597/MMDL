import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    stage=load(ROOT/"status.json") or {}
    state=dict(kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),controller=stage,
        clock=load(ROOT/"started.json"),rounds=load(ROOT/"rounds.json") or [],nominee=load(ROOT/"nominee.json"),
        final=load(ROOT/"full_result.json"))
    cached={}
    for path in sorted((ROOT/"benchmark").glob("*/predictions.jsonl")):
        with path.open() as f:n=sum(bool(line.strip()) for line in f)
        cached[path.parent.name]=dict(single_call_responses=n,screen=load(path.parent/"screen/summary.json"))
    state["benchmark"]=cached
    state["training"]={str(p.parent.relative_to(ROOT)):load(p) for p in sorted((ROOT/"models").glob("r*/*/progress.json"))}
    print(json.dumps(state,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
