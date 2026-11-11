"""Count visible, first-E questions for the original D prompts; ignore inference-time answers."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding="utf8"))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def inp(path):return list(map(json.loads,path.read_text(encoding="utf8").splitlines()))
def normalized(s):return " ".join(re.findall(r"[a-z0-9]+",s.lower()))

def main():
    groups=read(ROOT/"groups.json");sample={r["id"]:r for r in read(ROOT/"inputs.json")}
    targets={r["id"]:r["checked_focus"] for r in inp(ROOT.parent/"data/selection.jsonl")}
    result={};details=[]
    for a in ("standard","checked"):
        rows=inp(ROOT/"eval"/a/"predictions.jsonl")
        man=read(ROOT/"eval"/a/"manifest.json")
        assert man["weights"]==read(ROOT.parent/"eval"/a/"manifest.json")["weights"]
        assert man["inputs_sha256"]==sha(ROOT/"inputs.json") and man["eval_script_sha256"]==sha(ROOT/"probe.py")
        assert [r["id"] for r in rows]==list(sample)
        flags={g:Counter() for g in ("direct","selected")}
        for r in rows:
            rid=r["id"];group=groups[rid];c=flags[group];out=r["raw"]
            assert (r["prompt"],r["image_sha256"])==(sample[rid]["input_prompt"],sample[rid]["image_sha256"])
            e=re.search(r"(?ims)^\s*(?:\*\*)?Evidence\s*:(?:\*\*)?\s*(.*?)(?=^\s*(?:\*\*)?(?:Rule|Application|Final answer)\s*:|\Z)",out)
            et=e[1].strip() if e else ""
            q=[j.strip() for j in re.findall(r"[^.?!\n]*\?",et) if j.strip(" ?")]
            exact=any(normalized(i)==normalized(targets.get(rid,"")) for i in q)
            c["n"]+=1;c["with_E"]+=bool(e);c["with_question"]+=bool(q);c["exact"]+=exact
            c["length"]+=r["finish_reason"]=="length";c["E_may_be_truncated"]+=bool(e and e.end()==len(out) and r["finish_reason"]=="length")
            details.append(dict(id=rid,arm=a,group=group,E=et,questions=q,target=targets.get(rid),exact=exact))
        result[a]=flags
    for f,obj in [("summary.json",result),("audit.json",details)]:
        (ROOT/f).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result))
if __name__=="__main__":main()
