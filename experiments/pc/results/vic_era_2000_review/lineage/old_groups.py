"""Reapply fixed historical outcome labels to existing round-4 responses without a model call."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESULTS=HERE.parents[1]
TAX=RESULTS/"mmmu900_epochs_20261031/vict_comparison_20261103/question_comparison.json"
RUN=RESULTS/"era_a_pair_round"
FILES={"ordinary":RUN/"control_recovery/score/results.json","a_pair":RUN/"mmmu/candidate/candidate/full/results.json"}


def main():
    labels={r["id"]:r for r in json.loads(TAX.read_text(encoding="utf-8-sig"))}
    armrows={arm:{r["id"]:r for r in json.loads(path.read_text(encoding="utf8"))["records"]} for arm,path in FILES.items()}
    counts=defaultdict(Counter)
    changes=defaultdict(Counter)
    for rid,row in labels.items():
        gr=row["baseline_group"]
        t=counts[gr]
        t["historical_n"]+=1
        for m in ("baseline","epoch_2","epoch_3","epoch_4"):
            t[m]+=row["models"][m]["hit"]
        r=armrows["a_pair"][rid]
        assert r["question"]==row["question"]
        t["a_pair_hit"]+=r["grading"]["final"]["hit"]
        c=armrows["ordinary"].get(rid)
        if c:
            assert c["question"]==row["question"]
            t["ordinary_n"]+=1
            t["ordinary_hit"]+=c["grading"]["final"]["hit"]
            a,b=c["grading"]["final"]["hit"],r["grading"]["final"]["hit"]
            changes[gr][("both_right" if a and b else "gain" if b else "loss" if a else "both_wrong")]+=1
    out=dict(source_sha256={str(p.relative_to(RESULTS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [TAX,*FILES.values()]},
             counts=counts,ordinary_to_a_pair_common=changes,
             note="V/I/C/T fixed labels apply only to historically misanswered baseline rows. Originally correct rows lack error types, so regressions from such rows cannot be attributed to V/I/C with this taxonomy.")
    (HERE/"old_groups.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(out["counts"],ensure_ascii=False))


if __name__=="__main__":
    main()
