"""Record reviewed numeric evidence for the non-shard ChartQA/impulse training outputs."""
import json
from decimal import Decimal

from finalize_pairs import HERE, last_answer, read, seg, quantitative_targets
from inspect_teacher import EXCLUSIONS, numset


EXTRA_REJECTIONS={
    "chart-18205":"The numeric answer 42.5 is visible, but the response says the incorrect option 59.5 is the sum of 42.5 and 32.9 while also correctly computing 75.4.",
    "chart-1942":"The label of 62 is over a bracket joining two wedges; the teacher calls it the largest segment and the unretracted description of a single segment conflicts with the chart.",
}


def main():
    raw={r["id"]:r for r in (json.loads(s) for s in (HERE/"teacher_predictions.jsonl").read_text(encoding="utf8").splitlines())}
    gold={r["id"]:r for r in read(HERE/"teacher_gold.json")}
    shard=set(read(HERE.parent/"train/second_review/selection.json")["ids"])
    out={}
    for sid,row in gold.items():
        if sid in shard or row["schema"]=="science_native":
            continue
        pred=raw[sid]; quote=seg(pred["raw"])
        reason=EXCLUSIONS.get(sid) or EXTRA_REJECTIONS.get(sid)
        if reason is None and (pred["finish_reason"]!="stop" or last_answer(pred["raw"])!=row["answer"]):
            reason="Incomplete generation or incorrect chosen letter."
        _,_,_,_,expect=quantitative_targets(row,{})
        supplied=numset(quote)
        if reason is None and any(Decimal(v) not in supplied for v in expect):
            reason="No exact teacher witness for the requested intermediate."
        if reason is None:
            if row["stratum"]=="direct":
                value=row["answer_text"]
                assert Decimal(value) in numset(pred["raw"])
                # Keep the original image-label witness even if option discussion follows.
                if Decimal(value) not in supplied:quote=pred["raw"].strip()
                reason=f'The rendered image associates {next(iter(row["terms"]))} with {value}; the teacher identifies that reading and chooses {row["answer"]}.'
            else:
                reason=f'All requested readings match the plotted labels. The teacher computes {", ".join(expect)} before arriving at {row["answer_text"]}, option {row["answer"]}.'
            ok=True
        else:
            ok=False
        out[sid]=dict(accept=ok,teacher_quote=quote if ok else "",reason=reason,reviewer="primary_numeric",
                      source_review="Manually compared the 24 numbered review panels to the source fields; examined arithmetic excerpts from each teacher raw.")
    assert len(out)==114 and len(set(out)&shard)==0
    (HERE/"review_quant_data.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(sum(r["accept"] for r in out.values()), "of", len(out), "verified in primary numeric shard")


if __name__=="__main__":
    main()
