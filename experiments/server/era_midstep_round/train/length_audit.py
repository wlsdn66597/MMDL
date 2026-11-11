"""Describe supervision amount in each arm before either candidate is trained."""
from collections import defaultdict
import argparse
import json
from statistics import median

import train_midstep as train


def compare(rows):
    standard = [r["standard"] for r in rows]
    preserved = [r["preserved"] for r in rows]
    deltas = [p-s for p,s in zip(preserved,standard)]
    return dict(n=len(rows),edited=sum(r["edited"] for r in rows),
                standard_total=sum(standard),preserved_total=sum(preserved),
                aggregate_percent_change=100*(sum(preserved)/sum(standard)-1),
                median_standard=median(standard),median_preserved=median(preserved),median_delta=median(deltas),
                preserved_longer=sum(d>0 for d in deltas),preserved_shorter=sum(d<0 for d in deltas),
                delta_range=[min(deltas),max(deltas)])


def preview(rows):
    config = train.load_config(train.engine.CFG)
    processor = train.load_processor(config)
    samples = []
    for r in rows:
        lengths = {a:len(processor.tokenizer(r[key], add_special_tokens=False).input_ids)+1
                   for a,key in train.ARM_FIELD.items()}
        samples.append(dict(id=r["id"],stratum=r["stratum"],target_tokens=lengths))
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preview",action="store_true",help="Measure candidate SFT targets without freezing the final preflight.")
    args = p.parse_args()
    new = {r["id"]:r for r in train.new_rows()}
    if args.preview:
        samples = preview(train.shared_rows(train.load_config(train.engine.CFG))+list(new.values()))
    else:
        samples = train.engine.read(train.PREFLIGHT)["samples"]
    rows, grouped = [], defaultdict(list)
    for sample in samples:
        rid = sample["id"]
        r = new.get(rid)
        src = "shared_era200" if r is None else r.get("source") or r.get("domain") or "source_unspecified"
        edited = r is not None and r["control_response"]!=r["midstep_response"]
        entry = dict(id=rid,source=src,stratum=sample["stratum"],edited=edited,**sample["target_tokens"])
        labels = ["all","source/"+src,"stratum/"+sample["stratum"]]
        if r is not None:
            labels.extend(["new","new/edited" if edited else "new/identical"])
            if edited:
                labels.extend(["edited_source/"+src,"edited_stratum/"+sample["stratum"]])
        rows.append(entry)
        for label in labels:
            grouped[label].append(entry)
    summary = {name:compare(items) for name,items in grouped.items()}
    result = dict(preflight_sha256=None if args.preview else train.engine.sha(train.PREFLIGHT),summary=summary,
                  samples=rows,measurement="Qwen3-VL pinned processor; positive assistant tokens including EOS per epoch.")
    if args.preview:
        train.write_json(train.HERE/"length_preview.json",result)
    else:
        train.freeze(train.HERE/"length_audit.json",result)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
