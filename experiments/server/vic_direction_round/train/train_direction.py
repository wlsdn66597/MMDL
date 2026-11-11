"""Match the two orientation-training arms on the same initial model and image exposures."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
PAIR_FILE = DATA / "direction_pairs.jsonl"
PAIR_RECEIPT = DATA / "direction_receipt.json"
PREFLIGHT = HERE / "data/receipt.json"
R5 = Path.home() / "era_midstep_round"
sys.path.insert(0, str(R5 / "train"))
import train_midstep as previous

engine = previous.engine
expect = previous.require
ARMS = ("control", "reverse")
R5_TRAINER_SHA = "9c1fb13b4426ac4600d1bb6961832aaff47fe72668ac930cbe7d245b6cc91ca9"
R5_PAIRS_SHA = "1ce3771d18e1dee76dbb4d9ae846e4aadb32ef6f3a1597f40ce2d5a08bb03716"
R5_RECEIPT_SHA = "e080f38be940950ed5446ee65b5543f0f893cb3e57370e2e41bc114e194a9e11"


def twin(row, name, response_field):
    return dict(id=row["id"], image_path=row["image_path"], image_sha256=row["image_sha256"],
                group_id=row["id"], stratum=name, role="shared",
                control_prompt=row["prompt"], reverse_prompt=row["prompt"],
                control_response=row[response_field], reverse_response=row[response_field])


def shared(cfg):
    expect(engine.sha(previous.ADDED)==R5_PAIRS_SHA and engine.sha(previous.RELEASE)==R5_RECEIPT_SHA,
           "R5 training release changed")
    shared_200 = [twin(row, "shared200", "control_response") for row in previous.shared_rows(cfg)]
    direct=[]
    for row in previous.new_rows():
        if row["stratum"]=="direct":
            expect(row["control_response"]==row["midstep_response"],f"R5 direct targets differ: {row['id']}")
            direct.append(twin(row,"r5_direct","control_response"))
    expect(len(direct)==48 and len(shared_200)==200,"Expected fixed common248")
    return shared_200+direct


def options(prompt):
    found=re.findall(r"(?m)^([A-Z])\. (.+)$",prompt)
    expect(found and [k for k,v in found]==list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(found)]),"Malformed option list")
    expect(len({v.strip() for k,v in found})==len(found),"Duplicate option text")
    return {k:v.strip() for k,v in found}


def final_choice(response):
    target=previous.TARGET.fullmatch(response)
    expect(target and all(s.strip() for s in target.groups()),"Expected nonempty E/R/A/final")
    return target.group(4)


def paired():
    metadata=engine.read(PAIR_RECEIPT)
    expect(metadata["verified"] is True and metadata["sha256"]==engine.sha(PAIR_FILE),"Direction release is unverified")
    rows=[json.loads(s) for s in PAIR_FILE.read_text(encoding="utf8").splitlines()]
    groups=defaultdict(dict)
    for row in rows:
        p=(DATA/row["image"]).resolve()
        expect(p.is_relative_to(DATA.resolve()) and p.is_file() and engine.sha(p)==row["image_sha256"],
               f"Invalid image: {row['id']}")
        row.update(image_path=str(p),stratum="vic_"+row["role"])
        group=groups[row["group_id"]]
        role=row["role"]
        expect(role in ("original","alternate") and role not in group, f"Repeated group slot: {row['id']}")
        group[role]=row
        cp,rp=row["control_prompt"],row["reverse_prompt"]
        ct,rt=row["control_response"],row["reverse_response"]
        opts=options(cp)
        expect(opts==options(rp),f"Choices differ across arms: {row['id']}")
        ac,ar=final_choice(ct),final_choice(rt)
        expect(ac in opts and ar in opts, f"Invalid target option: {row['id']}")
        if role=="original":
            expect(cp==rp and ct==rt, f"Shared original changed: {row['id']}")
        else:
            expect(cp!=rp and opts[ac]!=opts[ar], f"Direction did not change the answer: {row['id']}")
    expect(len(rows)==metadata["n"]==48 and len(groups)==metadata["groups"]==24, "Expected 24 paired VIC groups")
    image_hashes=set()
    for group, slots in groups.items():
        expect(set(slots)=={"original","alternate"},f"Incomplete group: {group}")
        first,second=slots["original"],slots["alternate"]
        image=first["image_sha256"]
        expect(image not in image_hashes and image==second["image_sha256"],f"Unexpected exposure groups: {group}")
        image_hashes.add(image)
        for arm in ARMS:
            expect(first[arm+"_prompt"]!=second[arm+"_prompt"],f"Variant repeats the question: {group}/{arm}")
        a=options(first["control_prompt"])
        b=options(second["control_prompt"])
        expect(set(a.values())==set(b.values()),f"Group choices changed beyond order: {group}")
        expect(a[final_choice(first["control_response"])]==b[final_choice(second["control_response"])],
               f"Paraphrase changed semantic answer: {group}")
    return rows


def feature(row, arm):
    return dict(sample_id=row["id"],image_path=row["image_path"],image_sha256=row["image_sha256"],
                input_prompt=row[arm+"_prompt"],response=row[arm+"_response"])


def identity():
    return dict(trainer=engine.sha(Path(__file__)),r5_helper=engine.sha(Path(previous.__file__)),
                optimizer=engine.sha(Path(engine.__file__)),checkpoint_helper=engine.sha(Path(engine.trainer_module().__file__)),
                config=engine.sha(engine.CFG))


def preflight(rows,cfg,common_only):
    proc=previous.load_processor(cfg)
    collator=previous.AssistantOnlyCollator(proc,cfg)
    audited=[]
    stats=defaultdict(lambda: {arm:[] for arm in ARMS})
    input_stats=defaultdict(lambda: {arm:[] for arm in ARMS})
    for row in rows:
        matched=row["role"]!="alternate"
        reports={}
        for arm in ARMS:
            if arm=="reverse" and matched:
                reports[arm]=reports["control"]
            else:
                batch,report=collator.encode(feature(row,arm))
                expect(report["assistant_end_in_loss"] and int(batch["labels"].ne(-100).sum())==report["supervised_tokens"],
                       f"Unsupervised EOS or masking error: {row['id']}")
                reports[arm]=report
            for key in ("all",row["stratum"]):
                stats[key][arm].append(reports[arm]["supervised_tokens"])
                input_stats[key][arm].append(reports[arm]["input_tokens"])
        for key in ("image_sha256","image_grid_thw","processed_image_sizes_wh"):
            expect(reports["control"][key]==reports["reverse"][key],f"Visual input changed: {row['id']}")
        same=reports["control"]["prefix_token_ids_sha256"]==reports["reverse"]["prefix_token_ids_sha256"]
        expect(same==matched, f"Unexpected tokenized-prefix equality: {row['id']}")
        audited.append(dict(id=row["id"],group=row["group_id"],role=row["role"],stratum=row["stratum"],
                            image_sha256=row["image_sha256"],prefix={a:reports[a]["prefix_token_ids_sha256"] for a in ARMS},
                            target_tokens={a:reports[a]["supervised_tokens"] for a in ARMS}))
    info=dict(verified=True,common_only=common_only,n=len(rows),shared200_sha256=previous.SHARED_SHA,
              direct_source_sha256=R5_PAIRS_SHA,rows_sha256=previous.canonical_json_hash(rows),code=identity(),
              epochs=2,updates=2*math.ceil(len(rows)/16),strata=dict(Counter(r["stratum"] for r in rows)),samples=audited)
    for k,data in (("token_stats",stats),("input_stats",input_stats)):
        info[k]={group:{arm:previous.measure(vals) for arm,vals in arms.items()} for group,arms in data.items()}
    if not common_only:
        info.update(pair_sha256=engine.sha(PAIR_FILE),pair_receipt_sha256=engine.sha(PAIR_RECEIPT))
    previous.freeze(HERE/"common_preflight.json" if common_only else PREFLIGHT,info)
    print(json.dumps({k:v for k,v in info.items() if k!="samples"},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--common-preflight",action="store_true")
    parser.add_argument("--arm",choices=ARMS)
    parser.add_argument("--merge",action="store_true")
    args=parser.parse_args()
    expect(engine.sha(Path(engine.__file__))==previous.ENGINE_SHA and engine.sha(Path(previous.__file__))==R5_TRAINER_SHA,
           "Pinned trainer implementation changed")
    config=previous.load_config(engine.CFG)
    hp=config["training"]
    expect((hp["seed"],hp["data_seed"],hp["gradient_accumulation_steps"],hp["micro_batch_size"],hp["learning_rate"])==
           (3407,3407,16,1,1e-4),"Optimizer configuration drift")
    rows=shared(config)
    if not args.common_preflight:
        rows+=paired()
    expect(len(rows)==len({r["id"] for r in rows}),"Duplicate sample IDs")
    if args.preflight or args.common_preflight:
        expect(args.arm is None and not args.merge,"Preflight runs separately")
        with previous.file_lock(HERE/".preflight.lock"):
            preflight(rows,config,args.common_preflight)
        return
    expect(args.arm is not None, "Pass --arm or --preflight")
    proof=engine.read(PREFLIGHT)
    expect(proof["n"]==296 and proof["updates"]==38 and proof["verified"] and proof["code"]==identity() and
           proof["rows_sha256"]==previous.canonical_json_hash(rows) and proof["pair_receipt_sha256"]==engine.sha(PAIR_RECEIPT),
           "Changed dataset, source, or optimizer since preflight")
    schedule=dict(name="paraphrase_or_reverse_relation",epochs=2,lr=1e-4,preflight=engine.sha(PREFLIGHT))
    features=[feature(row,args.arm) for row in rows]
    args.round=6
    with patch.multiple(engine,ROOT=HERE,ROUNDS={6:schedule},dataset=lambda:features):
        engine.run(args)


if __name__=="__main__":
    main()
