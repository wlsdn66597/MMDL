"""Compare uniform ERA SFT targets using the archived fresh-base trainer on aix408-40."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ADDED = DATA / "train_pairs.jsonl"
RELEASE = DATA / "train_receipt.json"
PREFLIGHT = HERE / "data/receipt.json"
ARM_FIELD = {"standard":"control_response", "preserved":"midstep_response"}
PROJECT = Path.home() / "era_decoder_lora_20261030"
ARCHIVE = Path.home() / "era_single_call_campaign"
sys.path.insert(0, str(ARCHIVE))
sys.path.insert(0, str(PROJECT))
import train_round as engine
from era_lora.common import canonical_json_hash, load_config, write_json
from era_lora.data import AssistantOnlyCollator, load_paired_records, load_processor
from era_lora.train import file_lock


ENGINE_SHA = "a7a2b819fec2208faf91e0f905f83e5e13964d958e61302a13802c86c35b80c1"
SHARED_SHA = "6c19160e0a60b568854a8377d330a04300ffab52484d6143da4b3f12aa1afa53"
TARGET = re.compile(r"^Evidence: (.+?)\nRule: (.+?)\nApplication: (.+?)\nFinal answer: ([^\n]+)$", re.S)


def require(check, description):
    if not check:
        raise ValueError(description)


def freeze(path, value):
    if path.exists():
        require(engine.read(path)==value, f"Frozen file changed: {path}")
    else:
        write_json(path,value)


def shared_rows(cfg):
    records, manifest = load_paired_records(cfg)
    require(manifest["items"]==200 and manifest["files"][cfg["dataset"]["era_jsonl"]]==SHARED_SHA,
            "The sealed common ERA training set changed")
    rows = []
    for record in records:
        target = record["response_era"]
        rows.append(dict(id=record["sample_id"], image_path=record["image_path"], image_sha256=record["image_sha256"],
                         prompt=record["input_prompt"], control_response=target, midstep_response=target,
                         stratum="shared200"))
    return rows


def new_rows():
    receipt = engine.read(RELEASE)
    require(receipt["verified"] is True and receipt["train_sha256"]==engine.sha(ADDED),
            "Unverified or changed paired data")
    data = [json.loads(line) for line in ADDED.read_text(encoding="utf8").splitlines()]
    require(len(data)==receipt["n"] and 0<len(data)<=192,"Selected pair count mismatch")
    for row in data:
        path = (DATA / row["image"]).resolve()
        require(path.is_relative_to(DATA.resolve()) and path.is_file() and engine.sha(path)==row["image_sha256"],
                f"Image mismatch: {row['id']}")
        row["image_path"] = str(path)
        targets = [TARGET.fullmatch(row[k]) for k in ARM_FIELD.values()]
        require(all(targets) and targets[0].group(4)==targets[1].group(4),f"Different final targets: {row['id']}")
        require(all(part.strip() for target in targets for part in target.groups()),f"Empty ERA field: {row['id']}")
    require(any(r["control_response"]!=r["midstep_response"] for r in data),
            "There is no supervised target difference to test")
    return data


def feature(row, arm):
    return dict(sample_id=row["id"], image_path=row["image_path"], image_sha256=row["image_sha256"],
                input_prompt=row["prompt"], response=row[ARM_FIELD[arm]])


def code_identity():
    return dict(trainer=engine.sha(Path(__file__)), engine=engine.sha(Path(engine.__file__)),
                optimizer_helper=engine.sha(Path(engine.trainer_module().__file__)), config=engine.sha(engine.CFG))


def measure(values):
    return dict(n=len(values),total=sum(values),mean=statistics.mean(values),
                median=statistics.median(values),min=min(values),max=max(values))


def preflight(rows, config, common_only):
    processor = load_processor(config)
    collator = AssistantOnlyCollator(processor,config)
    audits = []
    lengths = defaultdict(lambda: defaultdict(list))
    for row in rows:
        reports = {}
        for arm in ARM_FIELD:
            # Identical targets share the tokenization measurement, which is independent of the arm name.
            if arm=="preserved" and row["control_response"]==row["midstep_response"]:
                reports[arm] = reports["standard"]
            else:
                b, report = collator.encode(feature(row,arm))
                require(report["assistant_end_in_loss"] and report["supervised_tokens"]==int(b["labels"].ne(-100).sum()),
                        f"Invalid supervision mask: {row['id']}")
                reports[arm] = report
            for group in ("all", row["stratum"]):
                lengths[group][arm].append(reports[arm]["supervised_tokens"])
        for key in ("prefix_token_ids_sha256","input_prompt_sha256","image_sha256","image_grid_thw",
                    "processed_image_sizes_wh","input_tokens"):
            require(reports["standard"][key]==reports["preserved"][key],f"Input mismatch: {row['id']}/{key}")
        audits.append(dict(id=row["id"],stratum=row["stratum"],input_tokens=reports["standard"]["input_tokens"],
                           image_sha256=row["image_sha256"],prefix_sha256=reports["standard"]["prefix_token_ids_sha256"],
                           target_tokens={a:r["supervised_tokens"] for a,r in reports.items()}))
    identity = dict(verified=True,common_only=common_only,n=len(rows),common_sha256=SHARED_SHA,
                    rows_sha256=canonical_json_hash(rows),code=code_identity(),
                    epochs=2,updates=2*math.ceil(len(rows)/16),arm_target_files=dict(ARM_FIELD),
                    identical_id_prompt_and_image=True,new_targets_differ=sum(
                        r["control_response"]!=r["midstep_response"] for r in rows),
                    token_stats={g:{a:measure(v) for a,v in arms.items()} for g,arms in lengths.items()},
                    samples=audits)
    if not common_only:
        identity.update(selected_file_sha256=engine.sha(ADDED),release_sha256=engine.sha(RELEASE))
    path = HERE / "shared_preflight.json" if common_only else PREFLIGHT
    freeze(path,identity)
    print(json.dumps({k:v for k,v in identity.items() if k!="samples"}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--shared-preflight",action="store_true")
    p.add_argument("--arm",choices=tuple(ARM_FIELD))
    p.add_argument("--merge",action="store_true")
    args = p.parse_args()
    require(engine.sha(Path(engine.__file__))==ENGINE_SHA,"Original W0 training engine changed")
    config = load_config(engine.CFG)
    s = config["training"]
    require((s["seed"],s["data_seed"],s["gradient_accumulation_steps"],s["micro_batch_size"],s["learning_rate"])==
            (3407,3407,16,1,1e-4),"Training protocol mismatch")
    rows = shared_rows(config)
    if not args.shared_preflight:
        rows.extend(new_rows())
    require(len(set(r["id"] for r in rows))==len(rows),"Duplicate training ID")
    if args.preflight or args.shared_preflight:
        require(not args.arm and not args.merge,"CPU preflight must run separately")
        with file_lock(HERE/".preflight.lock"):
            preflight(rows,config,args.shared_preflight)
        return
    require(args.arm is not None,"Specify --arm or --preflight")
    proof = engine.read(PREFLIGHT)
    require(proof["verified"] and proof["code"]==code_identity() and proof["rows_sha256"]==canonical_json_hash(rows)
            and proof["release_sha256"]==engine.sha(RELEASE),"Inputs or code changed since CPU preflight")
    steps = dict(name="same_images_standard_or_preserved_ERA",epochs=2,lr=1e-4,preflight=engine.sha(PREFLIGHT))
    features = [feature(r,args.arm) for r in rows]
    args.round = 5
    # Both arm names take the archived engine's uniform, assistant-only SFT branch.
    # Only the certified target source differs. All optimizer, restoration, merge and identity checks are reused.
    with patch.multiple(engine,ROOT=HERE,ROUNDS={5:steps},dataset=lambda:features):
        engine.run(args)


if __name__=="__main__":
    main()
