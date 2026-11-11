"""Train matched single-call ERA models with identical inputs, E, R and terminal choices."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROWS_FILE = ROOT / "data/rows.jsonl"
PREFLIGHT = HERE / "data/receipt.json"
ARMS = ("ordinary", "enhanced")
GROUPS = {"A": 12, "SYMBOL": 6, "GENERAL": 6}
R6 = Path.home() / "vic_direction_round/train"
sys.path.insert(0, str(R6))
import train_direction as r6

previous = r6.previous
engine = r6.engine
require = r6.expect
R6_TRAINER = "56acb7d5cfa8bed81e1651357b457bb5cdec0451b03159d9c2fac76935c7ca5d"


def shared(cfg):
    rows = []
    for item in r6.shared(cfg):
        require(item["control_prompt"] == item["reverse_prompt"] and
                item["control_response"] == item["reverse_response"], f"Common target drift: {item['id']}")
        rows.append(dict(id=item["id"], group=item["group_id"], kind=item["stratum"],
                         image_path=item["image_path"], image_sha256=item["image_sha256"],
                         prompt=item["control_prompt"], ordinary=item["control_response"],
                         enhanced=item["control_response"]))
    return rows


def make_prompt(row):
    options = row["options"]
    require(isinstance(options, list) and len(options) == len(set(options)) and
            2 <= len(options) <= 26 and row["gold"] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]
            and len(row["gold"]) == 1, f"Invalid choices for {row['id']}")
    choices = "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(options))
    return (f"Question: {row['question']}\nOptions:\n{choices}\n"
            "Please select the correct answer from the options above.\n"
            'On the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.')


def render(target, gold):
    require(all(isinstance(target[k], str) and target[k].strip() for k in ("e", "r", "a")),
            "Every supervised section must contain a label.")
    text = f"Evidence: {target['e']}\nRule: {target['r']}\nApplication: {target['a']}\nFinal answer: {gold}"
    require(previous.TARGET.fullmatch(text).groups() == (target["e"], target["r"], target["a"], gold),
            "Malformed target")
    return text


def new_rows():
    groups = defaultdict(list)
    rows = []
    for line in ROWS_FILE.read_text(encoding="utf8").splitlines():
        item = json.loads(line)
        kind = item["kind"]
        require(kind in GROUPS, f"Unexpected task: {kind}")
        path = (ROOT / item["image"]).resolve()
        require(path.is_relative_to((ROOT/"data").resolve()) and path.is_file(), f"Invalid image: {path}")
        h = engine.sha(path)
        if "image_sha256" in item:
            require(item["image_sha256"] == h, f"Image hash drift for {item['id']}")
        targets = {arm:render(item[arm], item["gold"]) for arm in ARMS}
        require(all(item["ordinary"][k] == item["enhanced"][k] for k in ("e","r")),
                f"A fixed evidence or rule changed: {item['id']}")
        changed = targets["ordinary"] != targets["enhanced"]
        require(changed == (kind == "A"), f"Unexpected target change for {item['id']}")
        row = dict(id=item["id"], group=item["group"], kind=kind, prompt=make_prompt(item),
                   image_path=str(path), image_sha256=h, **targets)
        groups[item["group"]].append(row)
        rows.append(row)
    require(len(groups)==24 and len(rows)==48, "Expected 24 new images with two questions each.")
    types = Counter()
    images = set()
    for name, pair in groups.items():
        require(len(pair)==2 and pair[0]["kind"]==pair[1]["kind"] and
                pair[0]["image_sha256"]==pair[1]["image_sha256"] and
                pair[0]["prompt"]!=pair[1]["prompt"], f"Invalid pair: {name}")
        require(pair[0]["image_sha256"] not in images, f"Image belongs to two groups: {name}")
        images.add(pair[0]["image_sha256"])
        types[pair[0]["kind"]] += 1
    require(types == GROUPS, f"Unexpected group counts: {types}")
    return rows


def feature(row, arm):
    return dict(sample_id=row["id"], image_path=row["image_path"], image_sha256=row["image_sha256"],
                input_prompt=row["prompt"], response=row[arm])


def identity():
    return dict(trainer=engine.sha(Path(__file__)), r6_helper=engine.sha(Path(r6.__file__)),
                r5_helper=engine.sha(Path(previous.__file__)), optimizer=engine.sha(Path(engine.__file__)),
                checkpoint_helper=engine.sha(Path(engine.trainer_module().__file__)), config=engine.sha(engine.CFG))


def preflight(rows, cfg):
    collator = previous.AssistantOnlyCollator(previous.load_processor(cfg), cfg)
    lengths = defaultdict(lambda: {arm:[] for arm in ARMS})
    inputs = defaultdict(list)
    audit = []
    changed_ids = []
    for row in rows:
        different = row["ordinary"] != row["enhanced"]
        if different:
            changed_ids.append(row["id"])
        reports = {}
        for arm in ARMS:
            if arm == "enhanced" and not different:
                reports[arm] = reports["ordinary"]
            else:
                batch, report = collator.encode(feature(row,arm))
                require(report["assistant_end_in_loss"] and
                        int(batch["labels"].ne(-100).sum()) == report["supervised_tokens"],
                        f"Invalid loss mask at {row['id']}")
                reports[arm] = report
        for field in ("prefix_token_ids_sha256","input_prompt_sha256","image_sha256","image_grid_thw",
                      "processed_image_sizes_wh","input_tokens"):
            require(reports["ordinary"][field]==reports["enhanced"][field], f"Unequal {field} for {row['id']}")
        groups = ["all",row["kind"]]
        if different:
            groups.append("modified24")
        for key in groups:
            inputs[key].append(reports["ordinary"]["input_tokens"])
            for arm in ARMS:
                lengths[key][arm].append(reports[arm]["supervised_tokens"])
        audit.append(dict(id=row["id"],group=row["group"],kind=row["kind"],changed=different,
                          prefix_sha256=reports["ordinary"]["prefix_token_ids_sha256"],
                          image_sha256=row["image_sha256"],
                          target_tokens={a:reports[a]["supervised_tokens"] for a in ARMS}))
    require(len(changed_ids)==24, "A changes must affect exactly 24 targets")
    token_stats = {k:{arm:previous.measure(nums) for arm,nums in stats.items()} for k,stats in lengths.items()}
    diffs = {k:stats["enhanced"]["total"]-stats["ordinary"]["total"] for k,stats in token_stats.items()}
    explicit = token_stats["modified24"]
    require(abs(diffs["modified24"]) <= .02 * explicit["ordinary"]["total"], "Unmatched training target lengths")
    info = dict(verified=True,n=296,epochs=2,updates=38,changed_ids=changed_ids,
                pair_sha256=engine.sha(ROWS_FILE),shared200_sha256=previous.SHARED_SHA,
                direct_source_sha256=r6.R5_PAIRS_SHA,rows_sha256=previous.canonical_json_hash(rows),
                identical_input_prefixes=True,identical_evidence_rules_and_finals=True,
                token_stats=token_stats,input_stats={k:previous.measure(v) for k,v in inputs.items()},
                target_delta=diffs,samples=audit,code=identity())
    previous.freeze(PREFLIGHT,info)
    print(json.dumps({k:v for k,v in info.items() if k not in ("samples","changed_ids")}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--arm",choices=ARMS)
    parser.add_argument("--merge",action="store_true")
    args=parser.parse_args()
    require(engine.sha(Path(r6.__file__))==R6_TRAINER and
            engine.sha(Path(engine.__file__))==previous.ENGINE_SHA and
            engine.sha(Path(previous.__file__))==r6.R5_TRAINER_SHA, "Archived helper drift")
    cfg=previous.load_config(engine.CFG)
    train=cfg["training"]
    require((train["seed"],train["data_seed"],train["gradient_accumulation_steps"],train["micro_batch_size"],
             train["learning_rate"])==(3407,3407,16,1,1e-4),"Optimizer mismatch")
    rows=shared(cfg)+new_rows()
    require(len(rows)==len({r["id"] for r in rows})==296,"Roster mismatch")
    require(not {r["image_sha256"] for r in rows[:248]} & {r["image_sha256"] for r in rows[248:]},
            "New group duplicates a common training image")
    if args.preflight:
        require(not args.arm and not args.merge,"Preflight is a separate stage")
        with previous.file_lock(HERE/".preflight.lock"):
            preflight(rows,cfg)
        return
    require(args.arm,"Pass --arm or --preflight")
    proof=engine.read(PREFLIGHT)
    require(proof["verified"] and proof["code"]==identity() and proof["n"]==296 and proof["updates"]==38 and
            proof["rows_sha256"]==previous.canonical_json_hash(rows) and proof["pair_sha256"]==engine.sha(ROWS_FILE),
            "Training inputs differ from preflight")
    hp=dict(name="matched_ERA_application",epochs=2,lr=1e-4,preflight=engine.sha(PREFLIGHT))
    sequence=[feature(r,args.arm) for r in rows]
    args.round=9
    with patch.multiple(engine,ROOT=HERE,ROUNDS={9:hp},dataset=lambda:sequence):
        engine.run(args)


if __name__=="__main__":
    main()
