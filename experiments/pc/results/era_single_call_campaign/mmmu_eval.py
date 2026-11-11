"""One solver generation per checkpoint/item using the archived MMMU implementation.

The 90-item screen is a fixed, stratified development subset. Promotion locks a single
nominee, keeps its screen generations and judge decisions, and scores the full 900.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SEED = 3407
SCREEN_KEY = "era-single-call-screen-v1/"
BASELINE_RESULT = "3791f54f30b9e127fa995f2f442ab0460c988addb069377596ea5830a7a8f0d8"
PIXEL_FILE = "eafe5603b1de1af895d55a431a8792f7e53f700b89dcae2ecaf5ad9ff0ac6b74"
ROSTER_FILE = "13b00a2003f30e017df49c57f92781b97781f23c5fb578c808f59fc29e3bd612"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(data, out, ensure_ascii=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def freeze(path, data):
    if path.exists():
        require(read(path) == data, f"Immutable input changed: {path}")
    else:
        save(path, data)


def install_paths(args):
    sys.path.insert(0, str(args.project))
    sys.path.insert(0, str(args.project / "baseline/src"))
    import evaluate
    import prepare
    from era_lora.evaluate import SOURCE_HASHES, validate_frozen_config
    for rel, h in SOURCE_HASHES.items():
        require(sha(args.project / "baseline" / rel) == h, f"Archived solver/scorer source drift: {rel}")
    config = read(args.project / "baseline/protocol/config.seed3407.json")
    require(config == read(args.project / "baseline/src/config.json"), "Official config differs")
    validate_frozen_config(config, SEED)
    return evaluate, prepare, config


def plan(args):
    frozen, _, config = install_paths(args)
    roster_path = args.project / "baseline/protocol/evaluation_items.json"
    baseline_path = args.baseline / "baseline_seed3407/results.json"
    pixfile = args.baseline / "provenance/input-validation.json"
    require(sha(roster_path) == ROSTER_FILE and sha(baseline_path) == BASELINE_RESULT and
            sha(pixfile) == PIXEL_FILE, "Archived MMMU ID, baseline score or RGB checksum changed")
    roster = read(roster_path)
    archive = read(baseline_path)["records"]
    require(len(archive) == len(roster) == 900 and
            sum(r["grading"]["final"]["hit"] for r in archive) == 575, "575/900 baseline mismatch")
    ids = [r["id"] for r in roster]
    require(len(set(ids)) == 900 and [r["id"] for r in archive] == ids, "Original row order differs")
    for row, item in zip(archive, roster):
        fields = {k: row[k] for k in ("question","options","reference_answer","input_prompt","image_fields")}
        signature = hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        require(signature == item["input_sha256"], f"Archived input mismatch: {row['id']}")
    selected = set()
    for subject in config["dataset"]["configs"]:
        group = [r["id"] for r in roster if r["subject"] == subject]
        require(len(group) == 30, f"Roster count mismatch: {subject}")
        selected.update(sorted(group, key=lambda rid: hashlib.sha256((SCREEN_KEY + rid).encode()).digest())[:3])
    screen = [rid for rid in ids if rid in selected]
    require(len(screen) == 90, "Expected three per subject")
    plan_data = dict(seed=SEED, screen_key=SCREEN_KEY, screen_ids=screen, full_ids=ids,
                     screen_definition="3 lowest SHA256 values of (screen_key + id) per subject; original order",
                     archive_sha256=BASELINE_RESULT, roster_sha256=ROSTER_FILE, rgb_sha256=PIXEL_FILE,
                     solver_contract="one call per (checkpoint, item); unchanged prompt, engine, sampling and output budget")
    freeze(args.output_root / "roster.json", plan_data)
    for key, records in (("screen", [r for r in archive if r["id"] in selected]), ("full", archive)):
        file = args.output_root / f"{key}_baseline.json"
        freeze(file, dict(records=records, summary=dict(n=len(records), correct=sum(r["grading"]["final"]["hit"] for r in records)),
                          archive_sha256=BASELINE_RESULT, scope=key))
    return frozen, config, plan_data, {r["id"]:r for r in archive}


def reserve_full(root, name, checkpoint_fingerprint):
    # All processes visit here before any unblinded full item is submitted.
    import evaluate as frozen
    with frozen.run_lock(root / ".nominee.lock"):
        freeze(root / "nominee.json", dict(name=name, checkpoint=checkpoint_fingerprint))


def get_paths(args):
    home = Path.home() / "mmdl-cache/huggingface"
    return dict(model=str(home / "hub"), dataset=str(home / "datasets"))


def match_inputs(record, archived):
    keys = ("id","subject","dataset_index","question_type","question","options","reference_answer",
            "input_prompt","rendered_chat","image_fields","original_image_sizes_wh")
    require(all(record[k] == archived[k] for k in keys), f"Input differs from baseline for {record['id']}")
    require(record["model_response"]["input_tokens"] == archived["model_response"]["input_tokens"],
            f"Tokenized input differs from baseline for {record['id']}")


def verify_judge(args, config, manifest, run):
    import prepare
    prov=args.baseline/"provenance"
    require(sha(prov/"preparation.json")=="90bbb65cf9b085eb680c6ad5de665809b7046d27fa982f7776a89d7c57e9cc90"
            and sha(prov/"model-auxiliary-hashes.json")=="6c3e387355d8223eb958e7aafa174ac6cbb9aeecb95ac639c797aeca51fa2886",
            "Judge provenance inventory changed")
    snapshot=prepare.snapshot(config,manifest["cache_paths"]["model"],judge=True)
    expected={**read(prov/"preparation.json")["model_sha256"]["judge"],**read(prov/"model-auxiliary-hashes.json")["judge"]}
    require(snapshot.name==config["judge"]["revision"] and all(sha(snapshot/f)==h for f,h in expected.items()),
            "Judge weights or tokenizer differ from archived baseline")
    freeze(run/"judge_identity.json",dict(revision=snapshot.name,files=expected))


def inference(args, run, scope_ids, archive, config, manifest):
    """Call unmodified frozen infer; restrict only the row list and add a durable per-ID submission ledger."""
    import evaluate as frozen
    import prepare
    import vllm
    original_load = prepare.load_data
    original_append = prepare.append
    original_engine = vllm.LLM
    pixels = {(x["id"], x["field"]): x for x in read(args.baseline / "provenance/input-validation.json")["images"]}
    wanted = set(scope_ids)
    existing = prepare.read_jsonl(run / "predictions.jsonl")
    seen = set()
    for r in existing:
        require(r["id"] in wanted and r["id"] not in seen, "Unexpected or duplicate solver ID")
        match_inputs(r, archive[r["id"]])
        seen.add(r["id"])
    intent = run / "solver_inflight.json"
    if intent.exists():
        rid = read(intent)["id"]
        require(rid in seen, f"An interrupted solver request for {rid} has no committed response. Refusing a second solver call.")
        intent.unlink()
    pending = [rid for rid in scope_ids if rid not in seen]
    if not pending:
        return
    index = 0

    def scoped_data(c, cache, offline=True):
        datasets, rows = original_load(c, cache, offline)
        chosen = [r for r in rows if r["id"] in wanted]
        require([r["id"] for r in chosen] == scope_ids, "Fresh dataset roster mismatch")
        for row in chosen:
            rid = row["id"]; archived = archive[rid]
            require(frozen.prompt_for(row, c) == archived["input_prompt"], f"Frozen raw prompt mismatch: {rid}")
            source = datasets[row["subject"]][row["dataset_index"]]
            fields = [f"image_{i}" for i in range(1,8) if source.get(f"image_{i}") is not None]
            require(fields == archived["image_fields"], f"Image roster mismatch: {rid}")
            for field in fields:
                image = source[field].convert("RGB")
                ref = pixels[rid, field]
                require(list(image.size) == ref["size_wh"] and hashlib.sha256(image.tobytes()).hexdigest() == ref["rgb_bytes_sha256"],
                        f"RGB hash mismatch: {rid}/{field}")
        return datasets, chosen

    class GuardedEngine:
        def __init__(self, *a, **kw):
            self.engine = original_engine(*a, **kw)

        def generate(self, prompts, *a, **kw):
            require(index < len(pending) and not intent.exists() and len(prompts) == 1, "Duplicate solver submission")
            rid = pending[index]
            require(prompts[0]["prompt"] == archive[rid]["rendered_chat"], f"Rendered prompt drift: {rid}")
            save(intent, {"id":rid})
            return self.engine.generate(prompts, *a, **kw)

    def append_once(path, record):
        nonlocal index
        require(path == run / "predictions.jsonl" and record["id"] == pending[index] == read(intent)["id"],
                "Solver output does not match in-flight row")
        match_inputs(record, archive[record["id"]])
        original_append(path, record)
        intent.unlink()
        index += 1

    with ExitStack() as stack:
        stack.enter_context(patch.object(prepare, "load_data", scoped_data))
        stack.enter_context(patch.object(prepare, "append", append_once))
        stack.enter_context(patch.object(vllm, "LLM", GuardedEngine))
        frozen.infer(SimpleNamespace(checkpoint=str(args.checkpoint)), config, run, manifest)
    require(index == len(pending), "Missing scheduled solver outputs")


def graded(records, attempts, config):
    import evaluate as frozen
    import parsers
    last = frozen.saved_judges(records, attempts, config)
    pending = []
    for r in records:
        result = parsers.grade_response(r)
        if result["status"] == "pending_judge":
            prev = last.get(r["id"])
            if prev is not None:
                parsed = parsers.grade_judge(r, prev["raw"])
                if parsed["judge_parser_decision"]["decision_parsed"] or prev["attempt"] >= config["judge"]["max_attempts"]:
                    result = parsed
            if result["status"] == "pending_judge":
                pending.append(r["id"])
        r["grading"] = {"final":result}
    return pending


def evaluate(args, config, roster, archive):
    import evaluate as frozen
    import parsers
    import prepare
    from era_lora.evaluate import checkpoint_preflight
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", args.name), "Invalid candidate name")
    scope_ids = roster[args.scope + "_ids"]
    run = args.output_root / args.name
    run.mkdir(parents=True, exist_ok=True)
    cp_info = checkpoint_preflight(args.checkpoint)
    manifest = dict(configuration=config, fingerprint=hashlib.sha256(json.dumps(cp_info, sort_keys=True).encode()).hexdigest(),
                    cache_paths=get_paths(args), model_identity=dict(path=str(args.checkpoint),hashes=cp_info),
                    source_sha256=sha(Path(__file__)), baseline_sha256=BASELINE_RESULT,
                    pixels_sha256=PIXEL_FILE, solver_calls_per_item=1)
    with frozen.run_lock(run / ".run.lock"):
        freeze(run/"manifest.json", manifest)
        if args.scope == "full":
            require((run/"screen/complete.json").is_file(), "Complete the screen before nominating")
            screen_proof=read(run/"screen/complete.json")
            require(all(sha(run/"screen"/f)==h for f,h in screen_proof["outputs"].items()),
                    "Initial screen scoring artifacts changed")
            reserve_full(args.output_root, args.name, manifest["fingerprint"])
        dest = run / args.scope
        if (dest/"complete.json").exists():
            proof=read(dest/"complete.json")
            require(all(sha(dest/f)==h for f,h in proof["outputs"].items()), "Completed output checksum mismatch")
            print(json.dumps(read(dest/"summary.json")),flush=True)
            return
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
        os.environ.pop("VLLM_BATCH_INVARIANT",None)
        parsers.selfcheck()
        command=[sys.executable, str(Path(__file__).resolve()), "--name",args.name,"--checkpoint",str(args.checkpoint),
                 "--scope",args.scope,"--output-root",str(args.output_root),"--project",str(args.project),"--baseline",str(args.baseline)]
        if not args.score_only:
            subprocess.run(command+["--_stage","solver"],check=True)
        preds=prepare.read_jsonl(run/"predictions.jsonl")
        require(len(preds)==len(set(r["id"] for r in preds))==len(scope_ids) and set(scope_ids)=={r["id"] for r in preds},
                "Incomplete solver outputs")
        byid={r["id"]:r for r in preds}
        records=[byid[x] for x in scope_ids]
        frozen.check_records(records,config,partial=args.scope=="screen")
        for r in records:match_inputs(r,archive[r["id"]])
        attempts=prepare.read_jsonl(run/"judge_attempts.jsonl")
        pending=graded(records,attempts,config)
        if pending:
            require(not args.score_only,"Judge extraction still required")
            save(run/"pending_judge.json",pending)
            subprocess.run(command+["--_stage","judge"],check=True)
            require(not graded(records,prepare.read_jsonl(run/"judge_attempts.jsonl"),config),"Missing sealed judge output")
        basehits=[bool(archive[rid]["grading"]["final"]["hit"]) for rid in scope_ids]
        hits=[bool(r["grading"]["final"]["hit"]) for r in records]
        methods=Counter(r["grading"]["final"]["method"] for r in records)
        summary=dict(scope=args.scope,n=len(records),correct=sum(hits),baseline_correct=sum(basehits),
                     accuracy=sum(hits)/len(hits),accuracy_percent=100*sum(hits)/len(hits),
                     paired_gained=sum(b and not a for a,b in zip(basehits,hits)),paired_lost=sum(a and not b for a,b in zip(basehits,hits)),
                     length=sum(r["model_response"]["finish_reason"]=="length" for r in records),
                     judge_count=methods["local_qwen3_8b_judge_parser"],grading_methods=dict(methods))
        if args.scope=="full":
            screen={r["id"]:r for r in read(run/"screen/results.json")["records"]}
            require(all(byid[rid]==r for rid,r in screen.items()),"An already graded screen row changed in full evaluation")
            held=[r for r in records if r["id"] not in screen]
            require(len(held)==810,"Expected 810 untouched rows")
            summary["remaining810"]=dict(n=len(held),correct=sum(r["grading"]["final"]["hit"] for r in held),
                                         baseline_correct=sum(archive[r["id"]]["grading"]["final"]["hit"] for r in held))
        save(dest/"summary.json",summary)
        save(dest/"results.json",dict(records=records,summary=summary,manifest=manifest))
        save(dest/"complete.json",dict(scope=args.scope,n=len(records),outputs={f:sha(dest/f) for f in ("summary.json","results.json")},
                                     model=manifest["fingerprint"],predictions_sha256=sha(run/"predictions.jsonl")))
        print(json.dumps(summary),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--plan",action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--checkpoint",type=Path)
    ap.add_argument("--scope",choices=["screen","full"],default="screen")
    ap.add_argument("--score-only",action="store_true",help="Use previously saved solver and judge outputs without any new GPU calls.")
    ap.add_argument("--project",type=Path,default=Path.home()/"era_decoder_lora_20261030")
    ap.add_argument("--baseline",type=Path,default=Path.home()/"mmmu_baseline_20261009")
    ap.add_argument("--output-root",type=Path,default=HERE/"benchmark")
    ap.add_argument("--_stage",choices=["solver","judge"],help=argparse.SUPPRESS)
    a=ap.parse_args()
    frozen,config,roster,archive=plan(a)
    if a.plan:
        print(json.dumps(dict(screen=len(roster["screen_ids"]),screen_baseline=sum(archive[k]["grading"]["final"]["hit"] for k in roster["screen_ids"]),
                              full=len(roster["full_ids"]),full_baseline=575)),flush=True)
        return
    require(a.name and a.checkpoint,"--name and --checkpoint are required")
    a.checkpoint=a.checkpoint.resolve()
    if a._stage:
        run=a.output_root/a.name
        manifest=read(run/"manifest.json")
        require(a.name and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*",a.name) and
                str(a.checkpoint)==manifest["model_identity"]["path"] and manifest["source_sha256"]==sha(Path(__file__)),
                "Child process received a different solver identity")
        if a._stage=="solver":
            inference(a,run,roster[a.scope+"_ids"],archive,config,manifest)
        else:
            import judge
            verify_judge(a,config,manifest,run)
            judge.run(config,run,manifest)
    else:
        evaluate(a,config,roster,archive)


if __name__=="__main__":
    main()
