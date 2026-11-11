"""Evaluate new inputs using the archived solver, blind rubric and MMMU contract."""
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
HOME = Path.home()
R9 = HOME / "era_a_only_round"
HALF = HOME / "era_half_strength"
PROJECT = HOME / "era_decoder_lora_20261030"
EV = ROOT / "eval"
ARMS = ("base", "all50", "last50")
sys.path[:0] = [str(HALF), str(PROJECT)]
import scale


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def readlines(path):
    return list(map(json.loads, path.read_text().splitlines()))


def checkpoint(arm):
    return scale.BASE if arm == "base" else ROOT / "exports" / arm


def export(arm):
    from era_lora import train
    target = checkpoint(arm)
    receipt = ROOT / (arm + "-weights.json")
    if target.exists() and receipt.exists():
        assert scale.weights(target) == scale.read(receipt)["weights"]
        return
    temp = ROOT / "exports" / (arm + ".staging")
    temp.mkdir(exist_ok=True, parents=True)
    if arm == "all50":
        proof = scale.export("ordinary", scale.source("ordinary"), temp)
        assert proof == scale.read(HALF / "scale_receipts/ordinary.json")
    else:
        source = ROOT / "models/r9/ordinary"
        manifest = scale.read(source / "manifest.json")
        assert (ROOT / "training_verified").is_file() and manifest["frozen_verified"]
        assert scale.sha(source / "adapter/adapter_model.safetensors") == manifest["adapter_file_sha256"]
        original = train.discover_decoder_targets
        def last(model):
            return [n for n in original(model) if ".layers.35." in n]
        with patch.object(train, "discover_decoder_targets", last):
            proof = scale.export("last50", source, temp)
        assert proof["bf16_export_checked_targets"] == 4
        proof.pop("r9_training_result_sha256")
        proof["last_block_manifest_sha256"] = scale.sha(source / "manifest.json")
    scale.freeze(receipt, proof)
    temp.rename(target)


def infer(arm):
    spec = EV / "spec"
    source = spec / "inputs.jsonl"
    assert scale.sha(source) == "09da9c09700ddb3608ee9333894bb194d4d354b05fab1a51954597a4f2423723"
    assert scale.sha(spec / "gold.jsonl") == "d0202d06dca278a76d1e4a7e56a5a3d6005b6c673065c3fca3fce075200001b3"
    data = readlines(source)
    assert len(data) == len({x["id"] for x in data}) == 192
    images = []
    for row in data:
        assert set(row) == {"id","prompt","image","image_sha256","dataset"}
        path = (ROOT / row["image"]).resolve()
        assert path.is_relative_to(ROOT) and scale.sha(path) == row["image_sha256"]
        images.append(dict(id=row["id"], image=str(path), image_sha256=row["image_sha256"], prompt=row["prompt"]))
    runnerfile = HOME / "era_midstep_round/eval/run_inference.py"
    assert scale.sha(runnerfile) == "a59d89cee13548fba4c2bf416a05581e3364129a11c36cdf0c0795a2a6b0c594"
    runner = module("frozen_solver", runnerfile)
    runner.HERE = EV / "journal"
    class RoundEngine(runner.Engine):
        def one(self, row, root):
            return super().one(row, ROOT)
    runner.Engine = RoundEngine
    runner.inputs = lambda dataset: (source,images)
    runner.run("lastblock", arm, checkpoint(arm))
    journal = runner.HERE / "runs" / ("lastblock-" + arm)
    output = journal / "predictions.jsonl"
    manifest = scale.read(journal / "manifest.json")
    expected = scale.BASE_HASHES if arm == "base" else scale.read(ROOT / (arm+"-weights.json"))["weights"]
    assert manifest["weights"] == expected and manifest["inputs_sha256"] == scale.sha(source)
    old = scale.read(R9 / "eval/fresh/runs/r9-base/manifest.json")
    for field in ("config_sha256","engine_sha256","runner_sha256","batch","request_seed","max_output","eval_budget"):
        assert old[field] == manifest[field]
    completed = scale.read(journal / "done.json")
    assert completed["n"] == 192 and completed["output_sha256"] == scale.sha(output)
    for item,p in zip(data,readlines(output),strict=True):
        assert all(p[k] == item[k] for k in ("id","image_sha256","prompt")) and p["seed"] == 3407 and p["max_output"] == 2048
    folder = EV / "runs" / arm
    folder.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(output,folder / "predictions.jsonl")
    scale.freeze(folder / "receipt.json",dict(sha256=scale.sha(output), journal=scale.sha(journal / "done.json")))


def grade():
    sys.path.insert(0,str(R9 / "eval"))
    import grade as judge
    judge.HERE, judge.REVIEW, judge.ARMS = EV, EV / "review", ARMS
    judge.packet()
    judge.grade()
    old = scale.read(R9 / "eval/review/judge_identity.json")
    new = scale.read(judge.REVIEW / "judge_identity.json")
    for field in ("judge_files","rubric_sha256","program","config"):
        assert new[field] == old[field], field


def score():
    sys.path.insert(0,str(R9 / "eval"))
    import score as rubric
    rubric.HERE, rubric.REVIEW, rubric.ARMS = EV, EV / "review", ARMS
    rows, _, _ = rubric.records(final=False)
    rubric.store(EV / "scored.jsonl", rows)
    metrics, transitions = {}, {}
    for arm in ARMS:
        by = {r["id"]:r for r in rows if r["arm"] == arm}
        base = {r["id"]:r for r in rows if r["arm"] == "base"}
        groups = defaultdict(list)
        m = {}
        transitions[arm] = {}
        for ds in ("paired","general"):
            subset = [r for r in by.values() if r["dataset"] == ds]
            assert len(subset) == 96
            before = {r["id"]:base[r["id"]]["answer"] for r in subset}
            after = {r["id"]:r["answer"] for r in subset}
            tr = rubric.transitions(before,after)
            m[ds] = dict(correct=sum(after.values()), base_correct=sum(before.values()),
                         kept=len(tr["kept"]),lost=len(tr["lost"]),recovered=len(tr["recovered"]),
                         connected=rubric.bounds(r["joint"] for r in subset),
                         bare=sum(r["bare"] for r in subset))
            transitions[arm][ds] = tr
            if ds == "paired":
                for r in subset:
                    groups[r["group"]].append(r)
        assert len(groups) == 48 and all(len(g) == 2 for g in groups.values())
        m["paired"]["both_connected"] = rubric.bounds(rubric.every(r["joint"] for r in g) for g in groups.values())
        m["paired"]["both_correct"] = sum(all(r["answer"] for r in g) for g in groups.values())
        metrics[arm] = m
    def gate(candidate_index,control_index):
        new,old = metrics["last50"],metrics["all50"]
        differences = [new["paired"]["both_connected"][candidate_index]-old["paired"]["both_connected"][control_index],
                       new["general"]["connected"][candidate_index]-old["general"]["connected"][control_index]]
        return (min(differences) >= 0 and max(differences) > 0 and
                all(new[ds]["kept"] >= old[ds]["kept"] for ds in ("paired","general")))
    decision = "pass" if gate(0,1) else "fail" if not gate(1,0) else "review"
    result = dict(metrics=metrics,decision=decision,source=scale.sha(EV/"scored.jsonl"),
                  rubric_identity=scale.sha(EV/"review/judge_identity.json"))
    scale.freeze(ROOT / "evaluation.json",result)
    scale.freeze(ROOT / "transitions.json",transitions)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    {"export":export, "infer":infer, "grade":grade, "score":score}[action](*args)
