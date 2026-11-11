"""Grade only saved control responses with the original MMMU extraction protocol."""
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
OLD = Path.home() / "era_single_call_campaign"
sys.path.insert(0,str(OLD))
import mmmu_eval as official
from analysis_round import strict_era
from mmmu_outcomes import details, paired

ORIGINAL = ROOT / "mmmu/control/control"
REC = ROOT / "control_recovery"
SCORE = REC / "score"
FINGERPRINT = "f9df56ed0b3db53999c44f66f975054ba052978c8958efc62e442c39fe3dff4e"
EXCLUDED = "validation_Mechanical_Engineering_11"
SEALED = {
    "manifest.json": "57062abffafa395766575e279061a77c82866ba5df4d4191729eaa1f3113f173",
    "predictions.jsonl": "6645f2faf34be449a2c2b55d723d4697a4acdc212a5d7f45b728d5db230631e7",
    "judge_attempts.jsonl": "d03f871b4676e63b72bee40d9e88b6d0fc6ee913bda09ee289c36733e475fd49",
    "screen/complete.json": "8eca8bfab443e383cf9def1bed08143aded604835d3fc27f95babcd0531365c3",
}
CANDIDATE_SEAL = "1cda6e119da706869ef6a22024af44228ca65f6d367afb59e7d77dfd3f0be8ca"


def committed(data):
    return data[:data.rfind(b"\n")+1]


def replace_bytes(path, data):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def lines(blob):
    return [json.loads(s) for s in blob.splitlines() if s.strip()]


def check_seal(directory, digest):
    seal = directory/"complete.json"
    assert official.sha(seal)==digest, seal
    proof = official.read(seal)
    assert all(official.sha(directory/name)==sha for name,sha in proof["outputs"].items()), directory
    return proof


def inputs(archive, original_only, require_complete):
    for name, sha in SEALED.items():
        assert official.sha(ORIGINAL/name)==sha, name
    manifest = official.read(ORIGINAL/"manifest.json")
    assert manifest["fingerprint"]==FINGERPRINT
    assert manifest["source_sha256"]==official.sha(OLD/"mmmu_eval.py")
    assert manifest["model_identity"]["path"]==str(ROOT/"models/control/merged")
    merge=official.read(ROOT/"models/control/merge.json")
    hashes=manifest["model_identity"]["hashes"]
    assert {k:v for k,v in hashes.items() if k.endswith(".safetensors")}==merge["weights"]
    assert official.read(ORIGINAL/"solver_inflight.json")=={"id":EXCLUDED}
    screenproof = check_seal(ORIGINAL/"screen",SEALED["screen/complete.json"])
    assert screenproof["model"]==FINGERPRINT
    oldblob=(ORIGINAL/"predictions.jsonl").read_bytes()
    assert oldblob==committed(oldblob)
    originals=lines(oldblob)
    roster=official.read(ROOT/"mmmu/control/roster.json")
    original_ids=[r["id"] for r in originals]
    order=roster["screen_ids"]+[rid for rid in roster["full_ids"] if rid not in roster["screen_ids"]]
    assert len(original_ids)==719 and original_ids==order[:719] and order[719]==EXCLUDED
    assert hashlib.sha256(b"".join(oldblob.splitlines(keepends=True)[:90])).hexdigest()==screenproof["predictions_sha256"]
    extblob=b""
    if not original_only and (REC/"plan.json").exists():
        plan=official.read(REC/"plan.json")
        assert plan["original_ids"]==original_ids and plan["virgin_ids"]==order[720:]
        assert plan["fingerprint"]==FINGERPRINT and plan["interrupted_id"]==EXCLUDED
        assert plan["old_ledger_sha256"]==SEALED["predictions.jsonl"]
        supp=REC/"solver"
        if (supp/"predictions.jsonl").exists():
            assert official.read(supp/"manifest.json")==manifest
            extblob=committed((supp/"predictions.jsonl").read_bytes())
        new_ids=[r["id"] for r in lines(extblob)]
        assert new_ids==plan["virgin_ids"][:len(new_ids)] and len(new_ids)<=180
        done=REC/"solver_done.json"
        if done.exists():
            proof=official.read(done)
            assert proof["fingerprint"]==FINGERPRINT and proof["n"]==len(new_ids)==180
            assert proof["plan_sha256"]==official.sha(REC/"plan.json")
            assert proof["predictions_sha256"]==hashlib.sha256(extblob).hexdigest()
        elif require_complete:
            raise RuntimeError("Supplement has not sealed its 180 solver submissions")
    elif require_complete and not original_only:
        raise RuntimeError("Control recovery plan is missing")
    blob=oldblob+extblob
    records=lines(blob)
    assert len(records)==len({r["id"] for r in records})<=899
    for row in records:
        official.match_inputs(row,archive[row["id"]])
    return records,blob,manifest


def copied_attempts():
    original=(ORIGINAL/"judge_attempts.jsonl").read_bytes()
    assert original==committed(original)
    path=SCORE/"judge_attempts.jsonl"
    if not path.exists():
        replace_bytes(path,original)
    raw=path.read_bytes()
    assert raw.startswith(original)
    trimmed=committed(raw)
    if trimmed!=raw:
        # Preserve a torn tail as provenance; only complete rows may go back into the judge.
        witness=SCORE/("judge_torn_"+hashlib.sha256(raw).hexdigest()+".bin")
        if not witness.exists():
            replace_bytes(witness,raw[len(trimmed):])
        replace_bytes(path,trimmed)
    return lines(trimmed)


def audit_matches(records, archive):
    ids=[r["id"] for r in records]
    byid={r["id"]:r for r in records}
    screen={r["id"]:r for r in official.read(ORIGINAL/"screen/results.json")["records"]}
    assert len(screen)==90 and all(byid[rid]==r for rid,r in screen.items())
    candidate_dir=ROOT/"mmmu/candidate/candidate/full"
    check_seal(candidate_dir,CANDIDATE_SEAL)
    candidate={r["id"]:r for r in official.read(candidate_dir/"results.json")["records"]}
    assert sum(r["grading"]["final"]["hit"] for r in candidate.values())==511
    same=[(byid[rid],candidate[rid]) for rid in ids]
    compared=[(archive[rid],byid[rid]) for rid in ids]
    missing=sorted(set(archive)-set(ids))
    unresolved=[r["id"] for r in records if r["grading"]["final"]["hit"] is None]
    left=paired(compared)
    matched=paired(same)
    return dict(n=len(records),official_full900=False,permanent_missing_id=EXCLUDED,
                uncommitted_ids=missing,pending_judge_ids=unresolved,baseline_to_control=left,
                control_to_candidate=matched,details_vs_baseline=details(compared),
                full900_correct_range=[left["right"],left["right"]+len(missing)+len(unresolved)],
                strict_control=sum(map(strict_era,records)),strict_candidate_on_same_ids=sum(strict_era(y) for _,y in same),
                control_length=sum(r["model_response"]["finish_reason"]=="length" for r in records),
                grading_routes=dict(Counter(r["grading"]["final"].get("method","pending") for r in records)),
                screen_control_to_candidate=paired([(x,y) for x,y in same if x["id"] in screen]),
                outside_screen_control_to_candidate=paired([(x,y) for x,y in same if x["id"] not in screen]))


def main():
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--judge",action="store_true",help="Resolve pending extractions with the verified official judge.")
    group.add_argument("--partial",action="store_true",help="CPU only; report bounds from every committed response.")
    parser.add_argument("--original-only",action="store_true",help="Stage just the original 719; may precede the supplement.")
    args=parser.parse_args()
    paths=SimpleNamespace(project=Path.home()/"era_decoder_lora_20261030",
                          baseline=Path.home()/"mmmu_baseline_20261009",output_root=ROOT/"mmmu/control")
    frozen,config,_,archive=official.plan(paths)
    SCORE.mkdir(parents=True,exist_ok=True)
    with frozen.run_lock(SCORE/".score.lock"):
        rows,blob,manifest=inputs(archive,args.original_only,args.judge)
        current=SCORE/"predictions.jsonl"
        if current.exists():
            assert blob.startswith(current.read_bytes()),"Recovery scoring input must grow append-only"
        replace_bytes(current,blob)
        official.freeze(SCORE/"manifest.json",manifest)
        frozen.check_records(rows,config,partial=True)
        pending=official.graded(rows,copied_attempts(),config)
        if pending and args.judge:
            import judge
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
            os.environ.pop("VLLM_BATCH_INVARIANT",None)
            official.verify_judge(paths,config,manifest,SCORE)
            official.save(SCORE/"pending_judge.json",pending)
            judge.run(config,SCORE,manifest)
            pending=official.graded(rows,copied_attempts(),config)
            assert not pending,"Unresolved extractions after official judge"
        summary=audit_matches(rows,archive)
        assert summary["pending_judge_ids"]==pending
        official.save(SCORE/"summary.json",summary)
        official.save(SCORE/"results.json",dict(records=rows,summary=summary,manifest=manifest))
        original_seals=dict(SEALED)
        outputs={name:official.sha(SCORE/name) for name in ("predictions.jsonl","judge_attempts.jsonl","summary.json","results.json")}
        official.save(SCORE/"coverage_seal.json",dict(n=len(rows),pending=len(pending),official_full900=False,
                      original_immutable=original_seals,model=FINGERPRINT,outputs=outputs))
        print(json.dumps({name:summary[name] for name in ("n","full900_correct_range","control_to_candidate","permanent_missing_id")}),flush=True)


if __name__=="__main__":
    main()
