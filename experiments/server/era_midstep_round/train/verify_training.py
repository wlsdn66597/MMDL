"""Audit both finished R5 arms using only receipts, weights and the original optimizer schedule."""
import json
from pathlib import Path

import train_midstep as train


HERE = Path(__file__).resolve().parent
MODELS = HERE/"models/r5"
read, sha = train.engine.read, train.engine.sha


def main():
    proof = read(train.PREFLIGHT)
    shared = read(MODELS/"matched_init.json")
    ids = [r["id"] for r in proof["samples"]]
    helper = train.engine.trainer_module()
    schedule = helper.sample_schedule(len(ids),2,3407)
    expected_ids = [ids[i] for epoch in schedule for i in epoch]
    seq = train.canonical_json_hash(expected_ids)
    assert shared["sequence"] == seq and shared["base"] == train.engine.HASHES
    assert proof["epochs"] == 2 and sha(Path(train.engine.__file__)) == train.ENGINE_SHA
    summaries, identities = {}, []
    for arm in train.ARM_FIELD:
        folder = MODELS/arm
        manifest = read(folder/"manifest.json")
        ident = dict(manifest["identity"])
        assert ident.pop("arm") == arm == manifest["arm"]
        identities.append(ident)
        assert ident["sequence"] == seq and ident["seed"] == 3407 and ident["accumulation"] == 16
        assert ident["receipt"] == sha(train.PREFLIGHT) and ident["parent"] == train.engine.HASHES
        assert ident["hyperparameters"]["lr"] == 1e-4 and ident["hyperparameters"]["epochs"] == 2
        assert manifest["actual_steps"] == ident["steps"] == proof["updates"]
        assert manifest["seen"] == len(expected_ids) and manifest["complete"] and manifest["frozen_verified"]
        assert manifest["initial_adapter"] == shared["initial_adapter"]
        assert manifest["final_adapter"] != shared["initial_adapter"]
        assert manifest["objective_tokens"]["upweighted"] == 0
        tokens = manifest["objective_tokens"]["supervised"]
        assert tokens == 2*proof["token_stats"]["all"][arm]["total"]
        grad = read(folder/"gradient_audit.json")
        assert grad["all_trainables_have_gradient"] and grad["frozen_parameters_have_no_gradient"]
        assert grad["all_gradients_finite"] and grad["nonzero_gradient_names"]
        adapter = sha(folder/"adapter/adapter_model.safetensors")
        merged = read(folder/"merge.json")
        assert adapter == manifest["adapter_file_sha256"] == merged["adapter"]
        assert merged["parent"] == shared["base"]
        assert merged["weights"] == {p.name:sha(p) for p in (folder/"merged").glob("*.safetensors")}
        summaries[arm] = dict(steps=manifest["actual_steps"],presentations=manifest["seen"],
                supervised_tokens=tokens,loop_seconds=manifest["training_seconds"],
                trainable_parameters=manifest["trainable"],initial_nonzero_gradients=len(grad["nonzero_gradient_names"]),
                initial_adapter=manifest["initial_adapter"],final_adapter=manifest["final_adapter"],
                adapter_sha256=adapter,merged_weights=merged["weights"])
    assert identities[0] == identities[1]
    assert summaries["standard"]["merged_weights"] != summaries["preserved"]["merged_weights"]
    result = dict(verified=True,base_revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
                  preflight_sha256=sha(train.PREFLIGHT),selected_file_sha256=proof["selected_file_sha256"],
                  exposure_order_sha256=seq,n_per_epoch=len(ids),common=200,new=len(ids)-200,
                  new_targets_differ=proof["new_targets_differ"],new_strata={
                      k:v["standard"]["n"] for k,v in proof["token_stats"].items() if k not in ("shared200","all")},
                  target_token_stats=proof["token_stats"], arms=summaries)
    train.freeze(HERE/"training_result.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
