"""Recompute the pair of R6 exposure and update manifests, including the merged weight seals."""
import json
from collections import Counter
from pathlib import Path

import train_direction as train

MODEL_DIR=train.HERE/"models/r6"
read,sha=train.engine.read,train.engine.sha


def main():
    receipt=read(train.PREFLIGHT)
    assert receipt["verified"] and receipt["n"]==296 and receipt["updates"]==38
    slots=receipt["samples"]
    init=read(MODEL_DIR/"matched_init.json")
    expected_schedule=train.engine.trainer_module().sample_schedule(296,2,3407)
    ids=[slots[i]["id"] for epoch in expected_schedule for i in epoch]
    groups=Counter(slots[i]["group"] for epoch in expected_schedule for i in epoch if slots[i]["role"]!="shared")
    assert len(groups)==24 and set(groups.values())=={4}
    assert len(ids)==592 and set(Counter(ids).values())=={2}
    sequence=train.previous.canonical_json_hash(ids)
    assert init["sequence"]==sequence and init["base"]==train.engine.HASHES
    out={};identities=[]
    for arm in train.ARMS:
        folder=MODEL_DIR/arm
        manifest=read(folder/"manifest.json")
        ident=dict(manifest["identity"])
        assert ident.pop("arm")==arm==manifest["arm"]
        identities.append(ident)
        assert ident["parent"]==init["base"] and ident["receipt"]==sha(train.PREFLIGHT) and ident["sequence"]==sequence
        assert manifest["initial_adapter"]==init["initial_adapter"] and manifest["frozen_verified"] and manifest["complete"]
        assert manifest["final_adapter"]!=init["initial_adapter"] and manifest["actual_steps"]==38 and manifest["seen"]==592
        assert ident["accumulation"]==16 and ident["seed"]==3407 and ident["hyperparameters"]["epochs"]==2
        assert ident["hyperparameters"]["lr"]==1e-4 and manifest["objective_tokens"]["upweighted"]==0
        assert manifest["objective_tokens"]["supervised"]==2*receipt["token_stats"]["all"][arm]["total"]
        grad=read(folder/"gradient_audit.json")
        assert grad["all_trainables_have_gradient"] and grad["frozen_parameters_have_no_gradient"] and grad["all_gradients_finite"]
        assert grad["nonzero_gradient_names"]
        merged=read(folder/"merge.json")
        assert merged["parent"]==init["base"] and sha(folder/"adapter/adapter_model.safetensors")==merged["adapter"]==manifest["adapter_file_sha256"]
        assert merged["weights"]=={p.name:sha(p) for p in (folder/"merged").glob("*.safetensors")}
        out[arm]=dict(updates=38,presentations=592,trainable=manifest["trainable"],initial_adapter=init["initial_adapter"],
                      final_adapter=manifest["final_adapter"],supervised_tokens=manifest["objective_tokens"]["supervised"],
                      loop_seconds=manifest["training_seconds"],weights=merged["weights"])
    assert identities[0]==identities[1]
    assert out["control"]["weights"]!=out["reverse"]["weights"]
    report=dict(verified=True,base_revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",n=296,epochs=2,
                exposure_order_sha256=sequence,preflight_sha256=sha(train.PREFLIGHT),group_exposures=groups,
                pair_sha256=receipt["pair_sha256"],length_stats=receipt["token_stats"],strata=receipt["strata"],arms=out)
    train.previous.freeze(train.HERE/"training_result.json",report)
    print(json.dumps(report,ensure_ascii=False))


if __name__=="__main__":
    main()
