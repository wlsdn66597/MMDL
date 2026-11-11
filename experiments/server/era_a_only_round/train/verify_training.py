"""Verify R9's paired optimizer schedules, target mask counts and exported weights."""
from collections import Counter
import json

import train_a as train


MODEL_DIR=train.HERE/"models/r9"
read,sha=train.engine.read,train.engine.sha


def report_text(receipt, results):
    arms=results["arms"]
    labels=["| 항목 | 일반 ERA | A 보강 |","|---|---:|---:|"]
    for name, key in [("2 epoch 문항 노출","presentations"),("업데이트","updates"),("학습 대상 파라미터","trainable"),
                      ("두 epoch 감독 토큰","supervised_tokens"),("학습 루프 시간(초)","loop_seconds")]:
        values=[arms[a][key] for a in train.ARMS]
        cells=[f"{v:,.2f}" if isinstance(v,float) else f"{v:,}" for v in values]
        labels.append(f"| {name} | {' | '.join(cells)} |")
    stats=receipt["token_stats"]
    group_rows=["| 문항 유형 | 문항 수 | 일반 ERA 토큰 | 보강형 토큰 | 차이 |","|---|---:|---:|---:|---:|"]
    for group in ("all","modified24","A","SYMBOL","GENERAL","shared200","r5_direct"):
        plain,strong=(stats[group][a] for a in train.ARMS)
        group_rows.append(f"| {group} | {plain['n']} | {plain['total']:,} | {strong['total']:,} | {strong['total']-plain['total']:+,} |")
    return "\n".join([
        "# 일반 ERA와 A 보강형의 학습 조건","",
        "두 모델은 LoRA 전 Qwen3-VL-4B-Instruct의 같은 가중치와 같은 LoRA 초기값에서 출발했다. "
        "200개 공통 문항, 앞선 라운드의 직접 판독 48문항, 신규 이미지 24장의 48문항을 같은 순서로 학습했다. "
        "새 데이터 중 24문항에서만 A에 차이가 있었다. 질문, 보기, 이미지, E, R과 마지막 답은 두 군에서 동일하다.","",
        *labels,"","각 epoch의 감독 토큰은 Qwen 토크나이저의 assistant 본문과 마지막 종료 토큰을 합산했다.","",
        *group_rows,"","시드 3407, 학습률 0.0001, micro batch 1, 누적 배치 16, BF16과 decoder attention LoRA를 사용했다. "
        "각 이미지 쌍의 두 문항은 두 epoch 동안 각각 두 번, 총 네 번 노출됐다. 양쪽 학습 기록에서 초기 본체·어댑터, "
        "샘플 순서, 38회 업데이트, 동결층 무변경, 유한한 기울기와 병합 가중치 해시를 검증했다. "
        "[사전 검수](data/receipt.json) · [학습 및 병합 영수증](training_result.json)",""])


def main():
    receipt=read(train.PREFLIGHT)
    assert receipt["verified"] and receipt["n"]==296 and receipt["updates"]==38 and len(receipt["changed_ids"])==24
    assert receipt["identical_input_prefixes"] and receipt["identical_evidence_rules_and_finals"]
    slots=receipt["samples"]
    schedule=train.engine.trainer_module().sample_schedule(296,2,3407)
    ids=[slots[i]["id"] for epoch in schedule for i in epoch]
    groups=Counter(slots[i]["group"] for epoch in schedule for i in epoch if slots[i]["kind"] in train.GROUPS)
    assert len(groups)==24 and set(groups.values())=={4}
    assert len(ids)==592 and len(set(ids))==296 and set(Counter(ids).values())=={2}
    sequence=train.previous.canonical_json_hash(ids)
    init=read(MODEL_DIR/"matched_init.json")
    assert init["sequence"]==sequence and init["base"]==train.engine.HASHES
    arms={}
    identities=[]
    for arm in train.ARMS:
        folder=MODEL_DIR/arm
        manifest=read(folder/"manifest.json")
        identity=dict(manifest["identity"])
        assert identity.pop("arm")==arm==manifest["arm"]
        identities.append(identity)
        assert identity["parent"]==init["base"] and identity["receipt"]==sha(train.PREFLIGHT)
        assert identity["sequence"]==sequence and identity["accumulation"]==16 and identity["seed"]==3407
        assert identity["hyperparameters"]["epochs"]==2 and identity["hyperparameters"]["lr"]==1e-4
        assert manifest["initial_adapter"]==init["initial_adapter"] and manifest["frozen_verified"] and manifest["complete"]
        assert manifest["actual_steps"]==38 and manifest["seen"]==592 and manifest["final_adapter"]!=init["initial_adapter"]
        assert manifest["objective_tokens"]["upweighted"]==0
        assert manifest["objective_tokens"]["supervised"]==2*receipt["token_stats"]["all"][arm]["total"]
        grad=read(folder/"gradient_audit.json")
        assert grad["all_trainables_have_gradient"] and grad["frozen_parameters_have_no_gradient"]
        assert grad["all_gradients_finite"] and grad["nonzero_gradient_names"]
        merged=read(folder/"merge.json")
        assert merged["parent"]==init["base"] and merged["adapter"]==sha(folder/"adapter/adapter_model.safetensors")==manifest["adapter_file_sha256"]
        assert merged["weights"]=={p.name:sha(p) for p in (folder/"merged").glob("*.safetensors")}
        arms[arm]=dict(updates=38,presentations=592,trainable=manifest["trainable"],initial_adapter=init["initial_adapter"],
                       final_adapter=manifest["final_adapter"],supervised_tokens=manifest["objective_tokens"]["supervised"],
                       loop_seconds=manifest["training_seconds"],weights=merged["weights"])
    assert identities[0]==identities[1] and arms["ordinary"]["weights"]!=arms["enhanced"]["weights"]
    results=dict(verified=True,base_revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
                 n=296,epochs=2,group_exposures=groups,exposure_order_sha256=sequence,
                 preflight_sha256=sha(train.PREFLIGHT),pair_sha256=receipt["pair_sha256"],
                 changed_ids=receipt["changed_ids"],length_stats=receipt["token_stats"],arms=arms)
    train.previous.freeze(train.HERE/"training_result.json",results)
    report=train.HERE/"training_report.md"
    report.write_text(report_text(receipt,results),encoding="utf8")
    print(json.dumps({"arms":arms,"proof":sha(train.HERE/"training_result.json")}),flush=True)


if __name__=="__main__":
    main()
