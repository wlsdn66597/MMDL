"""Read-only progress accounting and official parser replay, independent of the solver."""
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "mmmu"
RUN = OUT / "last50"
sys.path.insert(0,str(ROOT.parent / "era_half_strength_mmmu"))
import report_results as audit

KST = ZoneInfo("Asia/Seoul")


def inspect(baseline, roster, manifest, parser):
    pred_file = RUN / "predictions.jsonl"
    data = pred_file.read_bytes()
    raw = data.splitlines(keepends=True)
    if raw and not raw[-1].endswith(b"\n"):
        raw.pop()
    rows = [json.loads(line) for line in raw]
    early = set(roster["screen_ids"])
    order = roster["screen_ids"] + [k for k in roster["full_ids"] if k not in early]
    assert [r["id"] for r in rows] == order[:len(rows)]
    preds = {r["id"]:r for r in rows}
    for k,r in preds.items():
        assert all(r[f]==baseline[k][f] for f in audit.INPUT_FIELDS)
        assert r["model_response"]["input_tokens"]==baseline[k]["model_response"]["input_tokens"]
    done = (RUN / "full/complete.json").is_file()
    grades = audit.replay(RUN,preds,manifest,parser,allow_pending=not done)
    ids = [k for k in roster["full_ids"] if k in grades]
    counts, transitions = audit.counts(ids,baseline,preds,grades)
    now = datetime.now(KST)
    start = datetime.fromtimestamp((RUN / "manifest.json").stat().st_mtime,KST)
    previous = sum(baseline[k]["model_response"]["elapsed_seconds"] for k in order if k not in preds)
    observed = sum(r["model_response"]["elapsed_seconds"] for r in rows)
    reference = sum(baseline[k]["model_response"]["elapsed_seconds"] for k in preds)
    eta = now + timedelta(seconds=previous * (observed/reference if len(rows)>=90 and reference>0 else 1) + 120)
    state = audit.load(ROOT / "status.json")
    partial = dict(asof=now.isoformat(timespec="seconds"),stage=state["stage"] if "stage" in state else state["state"],
                   generated=len(rows),resolved=len(ids),pending_judge=len(rows)-len(ids),
                   correct=counts["correct"],baseline_on_resolved=counts["baseline_correct"],
                   transitions={k:len(v) for k,v in transitions.items()},eta_kst=eta.isoformat(timespec="minutes"),
                   eta_method="archived per-item times; calibrate on matched items after first 90 outputs")
    audit.store(ROOT / "official_progress.json",partial)
    # The experiment began with training, before the MMMU checkpoint was loaded.
    training_start = datetime.fromtimestamp((ROOT/"inventory.json").stat().st_mtime,KST)
    if (now-training_start).total_seconds()>=12*3600 and not (ROOT/"interim_12h.json").exists():
        audit.store(ROOT/"interim_12h.json",partial)
    if done:
        assert len(rows)==len(grades)==900 and not (RUN/"solver_inflight.json").exists()
        assert audit.load(OUT/"nominee.json") == {"name":"last50","checkpoint":manifest["fingerprint"]}
        _,_,screen = audit.verify_scope(RUN,"screen",roster["screen_ids"],baseline,manifest,preds,grades,audit.sha(b"".join(raw[:90])))
        summary,transitions,_ = audit.verify_scope(RUN,"full",roster["full_ids"],baseline,manifest,preds,grades,audit.sha(data),screen)
        tally = {k:len(v) for k,v in transitions.items()}
        assert summary["correct"]==575-tally["lost"]+tally["gained"]
        audit.store(ROOT/"official_transitions.json",transitions)
        audit.store(ROOT/"official_verification.json",dict(verified=True,summary=summary,maintained=summary["correct"]>=575,
             transitions=tally,finished=now.isoformat(timespec="seconds"),model=manifest["fingerprint"],
             checkpoint_receipt=audit.digest(ROOT/"last50-weights.json"),journal_sha256=audit.sha(data)))
        text = (f"# 마지막 블록 ERA 50%의 공식 MMMU 결과\n\n"
                f"900문항 중 {summary['correct']}개 정답. LoRA 전 기준은 575개이고 차이는 {summary['correct']-575:+d}개다. "
                f"기존 정답 575개에서 {tally['kept']}개를 유지하고 {tally['lost']}개를 잃었다. "
                f"기존 오답 325개에서 {tally['gained']}개를 회복했다.\n\n"
                f"같은 질문·이미지·프롬프트와 정답 추출 규약으로 고유 ID 900개를 단일 호출했다. "
                f"첫 90개 응답과 채점을 재사용했다. 토큰 상한으로 종료한 {summary['length']}개는 오답 처리했다.\n\n"
                "[검증 결과](official_verification.json) · [전환 ID](official_transitions.json)\n")
        (ROOT/"official_report.md").write_text(text,encoding="utf8")
    return done or state["state"] == "failed"


def main():
    baseline, roster = audit.originals(OUT)
    manifest = audit.load(RUN/"manifest.json")
    proof = audit.load(ROOT/"last50-weights.json")
    hashes = manifest["model_identity"]["hashes"]
    assert {k:v for k,v in hashes.items() if k.endswith(".safetensors")} == proof["weights"]
    assert proof["factor"]==0.5 and proof["bf16_export_checked_targets"]==4
    assert manifest["configuration"]==audit.load(audit.BASE/"protocol/config.seed3407.json")
    assert manifest["solver_calls_per_item"]==1 and manifest["source_sha256"]==audit.ENGINE_HASH
    parser = audit.parser_module()
    while True:
        if inspect(baseline,roster,manifest,parser):
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
