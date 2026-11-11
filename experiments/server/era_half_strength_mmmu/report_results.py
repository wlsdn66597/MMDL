"""Replay official parsing and verify a single-call, no-cue MMMU run without GPU calls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent
BASE = RESULTS / "mmmu_first_baseline_20261009"
PARSER = RESULTS.parent / "code/training/era_decoder_lora/baseline/src/parsers.py"
if not BASE.is_dir() and (RESULTS / "mmmu_baseline_20261009").is_dir():
    BASE = RESULTS / "mmmu_baseline_20261009"
    PARSER = RESULTS / "era_decoder_lora_20261030/baseline/src/parsers.py"
SOURCE = RESULTS / "era_single_call_campaign/mmmu_eval.py"
BASE_HASH = "3791f54f30b9e127fa995f2f442ab0460c988addb069377596ea5830a7a8f0d8"
ROSTER_HASH = "13b00a2003f30e017df49c57f92781b97781f23c5fb578c808f59fc29e3bd612"
RGB_HASH = "eafe5603b1de1af895d55a431a8792f7e53f700b89dcae2ecaf5ad9ff0ac6b74"
ENGINE_HASH = "ce72d13a3ac569fa2e67aafab3d54d5d2665f6960d356ea7212be0dfb54e3c3e"
PARSER_HASH = "5023ed38d2e8af91304b258d5b3798ffc03df24324f49740953d943378a30b04"
SCALE_HASH = "1fdf6aeaca39c491fe51a5dce15e5ca3a4d2a02a9f6161d13bbf6a5f8de9ca8e"
CONFIG_HASH = "086f672d994023879d0255d6979dcc0fc79c916716dbf7e97e55c15e960669de"
JUDGE_INVENTORY = "90bbb65cf9b085eb680c6ad5de665809b7046d27fa982f7776a89d7c57e9cc90"
JUDGE_AUX = "6c3e387355d8223eb958e7aafa174ac6cbb9aeecb95ac639c797aeca51fa2886"
SCREEN_KEY = "era-single-call-screen-v1/"
INPUT_FIELDS = ("id", "subject", "dataset_config", "dataset_index", "question_type", "question",
                "options", "reference_answer", "input_prompt", "rendered_chat", "image_fields", "original_image_sizes_wh")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def store(path, value):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    temporary.replace(path)


def parser_module():
    check(sha(PARSER.read_bytes().replace(b"\r\n", b"\n")) == PARSER_HASH, "Archived parser hash differs")
    spec = importlib.util.spec_from_file_location("replayed_official_parser", PARSER)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.selfcheck()
    return parser


def originals(output):
    baseline_file = BASE / "baseline_seed3407/results.json"
    roster_file = BASE / "protocol/evaluation_items.json"
    pixels_file = BASE / "provenance/input-validation.json"
    check((digest(baseline_file), digest(roster_file), digest(pixels_file)) ==
          (BASE_HASH, ROSTER_HASH, RGB_HASH), "Archived baseline/source inputs changed")
    check(sha(SOURCE.read_bytes().replace(b"\r\n", b"\n")) == ENGINE_HASH, "MMMU runner source changed")
    baseline_rows = load(baseline_file)["records"]
    ids = [r["id"] for r in baseline_rows]
    roster = load(output / "roster.json")
    check(len(ids) == len(set(ids)) == 900 and ids == roster["full_ids"], "Canonical 900-ID roster mismatch")
    check(sum(r["grading"]["final"]["hit"] for r in baseline_rows) == 575, "LoRA-free reference score differs")
    archived_items = load(roster_file)
    check([i["id"] for i in archived_items] == ids, "Source order differs")
    by_subject = defaultdict(list)
    for r, item in zip(baseline_rows, archived_items, strict=True):
        fields = {key:r[key] for key in ("question", "options", "reference_answer", "input_prompt", "image_fields")}
        check(sha(json.dumps(fields, ensure_ascii=False, sort_keys=True).encode()) == item["input_sha256"], r["id"])
        by_subject[r["subject"]].append(r["id"])
    check(len(by_subject) == 30 and all(len(group)==30 for group in by_subject.values()), "Invalid subject composition")
    selected = set()
    for group in by_subject.values():
        selected.update(sorted(group, key=lambda k: sha((SCREEN_KEY + k).encode()))[:3])
    check(roster["screen_ids"] == [k for k in ids if k in selected] and len(selected)==90, "Screen roster differs")
    check((roster["archive_sha256"], roster["roster_sha256"], roster["rgb_sha256"], roster["seed"], roster["screen_key"]) ==
          (BASE_HASH, ROSTER_HASH, RGB_HASH, 3407, SCREEN_KEY), "Roster provenance differs")
    return {r["id"]:r for r in baseline_rows}, roster


def verify_model(run):
    manifest = load(run / "manifest.json")
    receipt_file = RESULTS / "era_half_strength/scale_receipts/ordinary.json"
    check(digest(receipt_file) == SCALE_HASH, "50% scaling receipt differs")
    receipt = load(receipt_file)
    check(receipt["arm"]=="ordinary" and receipt["factor"]==0.5 and receipt["pre_bf16_exact_half_targets"]==144,
          "Wrong ordinary-50% model")
    model_hashes = manifest["model_identity"]["hashes"]
    check({k:v for k,v in model_hashes.items() if k.endswith(".safetensors")} == receipt["weights"],
          "Wrong ordinary-50% model weights")
    check(all(model_hashes.get(k)==v for k,v in receipt["processor"].items()), "Model processor differs")
    check(sha(json.dumps(model_hashes, sort_keys=True).encode()) == manifest["fingerprint"], "Model fingerprint differs")
    config_file = BASE / "protocol/config.seed3407.json"
    check(digest(config_file) == CONFIG_HASH, "Archived seed 3407 configuration changed")
    official_config = load(config_file)
    check(manifest["configuration"] == official_config and manifest["solver_calls_per_item"] == 1 and
          (manifest["source_sha256"], manifest["baseline_sha256"], manifest["pixels_sha256"]) ==
          (ENGINE_HASH, BASE_HASH, RGB_HASH), "Manifest differs from the official baseline protocol")
    return manifest


def replay(run, preds, manifest, parser, *, allow_pending):
    config = manifest["configuration"]
    judge_cfg = config["judge"]
    attempts_path = run / "judge_attempts.jsonl"
    attempts = lines(attempts_path) if attempts_path.exists() else []
    last = {}
    if attempts:
        identity = load(run / "judge_identity.json")
        weights_file, aux_file = (BASE/"provenance/preparation.json", BASE/"provenance/model-auxiliary-hashes.json")
        check((digest(weights_file),digest(aux_file)) == (JUDGE_INVENTORY,JUDGE_AUX), "Archived judge provenance changed")
        weights = load(weights_file)["model_sha256"]["judge"]
        aux = load(aux_file)["judge"]
        check(identity == {"revision":judge_cfg["revision"],"files":{**weights,**aux}}, "Wrong MMMU extraction judge")
    for a in attempts:
        k = a["id"]
        check(k in preds, f"Judge attempt without solver output: {k}")
        r = preds[k]
        choices, _ = parser.choice_info(r)
        options = "There are several options: \n" + "".join(f"{key}. {v}\n" for key,v in choices.items())
        prompt = judge_cfg["prompt_template"].format(r["question"], options, r["model_response"]["raw"].split("</think>")[-1].strip())
        check(a["input_prompt"] == prompt and a["input_sha256"] == sha(prompt.encode()), f"Wrong judge input: {k}")
        prior = last.get(k)
        expected_attempt = 1 if prior is None else prior["attempt"]+1
        check(a["attempt"] == expected_attempt <= judge_cfg["max_attempts"] and
              (prior is None or not prior["decision"]["decision_parsed"]), f"Invalid judge retry: {k}")
        check(a["decision"] == parser.grade_judge(r, a["raw"])["judge_parser_decision"], f"Judge parser mismatch: {k}")
        last[k] = a
    grades = {}
    for k,r in preds.items():
        g = parser.grade_response(r)
        if g["status"] == "pending_judge":
            a = last.get(k)
            if a and (a["decision"]["decision_parsed"] or a["attempt"] == judge_cfg["max_attempts"]):
                g = parser.grade_judge(r, a["raw"])
        else:
            check(k not in last, f"Judge called for parser-resolved item: {k}")
        if g["status"] == "graded":
            check(g["random_fallback"] is False and g["correct"] == bool(g["hit"]), f"Invalid grade: {k}")
            check(r["model_response"]["finish_reason"] != "length" or
                  (g["method"]=="output_limit_incorrect" and not g["hit"]), f"Length override mismatch: {k}")
            grades[k] = g
        else:
            check(allow_pending, f"Ungraded item: {k}")
    return grades


def counts(ids, baseline, preds, grades):
    transitions = {"kept":[], "lost":[], "gained":[], "both_wrong":[]}
    for k in ids:
        before, after = baseline[k]["grading"]["final"]["correct"], grades[k]["correct"]
        label = ("kept" if after else "lost") if before else ("gained" if after else "both_wrong")
        transitions[label].append(k)
    totals = {k:len(v) for k,v in transitions.items()}
    methods = Counter(grades[k]["method"] for k in ids)
    size, correct = len(ids), totals["kept"]+totals["gained"]
    summary = {"n":size, "correct":correct, "baseline_correct":totals["kept"]+totals["lost"],
               "accuracy":correct/size if size else None, "accuracy_percent":100*correct/size if size else None,
               "paired_gained":totals["gained"], "paired_lost":totals["lost"],
               "length":sum(preds[k]["model_response"]["finish_reason"]=="length" for k in ids),
               "judge_count":methods["local_qwen3_8b_judge_parser"], "grading_methods":dict(methods)}
    return summary, transitions


def verify_scope(run, scope, ids, baseline, manifest, preds, grades, pred_hash, screen=None):
    dest = run / scope
    proof = load(dest / "complete.json")
    check(proof["outputs"] == {f:digest(dest/f) for f in ("summary.json","results.json")}, f"{scope} outputs changed")
    check((proof["scope"], proof["n"], proof["model"], proof["predictions_sha256"]) ==
          (scope,len(ids),manifest["fingerprint"],pred_hash), f"{scope} completion receipt differs")
    saved = load(dest/"results.json")
    check(saved["manifest"] == manifest, f"{scope} manifest differs")
    records = saved["records"]
    check([r["id"] for r in records]==ids and len(set(ids))==len(ids), f"{scope} result roster differs")
    check(all(r == {**preds[r["id"]], "grading":{"final":grades[r["id"]]}} for r in records),
          f"{scope} model responses or replayed grades differ")
    summary, transitions = counts(ids, baseline, preds, grades)
    summary["scope"] = scope
    if screen is not None:
        check(all(screen.get(r["id"],r)==r for r in records), "Full run changed an already graded screen row")
        held, _ = counts([k for k in ids if k not in screen], baseline, preds, grades)
        summary["remaining810"] = {key:held[key] for key in ("n","correct","baseline_correct")}
    check(load(dest/"summary.json")==saved["summary"]==summary, f"{scope} summary differs from independently replayed scores")
    return summary, transitions, {r["id"]:r for r in records}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=ROOT/"mmmu")
    ap.add_argument("--name", default="ordinary50")
    ap.add_argument("--scope", choices=("screen", "full"), default="full")
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    baseline, roster = originals(args.output_root)
    run = args.output_root / args.name
    manifest = verify_model(run)
    pred_file = run/"predictions.jsonl"
    data = pred_file.read_bytes() if pred_file.exists() else b""
    check(not data or data.endswith(b"\n"), "Snapshot ends in an incomplete journal row")
    raw_rows = data.splitlines(keepends=True)
    rows = [json.loads(b) for b in raw_rows]
    screen_ids, ids = roster["screen_ids"], roster["full_ids"]
    screen_set = set(screen_ids)
    order = screen_ids + [k for k in ids if k not in screen_set]
    check([r["id"] for r in rows]==order[:len(rows)], "Duplicate, reordered or unknown solver submission")
    preds = {r["id"]:r for r in rows}
    for k,r in preds.items():
        check(all(r[key]==baseline[k][key] for key in INPUT_FIELDS) and
              r["model_response"]["input_tokens"]==baseline[k]["model_response"]["input_tokens"], f"Input drift: {k}")
    grades = replay(run, preds, manifest, parser_module(),
                    allow_pending=args.partial or args.scope=="screen")
    if args.partial:
        done = [k for k in ids if k in grades]
        summary, categories = counts(done, baseline, preds, grades)
        state = {"partial":True, "generated":len(preds), "parser_or_judge_resolved":len(done),
                 "pending_judge_ids":[k for k in preds if k not in grades], "ungenerated":900-len(preds),
                 "subset_counts":summary, "subset_transitions":{k:len(v) for k,v in categories.items()}}
        store(ROOT/"partial_status.json",state)
        print(json.dumps(state, ensure_ascii=False))
        return
    check(len(rows)>=90 and all(k in grades for k in screen_ids), "Screen not fully graded")
    screen_hash = sha(b"".join(raw_rows[:90]))
    screen_summary, _, screen = verify_scope(run,"screen",screen_ids,baseline,manifest,preds,grades,screen_hash)
    if args.scope=="screen":
        print(json.dumps(screen_summary,ensure_ascii=False))
        return
    check(len(rows)==len(grades)==900 and not (run/"solver_inflight.json").exists(), "Full solver run incomplete")
    check(load(args.output_root/"nominee.json")=={"name":args.name,"checkpoint":manifest["fingerprint"]}, "Full nominee differs")
    summary, transitions, _ = verify_scope(run,"full",ids,baseline,manifest,preds,grades,sha(data),screen)
    subject_counts = {}
    for subject in sorted({r["subject"] for r in baseline.values()}):
        subject_ids = [k for k in ids if baseline[k]["subject"]==subject]
        sub, _ = counts(subject_ids, baseline, preds, grades)
        subject_counts[subject] = {key:sub[key] for key in ("n","correct","baseline_correct","paired_lost","paired_gained")}
    verification = {"scope":"full","verified":True,"single_call_solver_ids":900,"screen_reused":90,
                    "baseline_correct":575,"baseline_incorrect":325,"maintained":summary["correct"]>=575,
                    "summary":summary,"transitions":{k:len(v) for k,v in transitions.items()},"by_subject":subject_counts,
                    "journal_sha256":sha(data),"screen_journal_sha256":screen_hash,"manifest_sha256":digest(run/"manifest.json"),
                    "config_seed":3407,"model_fingerprint":manifest["fingerprint"],"ordinary50_receipt_sha256":SCALE_HASH}
    store(ROOT/"verification.json",verification)
    store(ROOT/"transitions.json",transitions)
    render(summary, transitions, subject_counts, screen_summary)
    print(json.dumps({k:verification[k] for k in ("verified","maintained","summary","transitions")}, ensure_ascii=False))


def render(summary, transitions, subjects, screen):
    correct = summary["correct"]
    counts_by_group = {k:len(v) for k,v in transitions.items()}
    state = "충족했다" if correct>=575 else "미달했다"
    text = ["# 일반 ERA 50%의 MMMU 단일 호출 평가", "",
            f"일반 ERA 50%는 **{correct}/900 ({summary['accuracy_percent']:.2f}%)**으로 LoRA 전 기준 575/900에 {state}. "
            f"정답 차이는 {correct-575:+d}개, {(correct-575)/900*100:+.2f}%p다.", "",
            "| LoRA 전 문항 | 50% 정답 | 50% 오답 |", "|---|---:|---:|",
            f"| 정답 575개 | {counts_by_group['kept']} 유지 | {counts_by_group['lost']} 손실 |",
            f"| 오답 325개 | {counts_by_group['gained']} 회복 | {counts_by_group['both_wrong']} 양쪽 오답 |", "",
            f"MMMU validation 30과목의 900개 고유 ID를 기준과 같은 프롬프트로 각각 한 번 생성했다. "
            f"처음 90문항은 {screen['correct']}/90(기준 {screen['baseline_correct']}/90)이었고, "
            f"이 90개의 응답과 판정을 그대로 전체 점수에 포함했다. 시드는 3407, 출력 상한은 32,768토큰이다. "
            f"길이 제한 종료 {summary['length']}개는 오답으로 처리했고, 최종 답 판정에 별도 저지를 사용한 문항은 {summary['judge_count']}개다.", "",
            "이 평가의 입력에는 추가 ERA 지시가 없다. 이전 지시 파일럿의 진입 판정은 기존 기록에 보존한다. "
            "[기준점](../../reports/mmmu_first_baseline_result_20261009.md) · [검증 영수증](verification.json) · [문항별 유지·손실·회복 ID](transitions.json)", "",
            "| 과목 | LoRA 전 정답 | 일반 ERA 50% 정답 | 손실 | 회복 |", "|---|---:|---:|---:|---:|"]
    for name, values in subjects.items():
        text.append(f"| {name} | {values['baseline_correct']} | {values['correct']} | {values['paired_lost']} | {values['paired_gained']} |")
    (ROOT/"report.md").write_text("\n".join(text)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
