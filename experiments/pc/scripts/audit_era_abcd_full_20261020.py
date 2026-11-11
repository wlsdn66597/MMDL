"""Audit all four user-supplied archives without changing their contents."""
import ast
import csv
import hashlib
import html
import itertools
import json
import math
import re
import statistics
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tmp/era_abcd_review_20261020/extracted"
OUT = ROOT / "reports/era_abcd_review_20261020"
OUT.mkdir(exist_ok=True)
ARCHIVES = Path(r"C:\김선재\대학교\3학년\2학기\멀티모달딥러닝\프로젝트\데이터셋")
sys.path[:0] = [str(ROOT / ".runtime/tokenizers"), str(ROOT / "scripts/visual_groups")]
from tokenizers import Tokenizer
from visual_groups import fingerprint, distance, PROFILE, panels

def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding="utf-8-sig").splitlines() if s.strip()]

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def norm(text):
    return " ".join(re.findall(r"\w+|[%+−-]", str(text).lower()))

def numeric(text):
    text = str(text).strip().replace(",", "")
    match = re.fullmatch(r"\s*([$£€]?)([-+]?\d+(?:\.\d+)?)\s*(%|[A-Za-z]+)?\s*", text)
    if not match:
        return None
    return float(match[2]), (match[3] or "").lower()

def equivalent(a, b):
    if norm(a) == norm(b):
        return True
    # ChartQA serializes pairs as "[Palestine, Cyprus]", while choices use "and".
    def pair(text):
        text = str(text).strip()
        if text.startswith("[") and text.endswith("]"):
            return tuple(norm(v.strip(" '\"")) for v in text[1:-1].split(","))
        if " and " in text:
            return tuple(norm(v) for v in text.split(" and "))
        return None
    pa, pb = pair(a), pair(b)
    if pa and pb:
        return pa == pb
    # A's choices explicitly give both equivalent numeric forms.
    for compound, scalar in ((str(a), b), (str(b), a)):
        m = re.fullmatch(r"([0-9.]+)\s*\(([0-9.]+)%\)", compound)
        if m and math.isclose(float(m[1])*100,float(m[2]),rel_tol=1e-8):
            return equivalent(m[1],scalar)
    na, nb = numeric(a), numeric(b)
    if na is None or nb is None:
        return False
    va, ua = na
    vb, ub = nb
    # Original ChartQA answers may omit units; flag every conversion for review.
    return math.isclose(va, vb, rel_tol=1e-8, abs_tol=1e-8) or (
        (ua == "%" or ub == "%")
        and math.isclose(va / (100 if ua == "%" else 1), vb / (100 if ub == "%" else 1),
                         rel_tol=1e-8, abs_tol=1e-8)
    )

folders = {"A": BASE/"A/A_chartqa_50_draft_20261016", "B": BASE/"B/B_chartqa_50_draft_20261017",
           "C": BASE/"C", "D": BASE/"D"}
paths = {o: folders[o]/(f"draft/{o}50_paired_draft.jsonl" if o in "AB" else "paired_train.draft.jsonl")
         for o in "ABCD"}
data = {o: rows(p) for o, p in paths.items()}
source_dir = ROOT / "reports/era_dataset_examples_20261014"
chart_path = folders["A"]/"common_selector/data/chartqa_train_human.json"
locks = load(ROOT/"reports/era_selection_kit_20261015/source_lock.json")
for name, path in [("chartqa_train_human.json", chart_path),
                   ("scienceqa_problems.json", source_dir/"scienceqa_problems.json"),
                   ("scienceqa_splits.json", source_dir/"scienceqa_splits.json")]:
    assert sha(path) == locks[name]["sha256"], name
chart = load(chart_path)
science = load(source_dir/"scienceqa_problems.json")
splits = load(source_dir/"scienceqa_splits.json")
assignment = rows(ROOT/"reports/era_image_assignment_20261015/source_manifest.jsonl")
assign = {(r["source_dataset"], r["sample_id"] if r["source_dataset"]=="ChartQA" else r["source_id"]): r
          for r in assignment}
tokpath = ROOT/".runtime/qwen_tokenizer/tokenizer.json"
tok = Tokenizer.from_file(str(tokpath))
tree = ast.parse((ROOT/"code/evaluation/mmdl-visual-token-analysis/src/evaluate.py").read_text(encoding="utf-8"))
fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name=="prompt_for")
ns = dict(ast=ast, math=math, string=string)
exec(compile(ast.Module(body=[fn], type_ignores=[]), "baseline_prompt", "exec"), ns)
config = load(ROOT/"results/era_D_50_20261016/provenance/baseline_config.json")

records, images, summaries, selections, exports, contracts = [], [], {}, {}, {}, {}
for owner in "ABCD":
    folder = folders[owner]
    for p in data[owner]:
        sid = p["sample_id"]
        options = list(p["options"].values()) if isinstance(p["options"], dict) else p["options"]
        a = assign[(p["source_dataset"], p["source_id"])]
        src = chart[a["source_index"]] if owner in "AB" else science[p["source_id"]]
        original_question = src["query"] if owner in "AB" else src["question"]
        original_answer = src["label"] if owner in "AB" else src["choices"][src["answer"]]
        declared = folder / p["image_path"]
        actual = declared if declared.is_file() else folder/"images"/f'{p["source_id"]}.png'
        fp = fingerprint(actual)
        images.append(dict(sample_id=sid, owner=owner, image_path=str(actual), **fp))
        hint = p.get("hint", "")
        args = dict(question=p["question"], hint=hint, options=options, question_type="multiple-choice")
        canonical = ns["prompt_for"](args, config)
        nohint = ns["prompt_for"](dict(args, hint=None), config)
        idx = ord(p["answer_letter"]) - ord("A")
        source_checks = {
            "question": p["question"] == original_question,
            "original_question": p.get("original_question", original_question) == original_question,
            "original_answer": p.get("original_answer", src.get("answer")) == (src["label"] if owner in "AB" else src["answer"]),
            "source_answer_equivalent": equivalent(p["answer_value"], original_answer),
            "source_options_order": True if owner in "AB" else options == src["choices"],
            "hint": hint == ("" if owner in "AB" else src["hint"]),
            "train": p["source_split"] in ("train", "train_human") and (owner in "AB" or p["source_id"] in splits["train"]),
            "revision": p["source_revision"] == a["source_revision"],
            "fingerprint": p["source_fingerprint"] == a["source_fingerprint"] if "source_fingerprint" in a else True,
        }
        lengths = {}
        for kind in ("general", "era"):
            text = p["response_"+kind]
            body = text.rsplit("\nFinal answer:",1)[0]
            if kind == "era":
                body = re.sub(r"(?m)^(Evidence|Rule|Application):\s*", "", body)
            lengths[kind] = len(tok.encode(text, add_special_tokens=False).ids)
            lengths[kind+"_body"] = len(tok.encode(body, add_special_tokens=False).ids)
        lo, hi = sorted([lengths["general_body"], lengths["era_body"]])
        num = lambda text: sorted(set(re.findall(r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%|/\d+)?", text)))
        rec = {
            "sample_id": sid, "owner": owner, "source_id": p["source_id"],
            "source_checks": source_checks, "all_source_checks": all(source_checks.values()),
            "assignment": p["image_group_no"] == a["image_group_no"] and p["owner"] == a["owner"],
            "image_path_valid": declared.is_file(), "image_hash_valid": fp["file_sha256"] == p["image_sha256"],
            "answer_value_string_exact": 0 <= idx < len(options) and options[idx] == p["answer_value"],
            "answer_maps_to_option": 0 <= idx < len(options) and equivalent(options[idx], p["answer_value"]),
            "options_count": len(options),
            "normalized_options_unique": len({norm(x) for x in options}) == len(options),
            "options_matching_original": [string.ascii_uppercase[i] for i,x in enumerate(options) if equivalent(x, original_answer)],
            "original_answer_string_changed": str(p["answer_value"]) != str(original_answer),
            "baseline_exact_with_stored_hint": p["input_prompt"] == canonical,
            "baseline_exact_with_absent_hint": not hint and p["input_prompt"] == nohint,
            "response_format": all(p["response_"+k].splitlines()[-1] == "Final answer: "+p["answer_letter"]
                and len(re.findall(r"(?m)^Final answer:", p["response_"+k])) == 1 for k in ("general","era"))
                and re.findall(r"(?m)^(Evidence|Rule|Application):",p["response_era"]) == ["Evidence","Rule","Application"]
                and not re.search(r"(?m)^(Evidence|Rule|Application):",p["response_general"]),
            "quality_status": p.get("quality_status"), "tokens": lengths,
            "body_difference": (hi-lo)/lo, "length_review_flag": (hi-lo)/lo > .15,
            "number_sets_equal": num(p["response_general"]) == num(p["response_era"]),
            "numbers_general": num(p["response_general"]), "numbers_era": num(p["response_era"]),
            "plain_equals_unlabelled_era": norm(p["response_general"]) == norm(re.sub(
                r"(?m)^(Evidence|Rule|Application):\s*", "", p["response_era"])),
            "missing_common_fields": [k for k in ("source_url","source_license","original_question","original_answer",
                "original_options","options_generated","distractor_rationales","human_edits","skill_tags",
                "evidence_verification_note","rule_reference","calculation_verification_note",
                "source_disagreement_note","review_checks","review_note","overlap_status") if k not in p],
        }
        records.append(rec)
    cur = [r for r in records if r["owner"] == owner]
    summaries[owner] = {
        "rows": len(cur), "unique_source_ids": len({p["source_id"] for p in data[owner]}),
        "source_pass": sum(r["all_source_checks"] for r in cur),
        **{k:sum(bool(r[k]) for r in cur) for k in ("assignment","image_path_valid","image_hash_valid",
            "answer_maps_to_option","normalized_options_unique","response_format","baseline_exact_with_stored_hint",
            "baseline_exact_with_absent_hint","length_review_flag","plain_equals_unlabelled_era")},
        "answer_distribution": dict(Counter(p["answer_letter"] for p in data[owner])),
        "status": dict(Counter(p.get("quality_status") for p in data[owner])),
        "tokens": {k:dict(total=sum(r["tokens"][k] for r in cur),
            median=statistics.median(r["tokens"][k] for r in cur),
            p90=sorted(r["tokens"][k] for r in cur)[math.ceil(.9*len(cur))-1],
            max=max(r["tokens"][k] for r in cur)) for k in ("general","era","general_body","era_body")},
        "missing_fields": dict(Counter(k for r in cur for k in r["missing_common_fields"])),
        "skill_tags": dict(Counter(t for p in data[owner] for t in p.get("skill_tags", []))),
    }
    logpath = folder / (f"selection/{owner}_v{5 if owner=='A' else 4}/image_log.csv" if owner in "AB" else "image_log.csv")
    log = list(csv.DictReader(logpath.open(encoding="utf-8-sig",newline="")))
    selections[owner] = {"path": str(logpath), "rows": len(log), "columns": list(log[0]),
                         "status_counts": dict(Counter(r.get("상태",r.get("status")) for r in log))}
    active = [r for r in log if r["상태"] == "작성완료"]
    loggroups = {int(r["이미지그룹번호"]) for r in log}
    expected = set(range(1 if owner in "AC" else 2, max(loggroups)+1, 2))
    selections[owner].update(
        active_ids_match={r["원본문항ID"] for r in active} == {p["source_id"] for p in data[owner]},
        group_sequence_covered=expected == loggroups,
        exclusions_have_reason=all(r["메모"].strip() for r in log if r["상태"]=="제외"),
        fingerprint_consistent=all(r["원본지문"]==data[owner][0]["source_fingerprint"] for r in log),
    )
    pfile = folder / (f"draft/{owner}50_plain_draft.jsonl" if owner in "AB" else "train_plain.draft.jsonl")
    efile = folder / (f"draft/{owner}50_era_draft.jsonl" if owner in "AB" else "train_era.draft.jsonl")
    if not pfile.exists() or not efile.exists():
        exports[owner] = {"available": False}
        continue
    pp, ee = rows(pfile), rows(efile)
    exportchecks = []
    for pair, plain, era in zip(data[owner],pp,ee):
        def unpack(r):
            if "messages" in r:
                msgs = r["messages"]
                user = [m for m in msgs if m["role"]=="user"][0]
                assistant = [m for m in msgs if m["role"]=="assistant"][0]["content"]
                if isinstance(assistant,list):
                    assistant = "".join(v.get("text","") for v in assistant)
                content = user["content"]
                return (r.get("id",r.get("sample_id")), content, assistant, [m["role"] for m in msgs])
            return (r["sample_id"],(r["input_prompt"],r["image_path"]),r["response"],None)
        p, e = unpack(plain), unpack(era)
        exportchecks.append({"sample_id":pair["sample_id"], "ids":p[0]==e[0]==pair["sample_id"],
            "input_pair_identical":p[1]==e[1], "responses_match_paired":p[2]==pair["response_general"] and e[2]==pair["response_era"],
            "roles":p[3]})
        def check_input(raw):
            if "messages" not in raw:
                return raw["input_prompt"] == pair["input_prompt"] and raw["image_path"] == pair["image_path"]
            user = [m for m in raw["messages"] if m["role"]=="user"]
            if len(user) != 1:
                return False
            content = user[0]["content"]
            return len(content)==2 and {v["type"] for v in content}=={"image","text"} and all(
                v.get("text")==pair["input_prompt"] if v["type"]=="text"
                else v.get("path",v.get("image"))==pair["image_path"] for v in content)
        exportchecks[-1]["input_matches_paired_allowlist"] = check_input(plain) and check_input(era)
    exports[owner] = dict(available=True, plain_rows=len(pp),era_rows=len(ee),checks=exportchecks)

allpairs = []
for a,b in itertools.combinations(images,2):
    d = distance(a,b)
    if a["pixel_sha256"] == b["pixel_sha256"] or d["dhash_distance"] <= 4 or d["phash_distance"] <= 6:
        allpairs.append(dict(left=a["sample_id"],right=b["sample_id"],exact=a["pixel_sha256"]==b["pixel_sha256"],**d))
save("image_candidates_full.json", allpairs)
save("image_fingerprints_full.json", images)
ab_candidates = [p for p in allpairs if p["left"].startswith(("A","B")) or p["right"].startswith(("A","B"))]
panels(images, ab_candidates, OUT/"image_candidate_panels")

baseline = ROOT/"results/mmmu_first_baseline_20261009"
refpath = baseline/"baseline_seed3407/results.json"
reference = load(refpath)
mm_images, missing = [], []
for p in reference["records"]:
    for field in p["image_fields"]:
        ip = baseline/"analysis/images"/f'{p["id"]}_{field}.png'
        if not ip.exists():
            missing.append(str(ip))
        else:
            mm_images.append(dict(id=p["id"],field=field, image_path=str(ip),**fingerprint(ip)))
mm_exact, mm_near = [], []
for a in images:
    for b in mm_images:
        d = distance(a,b)
        if a["pixel_sha256"] == b["pixel_sha256"]:
            mm_exact.append(dict(sample_id=a["sample_id"], reference_id=b["id"],field=b["field"]))
        elif d["dhash_distance"]<=4 or d["phash_distance"]<=6:
            mm_near.append(dict(sample_id=a["sample_id"],reference_id=b["id"],field=b["field"],**d))
question_exact = [(p["sample_id"],r["id"]) for owner in data for p in data[owner]
                  for r in reference["records"] if norm(p["question"])==norm(r["question"])]
save("mmmu_overlap_full.json", dict(reference_sha256=sha(refpath),dataset=reference["manifest"]["configuration"]["dataset"],
    questions=len(reference["records"]),images=len(mm_images),missing=missing,exact_pixel=mm_exact,
    perceptual_candidates=mm_near, exact_question=question_exact,hash_profile=PROFILE,
    note="Perceptual candidates require visual judgement; zero exact matches is not complete leakage clearance. Separate dev/eval lists unavailable."))
save("mmmu_reference_fingerprints.json", mm_images)

all_records = [p for o in "ABCD" for p in data[o]]
source_duplicates = [v for v in (
    [p["sample_id"] for p in all_records if (p["source_dataset"],p["source_id"])==k]
    for k in set((p["source_dataset"],p["source_id"]) for p in all_records)) if len(v)>1]
report = dict(date="2026-10-20",scope="All 200 paired records; structural, source and local MMMU checks. Semantic review is separate.",
    archive_identities={o:dict(path=str(ARCHIVES/f"{o}_50.zip"),sha256=sha(ARCHIVES/f"{o}_50.zip"),
        manifest=str(paths[o]),manifest_sha256=sha(paths[o])) for o in "ABCD"},
    source_locks=locks, tokenizer_sha256=sha(tokpath), summaries=summaries, records=records, selections=selections,
    exports=exports, source_duplicate_groups=source_duplicates,
    internal_exact_pixel_pairs=[p for p in allpairs if p["exact"]],
    internal_perceptual_candidate_count=len(allpairs),AB_perceptual_candidate_count=len(ab_candidates),
    mmmu=dict(questions=len(reference["records"]),images=len(mm_images),missing=len(missing),
        exact_pixel=len(mm_exact),perceptual_candidates=len(mm_near),exact_question=len(question_exact)),
    limitations=["No human cross-review approval performed.", "Perceptual hash candidates are not duplicate verdicts.",
                 "No separate development/final evaluation lists available.",
                 "Full processor/chat-template truncation and loss-mask checks are not run."])
save("audit_full.json", report)
with (OUT/"item_checks_full.csv").open("w",encoding="utf-8-sig",newline="") as f:
    keys=["sample_id","owner","all_source_checks","assignment","image_path_valid","image_hash_valid",
          "answer_maps_to_option","response_format","body_difference","length_review_flag","number_sets_equal","quality_status"]
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in records)
for owner in "ABD":
    text = []
    for p in data[owner]:
        text.append(f'{p["sample_id"]} | {p["question"]}\nOptions: {p["options"]}\nOriginal answer: {p.get("original_answer")}\n'
                    f'PLAIN: {p["response_general"]}\nERA: {p["response_era"]}\nSource disagreement: {p.get("source_disagreement_note")}\n')
    (OUT/f"{owner}_content_for_review.txt").write_text("\n".join(text),encoding="utf-8")
print(json.dumps({k:report[k] for k in ("summaries","selections","source_duplicate_groups",
    "internal_exact_pixel_pairs","internal_perceptual_candidate_count","AB_perceptual_candidate_count","mmmu")},ensure_ascii=True,indent=2))
