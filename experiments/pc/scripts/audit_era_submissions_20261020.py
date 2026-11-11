"""Read-only audit of available submissions; never marks drafts accepted."""
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import string
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/era_abcd_review_20261020"
OUT.mkdir(exist_ok=True)
(ROOT / "tmp/era_abcd_review_20261020").mkdir(exist_ok=True)

def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def rows(p):
    return [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

sys.path.insert(0, str(ROOT / ".runtime/tokenizers"))
from tokenizers import Tokenizer
from PIL import Image

tok = Tokenizer.from_file(str(ROOT / ".runtime/qwen_tokenizer/tokenizer.json"))
source = read(ROOT / "reports/era_dataset_examples_20261014/scienceqa_problems.json")
splits = read(ROOT / "reports/era_dataset_examples_20261014/scienceqa_splits.json")
assignment = {
    r["source_id"]: r
    for r in rows(ROOT / "reports/era_image_assignment_20261015/source_manifest.jsonl")
    if r["source_dataset"] == "ScienceQA"
}
tree = ast.parse((ROOT / "code/evaluation/mmdl-visual-token-analysis/src/evaluate.py").read_text(encoding="utf-8"))
fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "prompt_for")
ns = {"ast": ast, "math": math, "string": string}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "baseline_prompt", "exec"), ns)
config = read(ROOT / "results/era_D_50_20261016/provenance/baseline_config.json")

# Remote hashes read from the authenticated HF file UI at the reviewed revision.
packages = {
    "C": (Path("C:/Users/sunj0/Downloads/C_50.zip"),
          ROOT / "results/C50_cross_review_20261017/source_C",
          "99224c9a9039b1fca04709e093ef153288612874a71e3e9aa8048fa16701abb9"),
    "D": (ROOT / "results/MMDL_ERA_Dataset_upload/D_50.zip",
          ROOT / "results/MMDL_ERA_Dataset_D50_package",
          "7ec2ed795b9bd3fe05dba3ca4fcf0659b47bb9dceab899fd7616b23a3c1197ba"),
}
checks, summaries, identities = [], {}, {}
for owner, (archive, folder, remote_hash) in packages.items():
    archive_hash = sha(archive)
    assert archive_hash == remote_hash, f"{owner}: remote archive differs"
    paired_path = folder / "paired_train.draft.jsonl"
    with zipfile.ZipFile(archive) as z:
        candidates = [n for n in z.namelist() if n.endswith("paired_train.draft.jsonl")]
        assert len(candidates) == 1
        assert z.read(candidates[0]) == paired_path.read_bytes()
        archive_image_hashes = {
            hashlib.sha256(z.read(n)).hexdigest()
            for n in z.namelist() if n.lower().endswith((".png", ".jpg", ".jpeg"))
        }
    identities[owner] = {
        "archive": str(archive), "archive_sha256": archive_hash,
        "remote_sha256_matches": True, "paired_bytes_match_archive": True,
        "paired_sha256": sha(paired_path),
    }
    data = rows(paired_path)
    owner_checks = []
    for p in data:
        sid = str(p["source_id"])
        original = source[sid]
        options = list(p["options"].values()) if isinstance(p["options"], dict) else p["options"]
        image_path = folder / p["image_path"]
        fallback = folder / "images" / f"{sid}.png"
        actual = image_path if image_path.exists() else fallback
        with Image.open(actual) as im:
            im.load()
            rgb = im.convert("RGB")
            pixel = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        image_hash = sha(actual)
        canonical = ns["prompt_for"](
            dict(question=p["question"], hint=p["hint"], options=options, question_type="multiple-choice"), config
        )
        tokens = {}
        for kind in ("general", "era"):
            text = p[f"response_{kind}"]
            body = text.rsplit("\nFinal answer:", 1)[0]
            if kind == "era":
                body = re.sub(r"(?m)^(Evidence|Rule|Application):\s*", "", body)
            tokens[kind] = len(tok.encode(text, add_special_tokens=False).ids)
            tokens[f"{kind}_body"] = len(tok.encode(body, add_special_tokens=False).ids)
        body_lengths = [tokens["general_body"], tokens["era_body"]]
        ratio = (max(body_lengths) - min(body_lengths)) / min(body_lengths)
        rec = {
            "sample_id": p["sample_id"], "owner": owner, "source_id": sid,
            "source_exact": p["question"] == original["question"]
                and p["hint"] == original["hint"] and options == original["choices"]
                and p["answer_letter"] == string.ascii_uppercase[original["answer"]]
                and p["answer_value"] == original["choices"][original["answer"]],
            "official_train": sid in splits["train"] and p["source_split"] == "train",
            "assignment_exact": p["image_group_no"] == assignment[sid]["image_group_no"]
                and p["image_group_no"] % 2 == (1 if owner == "C" else 0),
            "declared_image_path_exists": image_path.exists(),
            "image_hash_matches": image_hash == p["image_sha256"],
            "image_bytes_in_archive": image_hash in archive_image_hashes,
            "pixel_sha256": pixel,
            "baseline_prompt_exact": p["input_prompt"] == canonical,
            "only_empty_hint_difference": not p["hint"]
                and p["input_prompt"] == canonical.removeprefix("Hint: \n"),
            "response_format": all(
                p[f"response_{k}"].splitlines()[-1] == f"Final answer: {p['answer_letter']}"
                for k in ("general", "era")
            ) and re.findall(r"(?m)^(Evidence|Rule|Application):", p["response_era"])
                == ["Evidence", "Rule", "Application"]
                and not re.search(r"(?m)^(Evidence|Rule|Application):", p["response_general"]),
            "quality_status": p.get("quality_status"),
            "independent_human_review_recorded": p.get("review_checks", {}).get("independent_human_review", False),
            "tokens": tokens, "body_length_relative_difference": ratio,
            "length_review_flag": ratio > .15,
        }
        checks.append(rec)
        owner_checks.append(rec)
    summaries[owner] = {
        "rows": len(data), "unique_source_ids": len({p["source_id"] for p in data}),
        "status_counts": dict(Counter(p.get("quality_status") for p in data)),
        **{key: sum(r[key] for r in owner_checks) for key in (
            "source_exact", "official_train", "assignment_exact",
            "declared_image_path_exists", "image_hash_matches", "image_bytes_in_archive",
            "baseline_prompt_exact", "only_empty_hint_difference", "response_format",
            "independent_human_review_recorded", "length_review_flag")},
        "tokens": {
            k: {"total": sum(r["tokens"][k] for r in owner_checks),
                "median": statistics.median(r["tokens"][k] for r in owner_checks),
                "max": max(r["tokens"][k] for r in owner_checks)}
            for k in ("general", "era")
        },
    }

registry = read(ROOT / "results/CD_visual_groups_v1_20261017/visual_group_registry.json")
assert registry["source_manifest_sha256"]["C"] == identities["C"]["paired_sha256"]
assert registry["source_manifest_sha256"]["D"] == identities["D"]["paired_sha256"]
shared = [g for g in registry["groups"] if len(g["member_ids"]) > 1]
with (ROOT / "results/C50_cross_review_20261017/review_summary.csv").open(encoding="utf-8-sig", newline="") as f:
    prior_review = list(csv.DictReader(f))
content_flags = [r for r in prior_review if r["내용판정"] != "통과"]
save("audit.json", {
    "audit_date": "2026-10-20", "reviewed_hf_revision": "2f56421b95eec952c40b18380b8cc917b73258c1",
    "scope": "C/D current remote ZIP hashes verified; A/B upload presence only; no full-200 approval",
    "rules": "reports/era_pilot_team_rules_v1_20261014.md (v1.3)",
    "identities": identities, "summaries": summaries, "checks": checks,
    "prior_AI_content_flags_still_applicable_to_identical_C_archive": content_flags,
    "existing_visual_registry_source_manifests_match_current_submissions": True,
    "CD_visual_family_count": len(registry["groups"]),
    "CD_multi_record_visual_families": shared,
    "limitations": [
        "A/B archives not available: no per-record A/B verification.",
        "Content findings reused from 2026-10-17 AI review, not new human approval.",
        "No new full MMMU overlap run in this audit.",
        "Separate team development 200 and evaluation 300 lists not available.",
        "Shared visual families within all-train data are not themselves train/eval leakage.",
        "Image-only input validity, processor truncation, EOS/loss masks require later training preflight.",
    ],
})
print(json.dumps({"identities": identities, "summaries": summaries,
    "CD_visual_family_count": len(registry["groups"]),
    "shared_visual_family_count": len(shared),
    "C_content_flag_count": len(content_flags)}, ensure_ascii=True, indent=2))
