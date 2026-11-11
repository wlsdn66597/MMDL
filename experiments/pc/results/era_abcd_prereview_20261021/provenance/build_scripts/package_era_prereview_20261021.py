"""Seal this draft's files and verify the ZIP without publishing anything."""
import hashlib
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
OUT=ROOT/"results/era_abcd_prereview_20261021"
WORK=ROOT/"tmp/era_prereview_20261021"
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

report=read(OUT/"validation_report.json")
report["review_ui_checks"]=dict(html_structure="passed_200_articles_200_images",
    local_links="passed_no_missing_files",browser_visual_render="NOT_RUN",
    reason="Browser tool URL security policy disallowed file:// navigation; no alternative browser route attempted.")
save(OUT/"validation_report.json",report)
(OUT/"tools").mkdir(exist_ok=True)
shutil.copy2(ROOT/"scripts/verify_era_prereview_package_20261021.py",OUT/"tools/verify_package.py")
shutil.copy2(ROOT/"scripts/visual_groups/visual_groups.py",OUT/"tools/visual_groups.py")
code=OUT/"provenance/build_scripts"
code.mkdir(exist_ok=True)
for name in ["build_era_prereview_20261021.py","prepare_era_prereview_assets_20261021.py",
             "fetch_era_replacement_assets_20261021.py","era_prereview_content_20261021.json",
             "package_era_prereview_20261021.py"]:
    shutil.copy2(ROOT/"scripts"/name,code/name)
exclusions=OUT/"provenance/excluded_images"
exclusions.mkdir(exist_ok=True)
images={r["sample_id"]:r for r in read(ROOT/"reports/era_abcd_review_20261020/image_fingerprints_full.json")}
for sid in ["C_1003","C_10413"]:
    shutil.copy2(images[sid]["image_path"],exclusions/f"{sid}.png")
shutil.copy2(WORK/"10592.png",exclusions/"candidate_10592.png")
science=read(ROOT/"reports/era_dataset_examples_20261014/scienceqa_problems.json")
save(OUT/"provenance/excluded_candidate_records.json",[
    dict(source_id=sid,official_record=science[sid],
         inspection_basis="image_and_source_text" if sid=="10592" else "source_text_suffices_to_establish_no_image_required",
         image_path="excluded_images/candidate_10592.png" if sid=="10592" else None)
    for sid in ["10592","10605"]])
spec=importlib.util.spec_from_file_location("verify_package",OUT/"tools/verify_package.py")
validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
result=validator.check(OUT,ROOT)
result["package_integrity"]="validated_separately_after_manifest_creation"
result["browser_visual_render"]="NOT_RUN_policy_blocked_local_file_navigation"
save(OUT/"independent_verification.json",result)
# Archive reproducible image-observation contact sheets, not the large source parquet.
shutil.copytree(WORK/"ab_boards",OUT/"provenance/ab_visual_boards",dirs_exist_ok=True)
shutil.copytree(WORK/"new_candidate_panels",OUT/"provenance/new_candidate_panels",dirs_exist_ok=True)
files={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="PACKAGE_SHA256.json" and "__pycache__" not in p.parts}
# Do not import local modules again after this manifest is produced.
save(OUT/"PACKAGE_SHA256.json",files)
dest=OUT.with_suffix(".zip")
with zipfile.ZipFile(dest,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in files:z.write(OUT/name,arcname=OUT.name+"/"+name)
    z.write(OUT/"PACKAGE_SHA256.json",arcname=OUT.name+"/PACKAGE_SHA256.json")
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(files)+1
    for name,digest in files.items():
        assert hashlib.sha256(z.read(OUT.name+"/"+name)).hexdigest()==digest,name
summary=dict(package=dest.name,zip_sha256=sha(dest),zip_bytes=dest.stat().st_size,
    package_files=len(files),zip_file_hashes_verified=True,
    content_status="draft_before_human_cross_review",browser_visual_render="NOT_RUN")
save(dest.with_suffix(".zip.sha256.json"),summary)
print(json.dumps(summary,indent=2))
