"""Build the local, explicitly unapproved ABCD cross-review handoff."""
import ast
import copy
import csv
import hashlib
import html
import importlib.util
import itertools
import json
import math
import re
import shutil
import statistics
import string
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/era_abcd_prereview_20261021"
WORK = ROOT / "tmp/era_prereview_20261021"
AUDIT = ROOT / "reports/era_abcd_review_20261020"
VERSION = "era-abcd-prereview-v1-20261021"
sys.stdout.reconfigure(encoding="utf-8")
sys.path[:0] = [str(ROOT/".runtime/tokenizers"), str(ROOT/"scripts/visual_groups")]
from tokenizers import Tokenizer
from visual_groups import fingerprint, distance, is_candidate, assign, validated_decisions, PROFILE

def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def rows(path): return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(obj): return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def save(path,obj):
    path=OUT/path
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def jsonl(path,data):
    path=OUT/path
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in data),encoding="utf-8")
def writecsv(path,data,headers):
    path=OUT/path
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=headers);w.writeheader();w.writerows(data)
def norm(s): return " ".join(re.findall(r"\w+|[%+−-]", str(s).lower()))

OUT.mkdir(exist_ok=True)
# Regeneration is allowed only for this generator's own unapproved output.
marker=OUT/"BUILD_ID"
if any(OUT.iterdir()) and (not marker.exists() or marker.read_text()!=VERSION):
    raise RuntimeError("Refusing to overwrite an unrelated output directory")
marker.write_text(VERSION)
spec=importlib.util.spec_from_file_location("selector",ROOT/"reports/era_selection_kit_20261015/select_questions.py")
selector=importlib.util.module_from_spec(spec);spec.loader.exec_module(selector)
audit=read(AUDIT/"audit_full.json")
original={o:rows(audit["archive_identities"][o]["manifest"]) for o in "ABCD"}
original_byid={r["sample_id"]:r for batch in original.values() for r in batch}
fp_byid={r["sample_id"]:r for r in read(AUDIT/"image_fingerprints_full.json")}
edits=read(ROOT/"scripts/era_prereview_content_20261021.json")
observations=read(WORK/"ab_visual_observations.json")
assert len(observations)==100
excluded={"C_1003":"이미지가 필요하지 않은 clock 소리 문항.",
          "C_10413":"원본 food web 먹이관계의 과학적 타당성 불확실; 정답을 바꾸지 않고 대체."}
replacements=read(WORK/"replacement_candidates.json")
assert {r["source_id"] for r in replacements}=={"10598","10611"}
science=read(ROOT/"reports/era_dataset_examples_20261014/scienceqa_problems.json")
tokenizer=Tokenizer.from_file(str(ROOT/".runtime/qwen_tokenizer/tokenizer.json"))
tree=ast.parse((ROOT/"code/evaluation/mmdl-visual-token-analysis/src/evaluate.py").read_text(encoding="utf-8"))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="prompt_for")
namespace=dict(ast=ast,math=math,string=string)
exec(compile(ast.Module(body=[fn],type_ignores=[]),"baseline_prompt","exec"),namespace)
config=read(ROOT/"results/era_D_50_20261016/provenance/baseline_config.json")
def prompt(row):
    return namespace["prompt_for"](dict(question=row["question"],hint=row["hint"] or None,
                                        options=row["options"],question_type="multiple-choice"),config)

generation=dict(editor="Codex (GPT-6 family; exact serving model ID not exposed)",
    model_id=None,model_revision=None,generation_settings=None,human_editor=False,
    method="Same assistant jointly rewrites both responses from image and shared units; deterministic serialization.",
    prompt_version="era-prereview-paired-edit-v1-20261021",
    code_version=VERSION,date="2026-10-21")
paired=[];sources=[];history_reports={};logs={};changes=[]
for owner in "ABCD":
    source,source_fp=selector.load_sources(ROOT/"reports/era_dataset_examples_20261014",owner)
    source_index={r["source_id"]:(n,r) for n,_,batch in selector.grouped(source) for r in batch}
    history=selector.read_log(WORK/"C_history.csv" if owner=="C" else audit["selections"][owner]["path"])
    for rec in history:
        if owner=="C" and rec["원본문항ID"] in {"10598","10611"}:
            rec.update({"상태":"작성완료","사용모델":generation["editor"],"제작방법":"이미지와 고정 원본으로 공통 내용·일반·ERA 동시 작성; 독립 사람 검수 대기",
                        "메모":"2026-10-21 제외 문항 보충 초안. 최종 승인 아님."})
    selector.validate_log(history,owner,source,source_fp)
    rerun,selected,skipped=selector.select(owner,source,source_fp,50,history)
    assert len(selected)==50
    writecsv(f"selection/{owner}/image_log.csv",history,selector.HEADERS)
    jsonl(f"selection/{owner}/selected_questions.jsonl",selected)
    active={r["source_id"] for r in selected}
    assigned=[n for n,_,batch in selector.grouped(source) if selector.assigned_owner(batch[0]["dataset"],n)==owner and n<=max(r["image_group_no"] for r in selected)]
    visited={int(r["이미지그룹번호"]) for r in history}
    assert set(assigned)<=visited
    history_reports[owner]=dict(selected=50,history_rows=len(history),excluded=sum(r["상태"]=="제외" for r in history),
        assignment_sequence_covered=True,source_fingerprint=source_fp,selection_version=selector.VERSION,
        replay_selected_ids=sorted(active),binary_skipped=skipped)
    save(f"selection/{owner}/run_info.json",history_reports[owner]);logs[owner]=history
    batch=[copy.deepcopy(r) for r in original[owner] if r["sample_id"] not in excluded]
    if owner=="C":
        for candidate in replacements:
            sid=candidate["source_id"];src=science[sid]
            batch.append(dict(sample_id="C_"+sid,owner="C",source_dataset="ScienceQA",source_split="train",source_id=sid,
                source_revision="2cbf8318e07b9ece895bb2ae605e71e38d623264",
                image_path=candidate["image_path"],image_sha256=sha(WORK/f"{sid}.png"),
                image_group_no=candidate["image_group_no"],selection_version=candidate["selection_version"],
                source_fingerprint=candidate["source_fingerprint"],question=src["question"],hint=src["hint"],
                options=src["choices"],answer_letter=string.ascii_uppercase[src["answer"]],answer_value=src["choices"][src["answer"]],
                generation_method=generation["method"],model_id=None,model_revision=None,generation_settings=None,
                prompt_version=generation["prompt_version"],code_version=VERSION,human_edits=False))
    assert {r["source_id"] for r in batch}==active
    for row in sorted(batch,key=lambda r:r["image_group_no"]):
        sid=row["sample_id"];n,src=source_index[row["source_id"]]
        before=copy.deepcopy(row)
        row["reviewer"]={"A":"B","B":"A","C":"D","D":"C"}[owner]
        if isinstance(row["options"],dict):
            assert list(row["options"])==list(string.ascii_uppercase[:len(row["options"])])
            row["options"]=list(row["options"].values())
        # Original official fields remain unchanged and explicitly typed.
        row.update(original_question=src["question"],original_answer=src["original_answer"],
                   original_options=src["original_options"],hint=src["hint"],options_generated=owner in "AB")
        if owner in "CD":
            row["original_answer_value"]=src["original_options"][src["original_answer"]]
            row.setdefault("source_url","https://github.com/lupantech/ScienceQA/blob/"+row["source_revision"]+"/data/scienceqa/problems.json")
            row.setdefault("source_license","CC-BY-NC-SA-4.0 (official LICENSE-DATA; see provenance/ScienceQA_LICENSE-DATA.txt)")
            row.setdefault("distractor_rationales",None)
            row.setdefault("human_edits",None)
            row.setdefault("skill_tags",[science[row["source_id"]][k] for k in ["subject","topic","skill"]])
            row.setdefault("rule_reference",dict(type="locked_official_lecture_and_solution",reference="source_records.jsonl",source_id=row["source_id"]))
            row.setdefault("calculation_verification_note","기존 작성 자료 유지. 수치·논리의 독립 사람 검산 대기.")
            row.setdefault("source_disagreement_note",None)
        actual=Path(fp_byid[sid]["image_path"]) if sid in fp_byid else WORK/f'{row["source_id"]}.png'
        assert sha(actual)==row["image_sha256"]
        dest=OUT/"images"/owner/actual.name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(actual,dest)
        row["original_image_path"]=src["image_path"]
        row["submitted_image_path"]=before["image_path"]
        row["image_path"]=dest.relative_to(OUT).as_posix()
        row["input_prompt"]=prompt(row)
        if sid in edits:
            edit=edits[sid];units=edit["units"]
            row["response_era"]="\n".join(category.title()+": "+" ".join(u[1] for u in units if u[0]==category)
                                          for category in ["evidence","rule","application"])
            row["response_general"]=" ".join(units[i][2] for i in edit["order"])
            for field in ["response_general","response_era"]:row[field]+="\nFinal answer: "+row["answer_letter"]
            row["common_content"]={cat:" ".join(u[1] for u in units if u[0]==cat) for cat in ["evidence","rule","application"]}
            row["common_content"].update(conclusion=dict(letter=row["answer_letter"],value=row["answer_value"]),
                verification=edit["reason"]+" 실제 이미지 대조; AI 자체 수정 검토이며 사람 검수 아님.")
            row["semantic_units"]=[dict(id=f"U{i+1}",category=u[0],meaning=u[1],general_excerpt=u[2],era_excerpt=u[1]) for i,u in enumerate(units)]
            row["semantic_units"].append(dict(id="F",category="conclusion",meaning="동일 최종 답 문자",
                general_excerpt="Final answer: "+row["answer_letter"],era_excerpt="Final answer: "+row["answer_letter"]))
            row["revision_provenance"]=dict(generation,reason=edit["reason"],both_conditions_revised_together=True,
                original_generation_preserved=sid in original_byid,source="provenance/content_revisions.json")
            row["evidence_verification_note"]="2026-10-21 Codex가 현재 원본 이미지 직접 열람. "+edit["reason"]
            row["pair_semantic_status"]="AI_revised_shared_units_human_review_pending"
            if sid=="C_10611":
                row["calculation_verification_note"]="관측된 네 칸 MM/Mm/MM/Mm에서 mm=0, 전체=4, 확률 0/4. 코드 검산; 원본 solution 공란 보존."
            row.pop("length_check_line",None)
        else:
            row["pair_semantic_status"]="submitted_pair_human_review_pending"
            row.setdefault("evidence_verification_note","기존 작성자의 이미지·의미단위 자료를 이관. 이번 단계의 독립 사람 전수 검수는 미실시.")
            # Convert C's raw shared notes into the common structure without changing response text.
            if owner=="C":
                match=re.fullmatch(r"Evidence: (.*?)\nRule: (.*?)\nApplication: (.*?)\nFinal answer: ([A-Z])",row["response_era"],re.S)
                assert match,sid
                row["common_content"]=dict(zip(["evidence","rule","application"],match.groups()[:3]),
                    conclusion=dict(letter=row["answer_letter"],value=row["answer_value"]),
                    verification="제출 ERA 단계에서 구조화. 기존 part1_raw와 semantic_units 원문은 provenance/submissions/C.jsonl에 보존.")
        # Do not turn author self-checks or AI edits into independent human approval.
        row["submitted_review_checks"]=copy.deepcopy(before.get("review_checks"))
        row.pop("length_check_line",None)
        row["review_checks"]=dict(source_fields_checked=True,input_pair_checked=True,format_checked=True,
            independent_human_review="not_started",final_acceptance=False)
        row["quality_status"]="draft";row["reviewer_status"]="not_started"
        row["review_note"]="200문항 취합·통합 준비본. A↔B·C↔D 독립 사람 교차 검수 전."
        row["integration_version"]=VERSION
        row["overlap_status"]=dict(internal_image_candidates="AI_resolved_in_registry",
            mmmu_validation="hash_and_exact_question_screen_complete",separate_dev200="pending_manifest",
            separate_eval300="pending_manifest",pretraining_contamination="not_assessed")
        row["source_reference"]="source_records.jsonl"
        sources.append(dict(sample_id=sid,**src,source_revision=row["source_revision"],
                            image_group_no=n,source_fingerprint=source_fp))
        paired.append(row)
        changes.append(dict(sample_id=sid,event="replacement_authored" if sid not in original_byid else "integration_revision",
            date="2026-10-21",actor=generation["editor"],human_approval=False,
            reason=edits[sid]["reason"] if sid in edits else "스키마·경로·빈 hint 정책·검수 상태 통합; 응답 본문 유지.",
            changed_fields=[k for k in row if k not in before or row[k]!=before[k]],
            removed_fields=[k for k in before if k not in row],
            before_record_sha256=digest(original_byid[sid]) if sid in original_byid else None))

# Preserve exact original records and audit evidence, separate from active learning input.
for owner in "ABCD":jsonl(f"provenance/submissions/{owner}.jsonl",original[owner])
save("provenance/archive_identities.json",audit["archive_identities"])
save("provenance/content_revisions.json",edits)
save("provenance/ab_visual_observations.json",observations)
save("provenance/replacement_asset_provenance.json",read(WORK/"replacement_asset_provenance.json"))
save("provenance/source_lock.json",read(ROOT/"reports/era_selection_kit_20261015/source_lock.json"))
save("provenance/baseline_config.json",config)
save("provenance/previous_issues.json",read(AUDIT/"issues_full.json"))
for src,name in [
    (ROOT/"reports/era_pilot_team_rules_v1_20261014.md","team_rules_v1.3.md"),
    (ROOT/"results/era_D_50_20261016/provenance/ScienceQA_LICENSE-DATA.txt","ScienceQA_LICENSE-DATA.txt"),
    (ROOT/"reports/era_selection_kit_20261015/select_questions.py","select_questions.py")]:
    shutil.copy2(src,OUT/"provenance"/name)
for sid,reason in excluded.items():
    changes.append(dict(sample_id=sid,event="excluded_and_replaced",date="2026-10-21",reason=reason,
                        actor=generation["editor"],human_approval=False,before_record_sha256=digest(original_byid[sid])))
jsonl("excluded_items.jsonl",[dict(original_byid[sid],exclusion_reason=reason,quality_status="reject",
                                  exclusion_decision_type="AI_preparation_decision_not_human_review") for sid,reason in excluded.items()])
jsonl("source_records.jsonl",sources)

# Bind every visual judgement to actual image bytes. Reuse only matching old evidence.
images=[dict(sample_id=r["sample_id"],owner=r["owner"],source_id=r["source_id"],image_group_no=r["image_group_no"],
             image_path=str((OUT/r["image_path"]).resolve()),**fingerprint(OUT/r["image_path"])) for r in paired]
image_byid={r["sample_id"]:r for r in images}
old_registry=read(ROOT/"results/CD_visual_groups_v1_20261017/visual_group_registry.json")
active_ids=set(image_byid)
decisions=[d for d in old_registry["decisions"] if {d["left"],d["right"]}<=active_ids]
def decide(left,right,reason):
    return dict(left=left,right=right,left_sha256=image_byid[left]["file_sha256"],
        right_sha256=image_byid[right]["file_sha256"],verdict="different",
        reviewer="Codex AI visual inspection 2026-10-21; not human approval",reason=reason)
ab_candidates=[p for p in read(AUDIT/"image_candidates_full.json") if
               image_byid.get(p["left"],{}).get("owner") in {"A","B"} or image_byid.get(p["right"],{}).get("owner") in {"A","B"}]
assert len(ab_candidates)==346
for pair in ab_candidates:
    left,right=pair["left"],pair["right"]
    assert left in observations and right in observations
    decisions.append(decide(left,right,"직접 열람한 차트의 제목·대상·기간·데이터가 다름. Left: "+observations[left]+" Right: "+observations[right]))
decisions.extend([
    decide("C_10598","D_ScienceQA_10076","원형 자석 서식은 같지만 C는 S–S 대향, D는 S–N 대향. 극 표기와 하단 자석 방향이 달라 문제의 결정적 근거가 다름."),
    decide("C_10611","C_10329","C_10611은 M/M 행과 M/m 열: MM,Mm,MM,Mm. C_10329는 H/h 양축: HH,Hh,Hh,hh. 교배 조건·칸별 유전자형이 다름.")
])
decisions=validated_decisions(images,decisions)
groups=assign(images,decisions,old_registry)
previous_ids={r["sample_id"]:r["visual_group_id"] for r in old_registry["images"]}
assert all(r["visual_group_id"]==previous_ids[r["sample_id"]] for r in images if r["sample_id"] in previous_ids)
decision_map={tuple(sorted((d["left"],d["right"]))):d for d in decisions}
candidates=[]
for a,b in itertools.combinations(images,2):
    if a["pixel_sha256"]==b["pixel_sha256"] or is_candidate(a,b):
        decision=decision_map.get(tuple(sorted((a["sample_id"],b["sample_id"]))))
        status="same" if a["visual_group_id"]==b["visual_group_id"] else decision["verdict"] if decision else "pending"
        candidates.append(dict(left=a["sample_id"],right=b["sample_id"],status=status,**distance(a,b)))
assert all(p["status"]!="pending" for p in candidates)
registry_id="VGR_"+digest(dict(members=[(r["sample_id"],r["file_sha256"],r["visual_group_id"]) for r in images],decisions=decisions))[:24]
for row in paired:
    row["visual_group_id"]=image_byid[row["sample_id"]]["visual_group_id"]
    row["visual_group_registry_id"]=registry_id
    row["visual_group_registry"]="visual_group_registry.json"
    row["image_pixel_sha256"]=image_byid[row["sample_id"]]["pixel_sha256"]
multigroups=[g for g in groups if len(g["member_ids"])>1]
for row in images:row["image_path"]=Path(row["image_path"]).relative_to(OUT).as_posix()
registry=dict(schema_version="visual-groups-v1",registry_id=registry_id,hash_profile=PROFILE,
    source_manifest_sha256={o:sha(audit["archive_identities"][o]["manifest"]) for o in "ABCD"},
    groups=groups,images=images,decisions=decisions,candidates=candidates,unresolved_pairs=0,
    scope="Current 200 draft images; AI identity review, not final human content approval.")
save("visual_group_registry.json",registry)
save("provenance/visual_registry_migration.json",dict(
    previous_registry_id=old_registry["registry_id"],previous_file_sha256=sha(ROOT/"results/CD_visual_groups_v1_20261017/visual_group_registry.json"),
    removed_ids=sorted(excluded),added_ids=sorted(active_ids-set(previous_ids)),
    retained_ids=sorted(active_ids&set(previous_ids)),retained_group_ids_unchanged=True,
    reason="명시적 후보 교체·AB 추가. 기존 active 이미지 계열 ID 보존. 이전 등록부를 덮어쓰지 않음."))
save("repeated_visual_families.json",dict(
    policy="검수 전 초안에서는 9개 계열을 숨기거나 서로 다른 계열로 취급하지 않고 같은 visual_group_id로 묶어 보존한다.",
    final_retention="pending_human_cross_review",groups=multigroups,
    restriction="모두 train 후보이며 계열을 train/dev/eval 사이에 나누지 않는다. 반복 허용·제외·대체는 최종 고정 전에 명시적으로 결정해야 한다."))

# Recheck current 200, including replacements, against pinned MMMU reference evidence.
refs=read(AUDIT/"mmmu_reference_fingerprints.json")
reference_path=ROOT/"results/mmmu_first_baseline_20261009/baseline_seed3407/results.json"
reference=read(reference_path)
old_overlap=read(AUDIT/"mmmu_overlap_full.json")
assert sha(reference_path)==old_overlap["reference_sha256"]
mm_exact=[];mm_near=[]
for a in images:
    for b in refs:
        if a["pixel_sha256"]==b["pixel_sha256"]:mm_exact.append([a["sample_id"],b["id"],b["field"]])
        if is_candidate(a,b):mm_near.append(dict(sample_id=a["sample_id"],reference_id=b["id"],field=b["field"],**distance(a,b)))
mm_text=[[p["sample_id"],r["id"]] for p in paired for r in reference["records"] if norm(p["question"])==norm(r["question"])]
assert not mm_exact and not mm_near and not mm_text
save("mmmu_overlap_report.json",dict(reference_sha256=sha(reference_path),dataset=old_overlap["dataset"],
    questions=len(reference["records"]),images=len(refs),exact_pixel=mm_exact,perceptual_candidates=mm_near,
    exact_question=mm_text,hash_profile=PROFILE,reference_fingerprints_sha256=sha(AUDIT/"mmmu_reference_fingerprints.json"),
    scope="Current 200 drafts; reused pinned reference fingerprints and exact reference questions.",
    separate_dev200="not_run_missing_manifest",separate_eval300="not_run_missing_manifest",
    limitation="정확 질문·파일/픽셀·지각 해시 후보 검사. 모든 의미 중복이나 사전학습 오염 부재를 증명하지 않음."))

# Validate/export only the explicit student-input allowlist.
required="""sample_id owner reviewer source_dataset source_split source_id source_revision source_url source_license image_path image_sha256 image_group_no selection_version source_fingerprint original_question original_answer original_options question hint options options_generated distractor_rationales answer_letter answer_value input_prompt common_content semantic_units response_general response_era generation_method model_id model_revision generation_settings prompt_version code_version human_edits skill_tags evidence_verification_note rule_reference calculation_verification_note source_disagreement_note quality_status review_checks review_note overlap_status""".split()
token_rows=[];checks=[];plain=[];era=[];checklist=[]
def count(s): return len(tokenizer.encode(s,add_special_tokens=False).ids)
def body(s): return re.sub(r"(?m)^(?:Evidence|Rule|Application):\s*","",re.sub(r"\n?Final answer: [A-Z]\s*$","",s)).strip()
for row in paired:
    sid=row["sample_id"];owner=row["owner"]
    missing=[k for k in required if k not in row];assert not missing,(sid,missing)
    src=next(s for s in sources if s["sample_id"]==sid)
    assert row["question"]==row["original_question"]==src["question"]
    assert row["original_answer"]==src["original_answer"] and row["original_options"]==src["original_options"]
    assert row["hint"]==src["hint"] and row["input_prompt"]==prompt(row)
    assert row["source_split"] in {"train","train_human"} and row["quality_status"]=="draft"
    assert row["image_group_no"]%2==(1 if owner in "AC" else 0)
    assert len(row["options"])==len(set(map(norm,row["options"])))
    if owner in "CD":
        assert row["options"]==src["original_options"]
        assert row["answer_letter"]==string.ascii_uppercase[src["original_answer"]]
        assert row["answer_value"]==row["options"][src["original_answer"]]
    else:
        prev=original_byid[sid]
        assert row["answer_value"]==prev["answer_value"] and row["answer_letter"]==prev["answer_letter"]
        assert row["options"]==(list(prev["options"].values()) if isinstance(prev["options"],dict) else prev["options"])
    assert row["response_general"].endswith("Final answer: "+row["answer_letter"])
    assert row["response_era"].endswith("Final answer: "+row["answer_letter"])
    assert re.fullmatch(r"Evidence: .+\nRule: .+\nApplication: .+\nFinal answer: [A-Z]",row["response_era"],re.S)
    assert not re.search(r"(?m)^(Evidence|Rule|Application):",row["response_general"])
    assert norm(body(row["response_general"]))!=norm(body(row["response_era"]))
    if sid in edits:
        for unit in row["semantic_units"]:
            assert unit["general_excerpt"] in row["response_general"]
            assert unit["era_excerpt"] in row["response_era"]
    for field in ["response_general","response_era"]:
        if sid not in edits:assert row[field]==original_byid[sid][field]
    tg=count(row["response_general"]);te=count(row["response_era"])
    bg=count(body(row["response_general"]));be=count(body(row["response_era"]))
    diff=abs(bg-be)/min(bg,be)
    exception="두 조건의 의미 동등성·자연스러움 사람 재확인 대기." if diff>.15 else None
    if sid=="B-chartqa-006":exception="원제출 보고서의 Fiji 값 반복에 따른 길이 차이 예외를 이관. 사람 재확인 대기."
    if sid in edits and diff>.15:exception="수정 공통 의미 단위가 양쪽에 모두 존재하나 자연스러운 문장 표현의 길이 차이. 사람 검토 대기; 자동 합격 처리하지 않음."
    tok=dict(sample_id=sid,owner=owner,general=tg,era=te,general_body=bg,era_body=be,
             body_difference_ratio=round(diff,6),review_over_15pct=diff>.15,response_over_512=max(tg,te)>512,exception_note=exception)
    token_rows.append(tok);row["length_check"]=tok
    user={"role":"user","content":[{"type":"image","path":row["image_path"]},{"type":"text","text":row["input_prompt"]}]}
    for target,field in [(plain,"response_general"),(era,"response_era")]:
        target.append(dict(sample_id=sid,visual_group_id=row["visual_group_id"],visual_group_registry_id=registry_id,
                           messages=[copy.deepcopy(user),{"role":"assistant","content":row[field]}]))
    checks.append(dict(sample_id=sid,source_preserved=True,assignment_pass=True,image_sha256_verified=True,
        input_pair_identical=True,response_format_pass=True,human_review="not_started",
        changed_response=sid in edits,semantic_spans_checked=sid in edits))
    checklist.append(dict(sample_id=sid,owner=owner,reviewer=row["reviewer"],record_sha256=digest(row),
        image_path=row["image_path"],visual_group_id=row["visual_group_id"],status="not_started",
        visual_evidence="",rule_and_application="",unique_answer="",pair_equivalence="",image_necessity="",
        family_retention="",length_review="",verdict="",notes="",reviewer_name="",reviewed_at=""))
assert len(paired)==len(plain)==len(era)==200
assert len({r["sample_id"] for r in paired})==len({(r["source_dataset"],r["source_id"]) for r in paired})==200
assert len({r["original_image_path"] for r in paired})==200
assert len({r["image_sha256"] for r in paired})==len({r["image_pixel_sha256"] for r in paired})==200
assert all(p["messages"][0]==e["messages"][0] for p,e in zip(plain,era))
assert sum(g=="mm" for g in ["MM","Mm","MM","Mm"])==0
jsonl("paired_train.draft.jsonl",paired);jsonl("train_plain.draft.jsonl",plain);jsonl("train_era.draft.jsonl",era)
jsonl("review_log.jsonl",changes);jsonl("cross_review_checklist.jsonl",checklist)
writecsv("cross_review_checklist.csv",checklist,list(checklist[0]))
for reviewer in "ABCD":
    subset=[r for r in checklist if r["reviewer"]==reviewer]
    assert len(subset)==50
    writecsv(f"reviewer_{reviewer}/checklist.csv",subset,list(checklist[0]))
save("item_validation.json",checks)
save("token_report.json",dict(tokenizer="Qwen3-VL-4B-Instruct",
    revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",tokenizer_sha256=sha(ROOT/".runtime/qwen_tokenizer/tokenizer.json"),
    method="add_special_tokens=False; body excludes ERA labels and common Final answer line; not processor/chat-template lengths",
    totals={k:sum(r[k] for r in token_rows) for k in ["general","era","general_body","era_body"]},
    distributions={k:dict(min=min(r[k] for r in token_rows),max=max(r[k] for r in token_rows),mean=statistics.mean(r[k] for r in token_rows),median=statistics.median(r[k] for r in token_rows)) for k in ["general","era","general_body","era_body"]},
    exceptions=[r for r in token_rows if r["review_over_15pct"] or r["response_over_512"]],records=token_rows))
save("validation_report.json",dict(version=VERSION,date="2026-10-21",stage="prepared_for_human_cross_review",
    collection=dict(unique_questions=200,response_pairs=200,responses=400,owners=dict(Counter(r["owner"] for r in paired)),
        datasets=dict(Counter(r["source_dataset"] for r in paired)),all_split="train",all_status="draft"),
    automated_checks=dict(source_preservation=200,assignment=200,image_paths_and_hashes=200,input_pair_equality=200,
        response_format=200,required_metadata_fields=200,unique_source_ids=200,unique_source_image_paths=200,
        exact_file_duplicates=0,exact_pixel_duplicates=0,semantic_spans_checked_for_revised_pairs=len(edits)),
    revisions=dict(existing_pairs_changed=len(edits)-2,replacement_pairs=2,removed_submitted_ids=sorted(excluded),
        added_ids=["C_10598","C_10611"],additional_candidates_excluded=["10592","10605"],
        empty_hint_policy="Keep original hint field; pass None to baseline prompt_for only when empty; no evaluation code change."),
    visual_identity=dict(groups=len(groups),multiple_item_groups=len(multigroups),AB_pairs_resolved=346,
        all_candidate_pairs=len(candidates),unresolved_pairs=0,review_type="AI image identity review, not independent human content approval"),
    lengths=dict(over_15pct=sum(r["review_over_15pct"] for r in token_rows),over_512=sum(r["response_over_512"] for r in token_rows)),
    cross_review=dict(status="not_started",completed=0,assigned=200,reviewer_map={"A":"B","B":"A","C":"D","D":"C"}),
    final_training_release=False,
    pending=["A↔B·C↔D의 독립 사람 200문항 검수 및 수정 후 재검수",
             "반복 이미지 계열 9개의 최종 유지·제외·대체 결정",
             "별도 dev 200 / eval 300 확정 목록과 문항·계열 대조",
             "최종 고정 전 실제 processor/chat-template의 토큰·EOS·loss mask 검증"],
    limitations=["이번 신규 이미지 전체 내용 검수 범위는 수정·보충 문항. 나머지 전체 의미·정답의 사람 검수 완료를 주장하지 않음.",
                 "모델 ID/설정 미제공 값은 null 유지. 이전 생성 및 작성자 자체 검수는 원제출 스냅샷에 보존.",
                 "원격 Hugging Face/노션은 수정하지 않음."]))

# Local review interface: answers are initially collapsed, no judgement is prefilled.
def esc(s):return html.escape(str(s))
parts=['''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ERA 200문항 · 교차 검수 준비본</title><style>
body{font:16px/1.65 system-ui,"Malgun Gothic",sans-serif;max-width:1180px;margin:32px auto;padding:0 22px;background:#f5f7fb;color:#19283c}
h1{font-size:28px}header,article,.notice{background:white;border:1px solid #dae1eb;border-radius:12px;padding:22px;margin:16px 0}
.notice{border-left:5px solid #cc7a10}input,select{font:inherit;padding:8px;margin:4px;max-width:90%}
img{max-width:100%;max-height:650px;object-fit:contain;display:block;margin:16px auto}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}
pre{font:14px/1.6 system-ui;white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;font-weight:600}
.tag{display:inline-block;background:#e9effb;border-radius:7px;padding:3px 9px;font-size:14px}small{color:#536176}
@media(max-width:760px){.cols{grid-template-columns:1fr}body{padding:0 10px}article{padding:16px}}
</style><header><h1>ERA 200문항 · 교차 검수 준비본</h1>
<p>2026-10-21 · A/B/C/D 각 50문항 · 일반/ERA 각 200응답 · 전부 draft</p>
<p><b>교차 검수는 아직 시작하지 않았습니다.</b> 먼저 이미지와 문제로 답을 확인한 뒤 아래 풀이를 펼쳐 대조하세요.</p>
<p><a href="README.md">작업 설명</a> · <a href="cross_review_checklist.csv">200문항 검수표</a> ·
<a href="validation_report.json">통합 검증 결과</a> · <a href="token_report.json">길이 결과</a></p>
<label>검수 담당 <select id="reviewer"><option value="">전체</option><option>A</option><option>B</option><option>C</option><option>D</option></select></label>
<label>검색 <input id="search" placeholder="문항 ID 또는 질문"></label><span id="count"></span></header>
<div class="notice">확인된 9개 반복 이미지 계열은 동일 계열 ID로 묶었습니다. 현재 초안에 보존되어 있으며 최종 유지 여부는 검수 항목입니다.
별도 개발·평가 목록 대조와 실제 학습 processor 검증은 최종 고정 전에 필요합니다.</div>''']
multi_ids={sid for g in multigroups for sid in g["member_ids"]}
for row in paired:
    sid=row["sample_id"]
    warnings=[]
    if sid in edits:warnings.append(edits[sid]["reason"])
    if sid in multi_ids:warnings.append("동일 이미지 계열의 다른 문항이 있음: 최종 유지 여부 확인.")
    if row["length_check"]["review_over_15pct"]:warnings.append("본문 길이 차이 15% 초과: 의미 동등성·자연스러움 확인.")
    options="\n".join(f"{letter}. {value}" for letter,value in zip(string.ascii_uppercase,row["options"]))
    parts.append(f'<article data-reviewer="{row["reviewer"]}" data-search="{esc(sid+" "+row["question"])}" id="{esc(sid)}">'
        f'<h2>{esc(sid)}</h2><span class="tag">작성 {row["owner"]} → 검수 {row["reviewer"]}</span> <span class="tag">draft · 미검수</span>'
        f'<p>{esc(" / ".join(warnings))}</p><small>원본 그룹 {row["image_group_no"]} · 계열 {row["visual_group_id"]}</small>'
        f'<pre>{esc(row["input_prompt"])}</pre><a href="{esc(row["image_path"])}"><img loading="lazy" src="{esc(row["image_path"])}" alt="{esc(sid)} 원본 이미지"></a>'
        '<details><summary>풀이·공통 내용·수정 내역 열기</summary><div class="cols">'
        f'<section><h3>일반 풀이</h3><pre>{esc(row["response_general"])}</pre></section>'
        f'<section><h3>ERA 풀이</h3><pre>{esc(row["response_era"])}</pre></section></div>'
        f'<h3>공통 내용</h3><pre>{esc(json.dumps(row["common_content"],ensure_ascii=False,indent=2))}</pre>'
        f'<details><summary>의미 단위</summary><pre>{esc(json.dumps(row["semantic_units"],ensure_ascii=False,indent=2))}</pre></details>'
        f'<p>본문 토큰 일반 {row["length_check"]["general_body"]} / ERA {row["length_check"]["era_body"]} '
        f'· 차이 {row["length_check"]["body_difference_ratio"]:.1%}</p></details></article>')
parts.append('''<script>
const cards=[...document.querySelectorAll('article')],who=document.querySelector('#reviewer'),query=document.querySelector('#search');
function filter(){let n=0;for(const c of cards){const show=(!who.value||c.dataset.reviewer===who.value)&&c.dataset.search.toLowerCase().includes(query.value.toLowerCase());c.hidden=!show;if(show)n++;}document.querySelector('#count').textContent=` ${n}문항`;}
who.addEventListener('change',filter);query.addEventListener('input',filter);filter();
</script></html>''')
(OUT/"review.html").write_text("\n".join(parts),encoding="utf-8")
print(json.dumps(read(OUT/"validation_report.json"),ensure_ascii=False,indent=2))
