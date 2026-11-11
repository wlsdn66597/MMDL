"""Review saved solver outputs without any further model call. Chart and MMMU are separate endpoints."""
import argparse
import hashlib
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NUM = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)"
TERM = re.compile(r"(?im)\bFinal\s+answer\s*:\s*[* ]*([^\r\n]*)")
LABEL = re.compile(r"(?i)\b(Evidence|Rule|Application|Final\s+answer)\s*[* ]*:")


def read(path):
    return Path(path).read_text(encoding="utf8")


def load(path):
    return json.loads(read(path))


def lines(path):
    return [json.loads(x) for x in read(path).splitlines() if x.strip()]


def hashed(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def put(path, value, immutable=False, jsonl=False):
    if jsonl:
        data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in value)
    else:
        data = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf8")==data:
        return
    if immutable and path.exists():
        raise ValueError(f"Immutable file differs: {path}")
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(data,encoding="utf8")
    tmp.replace(path)


def index(items):
    indexed={r["id"]:r for r in items}
    if len(indexed)!=len(items):raise ValueError("Repeated item ID")
    return indexed


def normal_number(s):
    return Decimal(str(s).replace(",",""))


def chart_plan(args):
    rows=load(args.gold)
    natural=[r for r in rows if r["mode"] in ("rank","direct")]
    families=sorted({r["chart"] for r in natural},key=lambda x:hashed(["era-single-call-chart-v1",x]))
    if len(families)<args.families:raise ValueError("Too few chart families")
    chosen=set(families[:args.families])
    selected=[r for r in natural if r["chart"] in chosen]
    assert len(index(selected))==len(selected)==4*len(chosen)
    for r in selected:
        values=sorted(map(normal_number,r["rows"].values()))
        expected=normal_number(r["rows"][r["direct_name"]]) if r["mode"]=="direct" else sum(values[:3])
        assert normal_number(r["truth"])==expected==normal_number(r["options"]["ABCD".index(r["letter"])])
        assert (args.gold.parent/r["image"]).is_file()
        assert hashlib.sha256((args.gold.parent/r["image"]).read_bytes()).hexdigest()==r["image_sha256"]
    inputs=[{k:r[k] for k in ("id","image","image_sha256","question","options")} for r in selected]
    put(args.out/"inputs.json",inputs,immutable=True)
    put(args.out/"gold.json",selected,immutable=True)
    put(args.out/"manifest.json",dict(salt="era-single-call-chart-v1",n=len(selected),families=sorted(chosen),
        gold_sha256=hashed(selected),inputs_sha256=hashed(inputs),source_gold_sha256=hashed(rows),
        scope="source-table chart development diagnostic; same family layout variants are dependent"),immutable=True)


def final_choice(raw,options):
    fields=TERM.findall(raw.strip())
    field=(fields[-1] if fields else raw.strip().splitlines()[-1] if raw.strip() else "").strip(" *.")
    letter=re.match(r"(?i)^(?:Option |Choice )?[\[(]?([A-D])\b",field) if fields else re.fullmatch(r"(?i)(?:Option |Choice )?[\[(]?([A-D])[\])]?",field)
    if not letter and not fields:
        letter=re.search(r"(?i)\b(?:answer is|select|choose|matches)\s+(?:option\s+)?([A-D])\s*$",field)
    if letter:return letter[1].upper()
    if re.fullmatch(NUM,field):
        hits=[i for i,x in enumerate(options) if normal_number(x)==normal_number(field)]
        if len(hits)==1:return "ABCD"[hits[0]]
    return None


def format_ok(raw):
    matches=list(LABEL.finditer(raw))
    labels=[re.sub(r"\s+"," ",m[1].lower()) for m in matches]
    return labels==["evidence","rule","application","final answer"] and all(
        raw[a.end():b.start()].strip(" *\n") for a,b in zip(matches,matches[1:])) and bool(
            re.fullmatch(r"(?i)\s*\**Final\s+answer\s*:\s*\**\S.*",raw.strip().splitlines()[-1]))


def math_trace(eq,expected,truth):
    values=[normal_number(v) for v in eq[:3]]
    return sorted(values)==sorted(expected) and sum(values)==normal_number(eq[3])==truth


def chart_check(it,pred):
    """Detect actual row/operand witnesses in both formats. Only certify two tightly constrained full traces."""
    raw=pred["raw"];letter=final_choice(raw,it["options"]);stopped=pred["finish_reason"]=="stop"
    answer=(letter==it["letter"]) if stopped and letter is not None else False if not stopped else None
    # A heading-free proof and the same proof wrapped in ERA are scanned identically.
    text=re.sub(r"[*`$]","",raw)
    body=LABEL.sub("",TERM.split(text,maxsplit=1)[0] if TERM.search(text) else text)
    body=" ".join(body.split())
    table={k.casefold():normal_number(v) for k,v in it["rows"].items()}
    truth=normal_number(it["truth"])
    parts=[m for m in re.split(r"(?<=[.!?])\s+",body) if m]
    witnessed=False; certified=False; material_error=False
    if it["mode"]=="direct":
        name=re.escape(it["direct_name"])
        target=rf"(?:The\s+)?{name}(?:\s+bar)?\s*(?:[:=]|(?:is|has|shows|reads|denotes|represents)(?:\s+(?:labeled|labelled|value|a value of|measured at))?)\s*({NUM})"
        observed=list(re.finditer(target,body,re.I))
        witnessed=any(normal_number(m[1])==truth for m in observed)
        # An ordinary sentence 'Grove is 53' supplies both the grounded observation and direct lookup.
        if len(parts)==1:
            m=re.fullmatch(target+r"\s*\.?\s*",parts[0],re.I)
            if m:
                certified=normal_number(m[1])==truth
                material_error=not certified
    else:
        expected=list(map(normal_number,it["truth_vals"]))
        entity=rf"([A-Za-z]+)\s*(?:\(\s*|[:=]\s*|(?:at|is)\s+)({NUM})\)?"
        trio=rf"(?:the\s+)?(?:three\s+(?:shortest|smallest|lowest)(?:\s+(?:bars|values|rows))?|selected\s+bars)\s*(?:are|:)\s*{entity},\s*{entity},?\s+(?:and\s+)?{entity}"
        trace=re.search(trio,body,re.I)
        rows_ok=False
        if trace:
            names=[trace[i].casefold() for i in (1,3,5)]
            vals=[normal_number(trace[i]) for i in (2,4,6)]
            rows_ok=len(set(names))==3 and sorted(vals)==sorted(expected) and all(table.get(n)==v for n,v in zip(names,vals))
        equation=rf"(?<![\w.])({NUM})\s*\+\s*({NUM})\s*\+\s*({NUM})\s*=\s*({NUM})(?![\d])"
        arithmetic=any(math_trace(m.groups(),expected,truth) for m in re.finditer(equation,body))
        observed={}
        for name in it["rows"]:
            # Source names and row values can be reported as a list before the rule or in running prose.
            pattern=rf"\b{re.escape(name)}(?:\s+bar)?\s*(?:\(\s*|[:=]\s*|(?:at|is|has|shows|ends at)\s+)({NUM})(?![\w.](?:\d|\w))"
            vals={normal_number(m[1]) for m in re.finditer(pattern,body,re.I)}
            if vals:observed[name.casefold()]=vals
        all_needed=all(observed.get(name)=={value} for name,value in table.items() if value in expected)
        selects_three=bool(re.search(r"(?i)\b(?:three|3)\s+(?:smallest|shortest|lowest)\b",body))
        witnessed=(rows_ok or all_needed) and selects_three and arithmetic
        if trace and len(parts) in (2,3):
            # The entire explanation must fit; crossing out a provisional error or adding contradictory prose goes to review.
            m=re.fullmatch(trio+r"\s*\.",parts[0],re.I)
            rule_sentence=parts[1] if len(parts)==3 else ""
            rule_safe=not rule_sentence or re.fullmatch(r"(?i)(?:Add|Sum) the three (?:smallest|lowest|shortest)(?: chart)? (?:bars|values)\.",rule_sentence)
            eq=re.fullmatch(r"(?:Their sum is |The sum is |Summing gives )?"+equation+r"(?:\s*[,.;]?\s*(?:option|choice)\s+([A-D]))?\.?",parts[-1],re.I)
            if m and rule_safe and eq:
                certified=rows_ok and math_trace(eq.groups()[:4],expected,truth) and (not eq[5] or eq[5].upper()==it["letter"])
                material_error=not certified
    # A literal witness is a triage feature; certification requires checks across the full explanation.
    joint=False if answer is False or material_error else True if answer and certified else None
    return dict(answer=answer,joint=joint,format=format_ok(raw),witness=bool(answer and witnessed),
                chosen=letter,finish=pred["finish_reason"],source="whole text verified" if certified else "source evidence witness" if witnessed else "none")


def strict_era_supplement(it,pred,rec,engine):
    if engine is None or rec["joint"] is not None or rec["answer"] is not True:return
    result=engine.assess(it,pred,"natural")
    if result["joint"] is not None and result["answer"]==rec["answer"]:
        rec["joint"]=result["joint"];rec["source"]="verified ERA semantics"


def checker(args):
    import importlib.util
    path=args.semantics or ROOT.parent/"era_self_1k_v1"/"semantic_check.py"
    if not path.exists():raise FileNotFoundError(f"Required frozen chart checker {path}")
    spec=importlib.util.spec_from_file_location("campaign_chart_semantics",path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod,hashlib.sha256(path.read_bytes()).hexdigest()


def blind_id(rid,raw):
    return hashed(["single-ERA-audit-1",rid,raw])[:24]


def assessments(args):
    if not args.reviews:return {}
    reviewed={}
    for r in lines(args.reviews):
        key=r["blind_id"]
        if key in reviewed:raise ValueError("Duplicate review")
        if not r.get("reviewer") or not r.get("rationale"):raise ValueError("Missing reviewer/rationale")
        for dim in ("E","R","A","link"):
            if not any(r.get(dim) is v for v in (True,False,None)):raise ValueError(f"Invalid component {dim}")
            if dim not in r:raise ValueError(f"Missing component {dim}")
        reviewed[key]=r
    return reviewed


def attach_review(rec,raw,review):
    if not review:return
    vals=[review[k] for k in ("E","R","A","link")]
    quotes=review.get("quotes",[])
    if not isinstance(quotes,list) or any(not isinstance(q,str) or not q or q not in raw for q in quotes):
        raise ValueError("Review quotes must occur verbatim in the response")
    if not quotes or review.get("image_reviewed") is not True or review.get("coverage")!="full_response":
        raise ValueError("Final review requires a verbatim quote, image inspection and full response coverage")
    if rec["answer"] is False or any(x is False for x in vals):rec["joint"]=False
    elif rec["answer"] is True and all(x is True for x in vals):rec["joint"]=True
    else:rec["joint"]=None
    rec["source"]="adjudicated";rec["reviewer"]=review["reviewer"]


def counts(rr,key):
    vals=[x[key] for x in rr]
    return dict(n=len(vals),yes=vals.count(True),no=vals.count(False),unresolved=vals.count(None))


def finish(args,domain,records,queue,reviewed,extra):
    unknown=set(reviewed)-{r["blind_id"] for rr in records.values() for r in rr}
    if unknown:raise ValueError("Stale reviews: "+str(sorted(unknown)))
    summary={arm:{k:counts(rr,k) for k in ("answer","joint","format")} for arm,rr in records.items()}
    n=summary["baseline"]["joint"]["n"];assert n==summary["candidate"]["joint"]["n"] and n>0
    paired=list(zip(records["baseline"],records["candidate"]))
    lower=upper=0
    for b,c in paired:
        assert b["id"]==c["id"]
        if b["blind_id"]==c["blind_id"]:continue
        lower+=int(c["joint"] is True)-int(b["joint"] is not False)
        upper+=int(c["joint"] is not False)-int(b["joint"] is True)
    status="improved" if lower>0 else "not_improved" if upper<=0 else "pending_review"
    metrics=dict(domain=domain,connection_gate=status,denominator=n,baseline=summary["baseline"],candidate=summary["candidate"],
        paired_delta_count_bounds=[lower,upper],paired_delta_percentage_point_bounds=[100*lower/n,100*upper/n],
        meaning="bounds over unresolved review labels, not statistical confidence intervals",
        scoring_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        witness={arm:sum(r.get("witness",False) for r in rr) for arm,rr in records.items()},**extra)
    put(args.out/"connection.json",metrics)
    put(args.out/"scores.jsonl",[{**r,"arm":arm} for arm,rr in records.items() for r in rr],jsonl=True)
    put(args.out/"blind_queue.jsonl",sorted(queue.values(),key=lambda r:r["blind_id"]),jsonl=True)
    print(json.dumps({"domain":domain,"n":n,"connection_gate":status,"paired_bounds":[lower,upper]}))


def chart_score(args):
    gold=index(load(args.spec/"gold.json")); manifest=load(args.spec/"manifest.json")
    assert manifest["gold_sha256"]==hashed(list(gold.values()))
    engine,script_hash=checker(args)
    reviews=assessments(args);allrows={};queue={};source={};prompt_by_id={}
    for arm,path in (("baseline",args.base_preds),("candidate",args.preds)):
        preds=index(lines(path));assert preds.keys()==gold.keys(),(arm,"unexpected coverage")
        source[arm]=hashlib.sha256(path.read_bytes()).hexdigest()
        armrows=[]
        for rid,it in gold.items():
            p=preds[rid];assert p["image_sha256"]==it["image_sha256"]
            assert isinstance(p.get("prompt"),str) and prompt_by_id.setdefault(rid,p["prompt"])==p["prompt"],(rid,"prompt mismatch")
            rec=chart_check(it,p);strict_era_supplement(it,p,rec,engine)
            key=blind_id(rid,p["raw"]);rec.update(id=rid,blind_id=key)
            attach_review(rec,p["raw"],reviews.get(key))
            if rec["joint"] is None or (key not in reviews and int(key[:6],16)%10==0):
                queue[key]=dict(blind_id=key,id=rid,reference=it,raw=p["raw"],instruction="Inspect original image, evidence, applicable rule, substitution and consistency.")
            armrows.append(rec)
        allrows[arm]=armrows
    finish(args,"chart_source_table_diagnostic",allrows,queue,reviews,dict(inputs=source,semantic_checker_sha256=script_hash,
        families=len(manifest["families"]),scope="development charts; overlapping chart layouts and repeated data, no MMMU inference"))


# Pre-inspected directly against the fixed MMMU image index, before new model outputs.
# These five reference cards cover four distinct source-image sets. They guide reviewers,
# never solver inputs, official answer grading, or training. They are not human ground truth.
ECON = dict(
    pixel_hashes=["2e394aac38f23a85e1d0322efab80719642ed5512f22313af25cc4cc490c83e5",
                  "00af3128286214bf795bd3ef8656fe5f5d6ee59f74b929412d5f6de263e44da7"],
    E="At price 12 the original demand is 6000, new demand 2000, and supply 8000.",
    R="Add old and additional demand at the same price. Equilibrium requires total demand equal supply.",
    A="6000 + 2000 = 8000 equals supplied quantity at price 12. Answer the requested price or quantity.")
CARDS = {
    "validation_Economics_14": dict(ECON,gold="C",result="price 12"),
    "validation_Economics_25": dict(ECON,gold="C",result="quantity 8000"),
    "validation_Accounting_13": dict(gold="C",result="72000",
        pixel_hashes=["aeb6dea5546b3f14e70430865c5d1098e9b6b3318a760b620ace61aed32f7602"],
        E="The image gives direct labor 25000, factory depreciation 45000, factory utilities 2000.",
        R="Conversion cost is direct labor plus manufacturing overhead. Exclude direct materials and payroll staff salary.",
        A="25000 + 45000 + 2000 = 72000."),
    "validation_Finance_17": dict(gold="1000",result="1000",
        pixel_hashes=["9b3896409f1aae31e8cf391ce76c52f5a21f6e90232232e2ca9b9228100c8d8f"],
        E="Row B shows par 1000, coupon rate 12%, required annual return 12%, maturity 20 years.",
        R="Matching coupon and required rates using the same period yield a par bond.",
        A="Par shortcut gives 1000. Alternatively, discount 40 half-year 60 coupons and principal at 6% each half-year."),
    "validation_Physics_17": dict(gold="A",result="graph A",
        pixel_hashes=["fb2c873460f7248ce0360afded6a199cfb29f0b4a9417f6b3d3022bdd2a5e8fd"],
        E="A starts at the origin and curves upward; C is a straight line.",
        R="For constant resistance P=IV=I²R.",
        A="At fixed R the curve rises quadratically from the origin; choose A.")
}


def card_witness(rid,raw):
    """Observable named premises plus operation; do not elevate a matching passage to full-response correctness."""
    text=re.sub(r"(?<=\d),(?=\d{3}\b)","",raw.lower()).replace("\\","").replace("$","")
    if rid=="validation_Accounting_13":
        components=all(re.search(word+r"[^.\n]{0,70}?\b"+value+r"\b",text) for word,value in
                       (("direct labor","25000"),("depreciation","45000"),("utilit","2000")))
        terms=re.search(r"conversion costs?[^.\n]{0,100}labor[^.\n]{0,55}overhead",text)
        return bool(components and terms and "72000" in text)
    if rid.startswith("validation_Economics_"):
        eq=re.search(r"\b(?:6000\s*\+\s*2000|2000\s*\+\s*6000)\s*=\s*8000\b",text)
        return bool(eq and "equilibrium" in text and "suppl" in text and re.search(r"\b12\b",text))
    if rid=="validation_Finance_17":
        return bool("coupon" in text and re.search(r"yield|return|discount",text) and re.search(r"12\s*%",text)
                    and re.search(r"par|face value",text) and re.search(r"\b1000\b",text))
    if rid=="validation_Physics_17":
        algebra=re.sub(r"[\s{}_*]","",text)
        return bool(re.search(r"p=ri(?:\^2|²)|p=i(?:\^2|²)r",algebra)
                    and re.search(r"parabol|curv|quadratic",text) and re.search(r"\b(?:graph|option|curve)\s+a\b",text))
    return False


def official_items(path):
    data=load(path)
    if isinstance(data,list):
        rows=data
    else:
        rows=data["records"]
    return index(rows)


def choose_coverage(baseline,ids=None):
    if ids:
        wanted=load(ids)
        if isinstance(wanted,dict):wanted=wanted["screen_ids"]
        if len(set(wanted))!=len(wanted):raise ValueError("Repeated roster ID")
        if not set(wanted)<=baseline.keys():raise ValueError("Baseline lacks frozen IDs")
        return list(wanted)
    subjects=sorted({x["subject"] for x in baseline.values()})
    wanted=set()
    for sub in subjects:
        seq=[x["id"] for x in baseline.values() if x["subject"]==sub]
        if len(seq)<3:raise ValueError("Require at least three items per subject in the screen")
        seq.sort(key=lambda rid:hashlib.sha256(("era-single-call-screen-v1/"+rid).encode()).hexdigest())
        wanted.update(seq[:3])
    return [rid for rid in baseline if rid in wanted]


def official_hit(r):
    g=r.get("grading",{}).get("final",{})
    val=g.get("correct")
    if not isinstance(val,bool) or g.get("status")!="graded":raise ValueError("Require completed official grades with per-item correctness")
    if r["model_response"]["finish_reason"]=="length" and val:raise ValueError("Length truncation cannot be credited")
    return val


def public_item(r):
    return {k:r[k] for k in ("id","subject","question_type","question","options","reference_answer","image_fields")}


def prepare(args):
    base=official_items(args.baseline)
    ids=choose_coverage(base,args.ids)
    roster=[public_item(base[rid]) for rid in ids]
    put(args.out/"review_roster.json",roster,immutable=True)
    info=dict(n=len(roster),subjects=dict(Counter(r["subject"] for r in roster)),
        sampling="SHA256('era-single-call-screen-v1/'+id), 3 per subject unless a frozen --ids file is provided",
        roster_sha256=hashed(roster),reference_cards={rid:c for rid,c in CARDS.items() if rid in ids},
        source="original MMMU image bytes visually inspected for listed reference cards by assistant; no independent human label",
        rubric=dict(E="actual image evidence and relevant question conditions",R="correct rule under these conditions",
            A="applies the same evidence and rule to reach the selected answer",link="no unexplained premise switch or material contradiction"),
        absent_era_titles="Never score heading-free content as automatically incorrect. Missing information is unresolved.",
        official_judge_use="Only the baseline pipeline may call its judge for terminal answer extraction. This tool makes zero model calls.")
    put(args.out/"review_protocol.json",info,immutable=True)
    return ids


def mmmu_score(args):
    ids=prepare(args);base=official_items(args.baseline);cand=official_items(args.candidate)
    if not set(ids)<=cand.keys():raise ValueError("Candidate missing frozen review cases")
    reviews=assessments(args);allrows={};queue={}
    for rid in ids:
        a,b=base[rid],cand[rid]
        assert public_item(a)==public_item(b),(rid,"item mismatch")
        for k in ("input_prompt","rendered_chat","original_image_sizes_wh"):
            if k in a and k in b:assert a[k]==b[k],(rid,k)
        if rid in CARDS:assert str(a["reference_answer"])==CARDS[rid]["gold"],(rid,"wrong card reference")
    for arm,data in (("baseline",base),("candidate",cand)):
        armrows=[]
        for rid in ids:
            r=data[rid];raw=r["model_response"]["raw"];hit=official_hit(r);key=blind_id(rid,raw)
            rec=dict(id=rid,blind_id=key,answer=hit,joint=False if not hit else None,format=format_ok(raw),
                     witness=bool(hit and rid in CARDS and card_witness(rid,raw)),
                     source="official answer wrong" if not hit else "awaiting content review")
            attach_review(rec,raw,reviews.get(key))
            if key not in reviews and (hit or int(key[:6],16)%10==0):
                queue[key]=dict(blind_id=key,id=rid,reference=public_item(r),reference_card=CARDS.get(rid),
                    original_image_lookup="mmmu_first_baseline_20261009/analysis/images/index.json",
                    rubric="Use the uncropped original images, official answer and domain references as needed. Evaluate displayed evidence, rule, application and their connection regardless of section titles. Missing support is unknown, not a proven false claim.",
                    raw=raw)
            armrows.append(rec)
        allrows[arm]=armrows
    finish(args,"MMMU_semantic_review_subset",allrows,queue,reviews,dict(
        roster_sha256=hashed([public_item(base[r]) for r in ids]),
        inputs={arm:hashlib.sha256(path.read_bytes()).hexdigest() for arm,path in (("baseline",args.baseline),("candidate",args.candidate))},
        reference_card_ids=sorted(set(ids)&CARDS.keys()),witness_denominator=len(set(ids)&CARDS.keys()),
        witness_interpretation="literal source-premise/operation markers among correct responses on inspected reference cards; triage only, not full-response semantic grades",
        reference_provenance="five inspectable assistant-authored calibration cards, four distinct source image sets; not a human-authored MMMU-wide oracle",
        scope="review subset only; official 900 accuracy belongs to the separate frozen scoring protocol"))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("chart-plan");p.add_argument("--gold",type=Path,required=True);p.add_argument("--families",type=int,default=24)
    p.add_argument("--out",type=Path,required=True);p.set_defaults(action=chart_plan)
    p=sub.add_parser("chart-score");p.add_argument("--spec",type=Path,required=True);p.add_argument("--base-preds",type=Path,required=True)
    p.add_argument("--preds",type=Path,required=True);p.add_argument("--semantics",type=Path)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--reviews",type=Path);p.set_defaults(action=chart_score)
    for cmd,fn in (("prepare",prepare),("mmmu-score",mmmu_score)):
        p=sub.add_parser(cmd);p.add_argument("--baseline",type=Path,required=True);p.add_argument("--ids",type=Path)
        p.add_argument("--out",type=Path,required=True);p.set_defaults(action=fn)
        if cmd=="mmmu-score":
            p.add_argument("--candidate",type=Path,required=True);p.add_argument("--reviews",type=Path)
    args=ap.parse_args()
    if args.cmd=="chart-plan" and args.families<1:ap.error("families must be positive")
    args.action(args)


if __name__=="__main__":main()
