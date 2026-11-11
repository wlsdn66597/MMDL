"""Independent question-only comparison; remove true option lists and known metadata wrappers."""
from collections import Counter,defaultdict
import difflib
import json
import re
from zipfile import ZipFile
import check_overlap as audit

ROOT=audit.ROOT
def clean(q,opts=None):
    q=re.sub(r"<\s*image(?:[_ ]?\d+)?\s*>"," ",q,flags=re.I).strip()
    q=re.sub(r"(?is)^.*?\bQuestion:\s*", "",q,count=1)
    q=re.split(r"\n(?:On the last line|Show the image readings|Please select )",q,1)[0]
    mode="no_cut"
    explicit=re.search(r"(?im)^\s*(?:Options|Choices)\s*:",q)
    if explicit:
        q=q[:explicit.start()]
        mode="explicit_heading"
    elif isinstance(opts,list) and len(opts)>=2 and all(isinstance(v,str) and v.strip() for v in opts):
        markers=list(re.finditer(r"(?<!\w)(?:\(([A-H])\)|\[([A-H])\]|([A-H])[.、):])\s*",q))
        for mindex, m in enumerate(markers):
            tail=markers[mindex:]
            if len(tail)!=len(opts):continue
            if [next(x for x in t.groups() if x) for t in tail] != list("ABCDEFGH"[:len(opts)]): continue
            vals=[q[t.end():(tail[j+1].start() if j+1<len(tail) else len(q))] for j,t in enumerate(tail)]
            if [audit.qbody(val) for val in vals]!=[audit.qbody(val) for val in opts]:continue
            q=q[:m.start()];mode="provided_options_verified";break
        if mode=="no_cut":
            # Distinguish a line-oriented fallback from a verified option-list match.
            line_marks=list(re.finditer(r"(?m)^[ \t]*(?:\(([A-H])\)|\[([A-H])\]|([A-H])[.、):])\s*",q))
            code=[next(x for x in t.groups() if x) for t in line_marks]
            if len(code)>=2 and code==list("ABCDEFGH"[:len(code)]):
                q=q[:line_marks[0].start()];mode="labelled_lines_candidate"
    q=re.sub(r"(?i)\s*\bContext:\s*(?:N\s*/\s*A|None)\s*[.]?\s*$","",q)
    return audit.qbody(q),mode

def refs():
    files={
      "donor_train200":"code/training/era_decoder_lora/data/pilot200/exports/train_era.jsonl",
      "D226":"results/d226_echeck_20261105/data/standard.jsonl",
      "old_train449":"results/era_single_call_campaign/data/train.jsonl","old_dev76":"results/era_single_call_campaign/data/dev.jsonl",
      "round4_train164":"results/era_a_pair_round/data_work/train.jsonl","round4_dev48":"results/era_a_pair_round/connection_work/spec/inputs.json",
      "round5_teacher192":"results/era_midstep_round/data/teacher_gold.json","round5_train96":"results/era_midstep_round/data/train_pairs.jsonl",
      "round5_dev_charts96":"results/era_next_direction/diagnostic/inputs.json","round5_dev_ScienceQA19":"results/era_next_direction/retention/native_validation/inputs.json",
      "MMMU900":"results/mmmu_first_baseline_20261009/baseline_seed3407/results.json"}
    sets={}
    for k,file in files.items():
        p=ROOT/file
        rs=[json.loads(x) for x in p.read_text(encoding="utf-8-sig").splitlines()] if p.suffix==".jsonl" else audit.read(p)
        if k=="MMMU900":rs=rs["records"]
        sets[k]=[(r.get("id") or r["sample_id"],*clean(r.get("question") or r.get("input_prompt") or r["prompt"],r.get("options"))) for r in rs]
    return sets

def main():
    with ZipFile(audit.DEFAULT_ZIP) as archive:
        r=json.loads(archive.read("data.json"))
    cases=defaultdict(list);modes=Counter()
    for row in r:
        s,m=clean(row["question"],row["options"]); cases[s].append(row["id"]);modes[m]+=1
    corpus=refs()
    matches=[];summary={}
    for name,rr in corpus.items():
        inv=defaultdict(list)
        for rid,s,_ in rr: inv[s].append(rid)
        exact={s:{"vic":ids,"ref":inv[s]} for s,ids in cases.items() if s and s in inv}
        approx=[]; long=[(s,set(s.split())) for s in inv if len(s.split())>=12]
        for s,ids in cases.items():
            sc=set(s.split())
            if len(s.split())<12:continue
            for q,qc in long:
                if s==q or not (.80<=len(s)/len(q)<=1.25):continue
                if len(sc&qc)<.8*max(len(sc),len(qc)):continue
                ratio=difflib.SequenceMatcher(None,s,q,autojunk=False).ratio()
                if ratio>=.9:approx.append(dict(vic=ids,ref=inv[q],stem=s,other_stem=q,ratio=ratio))
        summary[name]={"exact_stems":len(exact),"vic_rows_with_exact_stem":len({i for p in exact.values() for i in p["vic"]}),
                       "near_text_vic":len({i for row in approx for i in row["vic"]})}
        matches.append(dict(set=name,exact=exact,similar=approx))
    duplicates=sorted([(len(v),s) for s,v in cases.items() if len(v)>1],reverse=True)
    audit.dump("question_stems.json",dict(option_treatment=modes,distinct_stems=len(cases),repeated_stems=len(duplicates),
                  rows_on_repeated_stems=sum(n for n,s in duplicates),repeated_stem_examples=duplicates[:10],summary=summary,matches=matches))
    print(json.dumps(dict(modes=modes,distinct=len(cases),repeated=len(duplicates),members=sum(n for n,s in duplicates),summary=summary)))

if __name__=="__main__":main()
