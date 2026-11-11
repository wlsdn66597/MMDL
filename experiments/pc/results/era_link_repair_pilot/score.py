"""Reference-assisted scoring; numeric checks plus inspectable per-response review overrides."""
import csv,hashlib,html,json,re
from collections import Counter
from decimal import Decimal as D
from pathlib import Path
from targets import section,first_set,nums,fmt
R=Path(__file__).resolve().parent
ARMS=("baseline","statement","question")
def read(p):return json.loads(p.read_text(encoding="utf8"))
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def triple(txt,vs):
    if not txt:return None
    line=txt.splitlines()[0]
    # direct first selection, not a later check or calculated sum
    cand=re.search(r"(?i)(?:initial (?:candidates|selection|values)|provisional (?:selection|values|set|candidates|three values)|initially (?:select|choose))\s*(?:are|is|of|:)?\s*([^?\n]+)",line)
    if cand:line=cand[1]
    line=re.sub(r"^(?:Check|Direct(?: read)?|Verify|Candidate check)\s*[.:–-]\s*","",line,flags=re.I)
    explicit=first_set(line,vs)
    if explicit:return explicit
    # An E may state one row per sentence, after a short "Check." prefix.
    # Only use statements designating chosen bars; exclude validation or comparison sentences.
    start=re.search(r"(?i)(?:smallest|shortest|lowest) (?:three|bar|bars|is|are)|(?:[A-Za-z]+ (?:is|at) [-+]?\d)",line)
    if not start:return None
    positions=line[start.start():]
    positions=re.split(r"(?i)\b(?:verify|verification|exclude[d]?|replace|revised|recheck|their sum|compare|instead)\b",positions,maxsplit=1)[0]
    numbers=[v for v in nums(positions) if v in vs.values()]
    if len(numbers)>=3:return sorted(numbers[:3])
    return None
def finalset(e,initial,vs):
    matches=list(re.finditer(r"(?i)(?:verified|confirmed|final|corrected) (?:three )?(?:values|selection|candidates|set|three bars|three)[ :]\s*(?:are|is)?\s*([^?\n]+)",e))
    for m in reversed(matches):
        val=first_set(m[1],vs)
        if val:return val
    # Many ordinary B outputs just pick a triple. A stated later replacement needs adjudication.
    if re.search(r"\breplace\b|\bswap\b|\binstead\b",e,re.I):return None
    return initial
def endletter(raw,opts):
    fields=re.findall(r"(?im)Final\s+answer\s*:\s*([^\n\r]*)",raw.strip())
    field=fields[-1].strip(" *") if fields else ""
    letter=re.match(r"(?:option\s+)?[\[(]?([A-D])\b",field,re.I)
    if letter:return letter[1].upper()
    numeric=re.fullmatch(r"[-+]?\d+(?:\.\d+)?[.% ]*",field)
    if numeric:
        match=[i for i,x in enumerate(opts) if D(x)==D(field.rstrip(".% "))]
        if len(match)==1:return "ABCD"[match[0]]
    if not fields:
        m=re.search(r"(?i)(?:the answer is|select(?:ing)?|matches option)\s+(?:option\s+)?([A-D])\b[ .!]*$",raw.strip())
        if m:return m[1].upper()
    return None
def inspect_trace(e,sel,vs):
    """Validate every fully parsed boundary comparison against the current set and all exclusions."""
    if not sel or len(sel)!=3:return None
    if not e:return None
    picked=set(sel);allvals=set(vs.values())
    pattern=r"(?:lowest|smallest) (?:excluded|unselected) (?:bar|value) is ([A-Za-z]+) at (\d+(?:\.\d+)?), (above|below) ([A-Za-z]+) at (\d+(?:\.\d+)?)"
    steps=re.findall(pattern,e,re.I)
    if not steps:return None
    lookup={k.lower():v for k,v in vs.items()}
    for excluded,n,op,chosen,k in steps:
        n,k=D(n),D(k)
        if lookup.get(excluded.lower())!=n or lookup.get(chosen.lower())!=k:return False
        if n!=min(allvals-picked) or k!=max(picked) or (op.lower()=="below")!=(n<k):return False
        if n<k:picked.remove(k);picked.add(n)
    return picked==set(sorted(allvals)[:3])
def assess(it,pred):
    raw=pred["raw"];e=section(raw,"Evidence");r=section(raw,"Rule");a=section(raw,"Application")
    vs={n:D(v) for n,v in it["rows"].items()}
    truth=set(map(D,it["truth_vals"])) if it["mode"]!="direct" else {D(it["truth"])}
    sel=triple(e,vs);verified=finalset(e,sel,vs)
    if it["mode"]=="direct":
        correct_read=it["direct_name"].lower() in e.lower() and D(it["truth"]) in nums(e)
        sel=verified=[D(it["truth"])] if correct_read else None
    trace=inspect_trace(e,sel,vs) if it["mode"]!="direct" else None
    label=endletter(raw,it["options"]);ans=pred["finish_reason"]=="stop" and label==it["letter"]
    rule=bool(r) and (bool(re.search(r"sum|add|total",r,re.I)) if it["mode"]!="direct" else bool(re.search(r"read|value|label|lookup|identify|locate|report",r,re.I)))
    if it["mode"]!="direct":
        eq=re.findall(r"(?<!\d)(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*\+\s*(\d+(?:\.\d+)?)\s*(?:=|gives|equals|is)\s*(\d+(?:\.\d+)?)",re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))","",a),flags=re.I)
        app=any(set(map(D,(x,y,z)))==truth and D(t)==sum(truth) for x,y,z,t in eq)
    else:app=D(it["truth"]) in nums(a)
    rec=dict(id=it["id"],mode=it["mode"],chart=it["chart"],layout=it["layout"],letter=label,answer=ans,
        initial=list(map(fmt,sel)) if sel else None,settled=list(map(fmt,verified)) if verified else None,
        init="right" if sel and set(sel)==truth else "wrong" if sel else "unclear",
        E=bool(verified) and set(verified)==truth and trace is not False,R=rule,A=app,parsed_trace=trace,
        format_ok=all(len(re.findall(r"(?im)^\s*"+name+r"\s*:",raw))==1 for name in ("Evidence","Rule","Application")) and
                  bool(re.fullmatch(r"(?i)Final answer:\s*[A-D]",raw.strip().splitlines()[-1])),
        explicit_question="?" in e,repair=bool(ans and sel and set(sel)!=truth and trace is True and verified and set(verified)==truth),
        draft_supported=bool(ans and it["mode"]=="draft" and trace is True and sel and set(sel)==set(map(D,it["draft"])) and verified and set(verified)==truth),
        reason="",tokens=pred["output_tokens"],finish=pred["finish_reason"])
    return rec
def main():
    gold=read(R/"holdout/gold.json");original={x["id"]:x for x in read(R/"holdout/inputs.json")}
    corrections=read(R/"adjudication.json") if (R/"adjudication.json").exists() else {}
    evidence=[];details={};stats={}
    freeze=read(R/"freeze.json")
    assert all(sha(R/p)==h for p,h in freeze.items())
    prompts={}
    for arm in ARMS:
        directory=R/"eval"/arm
        if not (directory/"predictions.jsonl").exists():continue
        preds=list(map(json.loads,(directory/"predictions.jsonl").read_text(encoding="utf8").splitlines()))
        assert len(preds)==len({p["id"] for p in preds})==72
        assert [p["id"] for p in preds]==list(original)
        byid={p["id"]:p for p in preds};items=[]
        for it in gold:
            rid=it["id"];pr=byid[rid]
            assert pr["image_sha256"]==sha(R/"holdout"/it["image"])==original[rid]["image_sha256"]
            if rid in prompts: assert pr["prompt"]==prompts[rid]
            else: prompts[rid]=pr["prompt"]
            rec=assess(it,pr);override=corrections.get(arm+"/"+rid,{})
            rec.update(override);rec["joint"]=bool(rec["answer"] and rec["E"] and rec["R"] and rec["A"])
            if rec["repair"]:assert rec["init"]=="wrong" and rec["answer"]
            rec["arm"]=arm;items.append(rec);evidence.append(rec)
        def group(rows):
            return dict(n=len(rows),answer=sum(x["answer"] for x in rows),joint=sum(x["joint"] for x in rows),
                strict_joint=sum(x["joint"] and x["format_ok"] for x in rows),
                first_wrong=sum(x["init"]=="wrong" for x in rows),first_unclear=sum(x["init"]=="unclear" for x in rows),
                grounded_repair=sum(x["repair"] for x in rows),format_ok=sum(x["format_ok"] for x in rows),
                E_questions=sum(x["explicit_question"] for x in rows),length=sum(x["finish"]=="length" for x in rows))
        stats[arm]={k:group([x for x in items if x["mode"] in modes]) for k,modes in
             [("natural",("rank","direct")),("rank",("rank",)),("direct",("direct",)),("draft",("draft",))]}
        for suffix in ("dkeep","dreplace"):
            rr=[x for x in items if x["id"].endswith(suffix)]
            stats[arm][suffix]=dict(n=len(rr),correct=sum(x["answer"] for x in rr),supported=sum(x["answer"] and x["draft_supported"] for x in rr))
        details[arm]=items
    save(R/"scores.json",evidence);save(R/"metrics.json",stats)
    if len(stats)==3:
        out={}
        for left,right in [("baseline","statement"),("baseline","question"),("statement","question")]:
            dd={}
            for key in ("joint","answer"):
                ct=Counter()
                for a,b in zip(details[left],details[right]):
                    assert a["id"]==b["id"]
                    if a["mode"]=="draft":continue
                    ct[("kept" if b[key] else "lost") if a[key] else ("gained" if b[key] else "both_failed")]+=1
                dd[key]=dict(ct)
            out[left+"-to-"+right]=dd
        save(R/"paired_changes.json",out)
if __name__=="__main__":main()
