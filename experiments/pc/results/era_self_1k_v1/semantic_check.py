"""Conservative, table-grounded scoring. Unparsed reasoning must be reviewed."""
import re
from decimal import Decimal

D = Decimal
NUM = r"[-+]?(?:\d[\d,]*\.\d+|\d{1,3}(?:,\d{3})+(?!\d)|\d+)"
TRIO = rf"{NUM}\s*(?:,\s*(?:and\s+)?|\s+and\s+|\+\s*){NUM}\s*(?:,\s*(?:and\s+)?|\s+and\s+|\+\s*){NUM}"
HEADER = re.compile(r"(?i)\b(Evidence|Rule|Application|Final\s+answer)\s*\**:\s*\**")
COMPARISON = re.compile(
    rf"The (?:lowest|smallest) excluded bar is ([\w -]+?) at ({NUM}), (above|below) ([\w -]+?) at ({NUM})\.", re.I
)


def decimal(value):
    return D(str(value).strip().replace(",", ""))


def numbers(text):
    return [decimal(v) for v in re.findall(rf"(?<![\w.])({NUM})(?!\w)", text)]


def terminal(raw, options):
    matches = list(re.finditer(r"(?i)\bFinal\s+answer\s*:\s*\**([^\r\n]*)", raw))
    field = (matches[-1][1] if matches else (raw.strip().splitlines() or [""])[-1]).strip(" \t*.")
    regex = r"^(?:option\s+)?[\[(]?([A-D])\b" if matches else r"^(?:option\s+)?[\[(]?([A-D])[\]).!\s]*$"
    letter = re.match(regex, field, re.I)
    if letter:
        return letter[1].upper()
    if not matches:
        letter = re.search(r"(?:the answer is|select(?:ing)?|matches? option)\s+(?:option\s+)?([A-D])\s*$", field, re.I)
        if letter:
            return letter[1].upper()
    numeric = re.fullmatch(rf"[\[(]?({NUM})[\])%]*", field)
    if numeric:
        candidates = [i for i,o in enumerate(options) if decimal(o)==decimal(numeric[1])]
        if len(candidates)==1:
            return "ABCD"[candidates[0]]
    return None


def sections(raw):
    matches = list(HEADER.finditer(raw))
    order = [m[1].lower() for m in matches]
    if order != ["evidence","rule","application","final answer"]:
        return None
    return [raw[a.end():b.start()].strip() for a,b in zip(matches,matches[1:])]


def evidence_content(raw):
    """Isolate E even if later headers are malformed, for first-read diagnostics."""
    match = re.search(r"(?i)\bEvidence\s*\**:\s*\**",raw)
    if not match:
        return ""
    end = re.search(r"(?i)\b(?:Rule|Application|Final\s+answer)\s*\**:",raw[match.end():])
    return raw[match.end():match.end()+end.start()].strip() if end else raw[match.end():].strip()


def table(g):
    return {k.lower():decimal(v) for k,v in g["rows"].items()}


def read_initial(e, g):
    if g["mode"]=="direct":
        name = re.escape(g["direct_name"])
        match = re.search(rf"\b{name}(?:\s+bar)?\s+(?:(?:is|reads?|has|ends at|shows|is labeled|has value|at)\s+)+(?:value\s+)?({NUM})",e,re.I)
        if match:
            return [decimal(match[1])]==[decimal(g["truth"])],None
        return None,None
    # Parse the first designated candidate set, never infer it from a gold answer or later summation.
    prefix = re.sub(r"^(?:(?:Check|Direct(?: read)?|Verify|Recheck)\s*\.\s*)+", "", e, flags=re.I)
    match = re.match(rf"(?:My\s+)?(?:initial\s+)?(?:candidates|selection|three (?:smallest|shortest) (?:values|bars)|three bars)\s*(?:are|is|:)\s*({TRIO})(?![\w\d])", prefix,re.I)
    if not match:
        match = re.match(rf"(?:(?:Is|Are|check)\s+)?({TRIO})\s*(?:=|the three|the sum|the smallest|the shortest)", prefix,re.I)
    if not match:
        ordinal = re.match(rf"(?:Is|Are)\s+({NUM})\s+the smallest\b",prefix,re.I)
        parts = re.findall(rf"\bIs\s+({NUM})\s+(?:the\s+)?(?:smallest|second(?:[- ]smallest)?|third(?:[- ]smallest)?)\b",prefix,re.I)
        if not ordinal or len(parts)<3:
            return None,None
        vals=[decimal(x) for x in parts[:3]]
    else:
        vals=numbers(match[1])
    if len(vals)!=3:
        return None,None
    return sorted(vals)==sorted(map(decimal,g["truth_vals"])),vals


def strip_start_ops(e):
    return re.sub(r"^(?:(?:Check|Direct(?: read)?|Recheck|Verify)\s*\.\s*)+", "",e,flags=re.I)


def step_trace(e, g):
    """Validate the entire canonical selection→comparison→decision→verified-set sequence."""
    current = strip_start_ops(e)
    match = re.match(rf"(?:My\s+)?initial candidates are\s+({TRIO})\.\s*",current,re.I)
    if not match:
        return None
    candidates = numbers(match[1])
    vs=table(g); all_values=set(vs.values()); picked=set(candidates)
    if len(candidates)!=3 or not picked<=all_values:
        return None
    current=strip_start_ops(current[match.end():]); corrected=False; full=True; last_action=None
    compared=False; valid_prefix=True
    direct=re.fullmatch(rf"The selected bars are distinct and the remaining bars are visibly longer\. Verified values:\s*({TRIO})\.?",current,re.I)
    if direct:
        settled=numbers(direct[1])
        valid=len(picked)==3 and picked==set(settled) and sorted(settled)==sorted(map(decimal,g["truth_vals"])) and min(all_values-picked)>max(picked)
        return dict(initial=candidates,settled=settled,changed=False,valid=valid,boundary=False)
    duplication=re.compile(rf"([\w -]+?) at ({NUM}) was counted more than once\. Count that row once and fill the vacant place with ([\w -]+?) at ({NUM}), the lowest unselected row\.\s*",re.I)
    while len(picked)<3:
        step=duplication.match(current)
        if not step:return None
        repeated,base,missing,entrant=step.groups()
        base,entrant=decimal(base),decimal(entrant)
        if not (vs.get(repeated.lower())==base and candidates.count(base)>1 and vs.get(missing.lower())==entrant and
                entrant in all_values-picked and entrant==min(all_values-picked)):
            return None
        picked.add(entrant);corrected=True
        current=current[step.end():]
    while True:
        ending=re.fullmatch(rf"(?:Verified|Confirmed) values:\s*({TRIO})\.?",current,re.I)
        if ending:
            settled=numbers(ending[1])
            return dict(initial=candidates, settled=settled, changed=corrected, boundary=compared,
                        valid=full and compared and last_action=="keep" and picked==set(settled) and
                        sorted(settled)==sorted(map(decimal,g["truth_vals"])))
        query=re.match(rf"(?:Is any excluded bar smaller than ({NUM})\?|Compare the smallest excluded bar with ({NUM})\.)\s*",current,re.I)
        if query:
            full &= decimal(query[1] or query[2])==max(picked)
            current=current[query.end():]
        step=COMPARISON.match(current)
        if not step:
            return None
        compared=True
        outside,n,op,inside,k=step.groups()
        n,k=decimal(n),decimal(k)
        is_valid=(vs.get(outside.lower())==n and vs.get(inside.lower())==k and n in all_values-picked and
                  n==min(all_values-picked) and k==max(picked) and (op.lower()=="below")==bool(n<k))
        valid_prefix &= is_valid
        full &= is_valid
        current=current[step.end():].lstrip()
        repl=re.match(rf"Replace ({NUM}) with ({NUM})\.\s*",current,re.I)
        if repl:
            valid_repl=n<k and k in picked and n in all_values-picked and decimal(repl[1])==k and decimal(repl[2])==n
            full &= valid_repl; valid_prefix &= valid_repl
            corrected |= valid_prefix
            if valid_repl:
                picked.remove(k);picked.add(n)
            last_action="replace";current=current[repl.end():]
        else:
            keep=re.match(r"Keep (?:the|my) selection\.\s*",current,re.I)
            if not keep:
                return None
            full &= n>=k;valid_prefix &= n>=k
            last_action="keep";current=current[keep.end():]


def checked_equation(text,g):
    match=re.fullmatch(rf"(?:(?:The (?:verified |requested )?(?:sum|total) is|Summing(?: these| those| the values)?(?: gives)?|Check)\s+)?({NUM})\s*\+\s*({NUM})\s*\+\s*({NUM})\s*=\s*({NUM})(?:,?\s*(?:option|choice)\s+([A-D]))?[.]?",text,re.I)
    if not match: return None
    vals=[decimal(x) for x in match.groups()[:3]]
    return sorted(vals)==sorted(map(decimal,g["truth_vals"])) and sum(vals)==decimal(match[4])==decimal(g["truth"]) and (match[5] is None or match[5].upper()==g["letter"])


def direct_evidence(e,g):
    value_re = rf"({NUM})"; name=re.escape(g["direct_name"])
    if not e:return None
    match=re.fullmatch(rf"(?:Direct(?: read)?\.\s*)?(?:The\s+)?{name}(?:\s+bar)?\s+(?:is labeled|has value|is|shows|is labeled with|reads|at)\s+{value_re}\.?",e,re.I)
    if not match:
        match=re.fullmatch(rf"Read (?:the labeled bar for {name}|the {name} bar(?:'s)? label)\. (?:Its|The) (?:printed )?value is {value_re}\.?",e,re.I)
    return None if not match else decimal(match[1])==decimal(g["truth"])


def rank_evidence(e,g):
    src=strip_start_ops(e);truth=sorted(map(decimal,g["truth_vals"]));vs=table(g)
    allvals=set(vs.values())
    # A correct table sorting is a check on the chart; every value including the cutoff must match.
    listing=re.fullmatch(rf"Is ({TRIO}) the three smallest(?: values| bars)?\? The (?:sorted values are|chart shows) (.+?)\. Yes\.", src, re.I)
    if listing:
        chosen=numbers(listing[1]); listed=numbers(listing[2])
        return sorted(chosen)==truth and listed==sorted(allvals)
    simple=re.fullmatch(rf"Is ({TRIO})\s*=\s*({NUM})\? (?:Yes|Verified)\.?",src,re.I)
    if simple:
        vals=numbers(simple[1]);return sorted(vals)==truth and sum(vals)==decimal(simple[2])
    # Short rank questions with verifiable ordering. Do not discard free-text clauses.
    actual=[];valid=True
    for i,ordinals in enumerate(("smallest", "second[- ]smallest|second|next", "third[- ]smallest|third")):
        pattern=rf"Is ({NUM}) (?:the )?(?:{ordinals})(?: value| bar)?\? Yes((?:(?:\.(?!\s|$))|[^.])*)\.(?:\s+|$)"
        m=re.match(pattern,src,re.I)
        if not m:return None
        value=decimal(m[1]);actual.append(value);valid &= value==truth[i]
        why=m[2].lstrip(", ").strip()
        why=re.sub(r"^(?:because\s+)?(?:it is\s+)?","",why,flags=re.I)
        if why:
            inequality=re.fullmatch(rf"({NUM})\s*([<>])\s*({NUM})",why)
            if inequality:
                a,b=decimal(inequality[1]),decimal(inequality[3])
                valid &= value in (a,b) and a in allvals and b in allvals and (a<b if inequality[2]=="<" else a>b)
            else:
                pattern=rf"(above|below) (?:([\w-]+) at )?({NUM}|[a-z]+)"
                for clause in re.split(r" (?:and|but) ",why,flags=re.I):
                    comp=re.fullmatch(pattern,clause,re.I)
                    if not comp:return None
                    ref=comp[3].lower()
                    if re.fullmatch(NUM,ref):
                        other=decimal(ref)
                        valid &= other in allvals and (comp[2] is None or vs.get(comp[2].lower())==other)
                    else:
                        if ref not in vs:return None
                        other=vs[ref]
                    valid &= value<other if comp[1].lower()=="below" else value>other
        src=src[m.end():]
    if src:
        last=re.fullmatch(rf"Verified(?: values:\s*({TRIO}))?\.",src,re.I)
        if not last:return None
        if last[1]:valid &= sorted(numbers(last[1]))==truth
    return valid


def safe_rule(r, direct):
    # Only certify a constrained rule. A contradictory free-form qualifier belongs in the audit queue.
    text=r.strip().rstrip(".")
    if direct:
        return bool(re.fullmatch(r"(?:Read|Extract|Report) the (?:requested |printed |labeled |chart )?value (?:printed )?(?:beside|on|from) the (?:named|specified|requested|selected)(?: chart)? (?:bar|label)",text,re.I))
    return bool(re.fullmatch(r"(?:Add|Sum) the three (?:smallest|shortest|lowest)(?: chart| charted| verified| selected)? (?:values|bars)(?:,? (?:after (?:verifying|confirming) (?:the |my )?(?:selection|selections|selected values|which bars they are|their identities|them)|using the evidence|after verifying they are (?:correct|distinct and correctly identified)))?",text,re.I))


def assess(g,p,mode):
    vs=table(g); raw=p["raw"]; observed=terminal(raw,g["options"]); correct=p.get("finish_reason")=="stop" and observed==g["letter"]
    blocks=sections(raw); e=evidence_content(raw); init,_=read_initial(e,g)
    exact_joint=repair=retain=grounded_keep=full_check=None
    trace=step_trace(e,g) if g["mode"]!="direct" else None
    evidence_ok=None
    if trace:
        evidence_ok=trace["valid"]
        init=sorted(trace["initial"])==sorted(map(decimal,g["truth_vals"]))
    elif g["mode"]=="direct":
        evidence_ok=direct_evidence(e,g)
        if evidence_ok is not None:init=evidence_ok
    else:
        evidence_ok=checked_equation(e,g)
        if evidence_ok is not None:
            if init is None:init=evidence_ok
        else:
            evidence_ok=rank_evidence(e,g)
    draft=p.get("draft") if mode=="self" else g.get("draft") if mode=="common" else None
    given=sorted(map(decimal,draft))==sorted(map(decimal,g["truth_vals"])) if draft is not None else None
    app_ok=None; rule_ok=False
    if blocks:
        _,rule,app=blocks;rule_ok=safe_rule(rule,g["mode"]=="direct")
        if g["mode"]=="direct":
            m=re.fullmatch(rf"The requested value is ({NUM}), option ([A-D])\.?",app,re.I)
            if m:app_ok=decimal(m[1])==decimal(g["truth"]) and m[2].upper()==g["letter"]
        else:
            app_ok=checked_equation(app,g)
    if not correct:
        exact_joint=repair=retain=grounded_keep=full_check=False
    else:
        if evidence_ok is False or app_ok is False:exact_joint=False
        elif evidence_ok is True and app_ok is True and rule_ok:exact_joint=True
        if trace and app_ok is True:
            right_set=sorted(trace["settled"])==sorted(map(decimal,g["truth_vals"]))
            repair=bool(trace["changed"] and right_set and init is False and
                        (draft is None or sorted(map(decimal,draft))==sorted(trace["initial"])))
            same_draft=draft is None or sorted(map(decimal,draft))==sorted(trace["initial"])
            full_check=bool(trace["valid"] and trace["boundary"] and right_set and same_draft and
                            (draft is None or given is True or repair is True))
            if given is True:
                retain=(sorted(trace["initial"])==sorted(map(decimal,draft)) and not trace["changed"] and right_set)
                grounded_keep=retain and full_check
        if init is True and draft is None:repair=False
        if given is True and exact_joint is True and evidence_ok is True and not trace:
            retain=True
        if given is False:retain=grounded_keep=False
    return dict(chosen=observed,answer=correct,finish=p.get("finish_reason"),format_ok=bool(blocks and all(blocks) and re.fullmatch(r"Final answer:\s*[A-D]",(raw.strip().splitlines() or [""])[-1],re.I)),
                initial=init,joint=exact_joint,repair=repair,retain=retain,grounded_keep=grounded_keep,full_check=full_check,
                supplied_right=given,trace=bool(trace),asks=("?" in e),tokens=p.get("output_tokens"))
