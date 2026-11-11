"""Analyze completed blind 1k pilot runs. stdlib only. Run 'python analyze.py'."""
import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from semantic_check import assess, decimal

ROOT=Path(__file__).resolve().parent
ARMS=("start","priorcontrol","priorpaired","A","B")
MAIN=("start","A","B")
FIELDS=("initial","joint","repair","retain","grounded_keep","full_check")


def read(p):return json.loads(p.read_text(encoding="utf8"))


def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    tmp.write_text(text,encoding="utf8")
    tmp.replace(path)


def json_text(value):return json.dumps(value,ensure_ascii=False,indent=2)+"\n"


def jsonl_text(rows):return "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)


def require(condition,msg):
    if not condition:raise ValueError(msg)


def blind_id(mode,arm,it,p):
    key=json.dumps([mode,arm,it["id"],it["image_sha256"],p["raw"]],ensure_ascii=False,separators=(",",":"))
    return hashlib.sha256(key.encode()).hexdigest()[:20]


def review_row(labels,row,raw):
    override=labels.get(row["blind_id"])
    if override is None:return
    require(bool(override.get("reviewer")) and bool(override.get("reason")), "Review requires reviewer and reason")
    allowed=set(FIELDS)|{"reviewer","reason","quote"}
    require(set(override)<=allowed,"Unknown review field")
    for key in FIELDS:
        if key in override:
            require(override[key] is None or type(override[key]) is bool,f"Invalid {key} review")
            row[key]=override[key]
    quote=override.get("quote","")
    if override.get("repair") or override.get("grounded_keep"):
        require(bool(quote) and quote in raw,"Grounding review requires verbatim evidence")
    require(row["answer"] or not any(row[k] is True for k in FIELDS[1:]),"Semantic success on incorrect answer")
    require(not row["repair"] or (row["initial"] is False if row["mode"]=="natural" else row["supplied_right"] is False),"Repair must start from wrong candidate")
    require(not row["retain"] or row["supplied_right"] is True,"Retention requires correct supplied draft")
    require(not row["grounded_keep"] or row["retain"],"Grounded keep requires retention")
    row["reviewer"]=override["reviewer"];row["review_reason"]=override["reason"]


def load(root,out,partial):
    receipt=read(root/"holdout/receipt.json")
    for name in ("gold","inputs"):
        require(sha(root/"holdout"/(name+".json"))==receipt[name+"_sha256"],"Frozen holdout hash changed: "+name)
    gold=read(root/"holdout/gold.json")
    byid={i["id"]:i for i in gold}
    require(len(byid)==len(gold)==receipt["n"],"Gold has repeated IDs or wrong count")
    for it in gold:
        vals=sorted(map(decimal,it["rows"].values()))
        target=decimal(it["rows"][it["direct_name"]]) if it["mode"]=="direct" else sum(vals[:3])
        require(target==decimal(it["truth"])==decimal(it["options"]["ABCD".index(it["letter"])]),"Gold arithmetic mismatch "+it["id"])
    labels=read(out/"adjudication.json") if (out/"adjudication.json").exists() else {}
    scores=[];queue=[];completed={};covs={};original={};prompts={};receipts={};imghashes={}
    for arm in ARMS:
        for mode in (("natural","common","self") if arm in MAIN else ("natural","common")):
            path=root/"eval"/arm/mode
            key=arm+"/"+mode; flag=path/"done.json"
            completed[key]=flag.is_file()
            if not completed[key]:
                if partial:continue
                raise FileNotFoundError(flag)
            file=path/"predictions.jsonl";mark=read(flag); receipts[key]=sha(file)
            require(mark["output_sha256"]==receipts[key],"Completed output hash changed "+key)
            records=[json.loads(s) for s in file.read_text(encoding="utf8").splitlines()]
            expected={r["id"] for r in gold if (r["mode"]=="draft")== (mode=="common")}
            cv={}
            if mode=="self":
                cv={x["id"]:x for x in read(path/"coverage.json")}
                require(set(cv)=={r["id"] for r in gold if r["mode"]=="rank"},"Self coverage does not match rank holdout")
                expected={k for k,x in cv.items() if x["draft"] is not None}
                covs[arm]=cv
            require(len(records)==mark["n"]==len(expected) and expected=={p["id"] for p in records},"Incomplete or unexpected outputs "+key)
            for pred in records:
                rid=pred["id"];g=byid[rid]
                if g["image"] not in imghashes:
                    imghashes[g["image"]]=sha(root/"holdout"/g["image"])
                require(pred["image_sha256"]==g["image_sha256"]==imghashes[g["image"]],"Image hash "+rid)
                if mode=="natural":
                    original[arm,rid]=pred["raw"]
                if mode!="self":
                    require(prompts.setdefault((mode,rid),pred["prompt"])==pred["prompt"],"Different input between arms "+rid)
                else:
                    require(pred["draft"]==cv[rid]["draft"] and pred["policy_check"]==cv[rid]["policy_check"] and
                            pred["first_pass_id"]==rid and pred["original_first_raw"]==original.get((arm,rid)),"Self receipt mismatch "+rid)
                scored=assess(g,pred,mode)
                scored.update(id=rid,chart=g["chart"],kind=g["mode"],arm=arm,mode=mode,
                              blind_id=blind_id(mode,arm,g,pred),reviewer=None,policy_check=pred.get("policy_check"))
                review_row(labels,scored,pred["raw"])
                scores.append(scored)
                if mode=="natural":
                    needs=scored["initial"] is None or scored["joint"] is None or (scored["initial"] is False and scored["repair"] is None)
                elif scored["supplied_right"]:
                    needs=scored["retain"] is None or scored["grounded_keep"] is None
                else:
                    needs=scored["repair"] is None or scored["full_check"] is None
                positive=scored["repair"] is True
                sample=int(scored["blind_id"][:8],16)%20==0
                if (needs or positive or sample) and scored["reviewer"] is None:
                    queue.append(dict(blind_id=scored["blind_id"],reason="unresolved" if needs else "positive repair" if positive else "negative audit sample",
                        mode=mode,id=rid,question=g["question"],rows=g["rows"],options=g["options"],true_option=g["letter"],
                        image=g["image"],draft=pred.get("draft",g.get("draft")),response=pred["raw"]))
    require(partial or set(labels)<={r["blind_id"] for r in scores},"Stale/unmatched adjudication keys")
    queue.sort(key=lambda x:x["blind_id"])
    return scores,queue,completed,covs,receipts


def counts(rows,field):
    values=[r[field] for r in rows]
    return {"yes":values.count(True),"no":values.count(False),"unresolved":values.count(None),"n":len(rows)}


def summary(rows):
    natural=[r for r in rows if r["mode"]=="natural"]
    false=[r for r in natural if r["initial"] is False]
    res=dict(natural=dict(answer=counts(natural,"answer"),joint=counts(natural,"joint"),
                          initial=counts(natural,"initial"),repair_if_wrong=counts(false,"repair"),
                          confirmed_repair_per100=round(100*sum(r["repair"] is True and r["initial"] is False for r in natural)/len(natural),2) if natural else None,
                          strict_format=counts(natural,"format_ok"),questions=counts(natural,"asks")))
    for mode in ("common","self"):
        subset=[r for r in rows if r["mode"]==mode]
        if not subset:continue
        keep=[r for r in subset if r["supplied_right"]]
        change=[r for r in subset if not r["supplied_right"]]
        res[mode]=dict(n=len(subset),keep_answer=counts(keep,"answer"),retain=counts(keep,"retain"),grounded_keep=counts(keep,"grounded_keep"),
                       change_answer=counts(change,"answer"),repair=counts(change,"repair"),full_check_after_change=counts(change,"full_check"),
                       avg_output_tokens=sum(r.get("tokens") or 0 for r in subset)/len(subset))
        if mode=="common":
            families=defaultdict(dict)
            for r in subset:families[r["chart"]][r["supplied_right"]]=r
            require(all(set(v)=={True,False} for v in families.values()),"Incorrect/missing paired common draft")
            def both(x,y):return False if x is False or y is False else True if x is True and y is True else None
            pairs=[dict(answer=both(v[True]["answer"],v[False]["answer"]),
                        evidence=both(v[True]["grounded_keep"],v[False]["repair"])) for v in families.values()]
            res[mode].update(pair_answer=counts(pairs,"answer"),pair_evidence=counts(pairs,"evidence"))
    return res


def strategies(rows,cov):
    """Compare terminal decisions on the very same cached drafts; account for all skipped inputs."""
    naturals={r["id"]:r for r in rows if r["mode"]=="natural"}
    replay={r["id"]:r for r in rows if r["mode"]=="self"}
    total=len(naturals)
    result={}
    for mode in ("forced","selected"):
        calls={rid for rid in replay if mode=="forced" or cov[rid]["policy_check"] is True}
        transitions=Counter();correct=0;tok=sum(r["tokens"] or 0 for r in naturals.values())
        for rid,row in naturals.items():
            after=replay[rid]["answer"] if rid in calls else row["answer"]
            correct+=after
            transitions[("kept" if after else "lost") if row["answer"] else ("gained" if after else "both_wrong")]+=1
            if rid in calls:tok+=replay[rid]["tokens"] or 0
        # Additional diagnostic: adopting the first candidate versus rescoring it, only on extracted drafts.
        draft_transitions=Counter()
        for rid,r in replay.items():
            initial=r["supplied_right"]
            after=r["answer"] if rid in calls else initial
            draft_transitions[("kept" if after else "lost") if initial else ("gained" if after else "both_wrong")]+=1
        result[mode]=dict(n=total,correct=correct,extra_calls=len(calls),call_rate=len(calls)/total if total else None,
                          avg_output_tokens=tok/total if total else None,first_terminal_to_final=dict(transitions),
                          from_frozen_draft=dict(draft_transitions))
    result.update(rank_coverage=len(replay),rank_n=len(cov),
                  policy_true=sum(r["policy_check"] is True for r in cov.values()),
                  policy_false=sum(r["policy_check"] is False for r in cov.values()),
                  policy_unknown=sum(r["policy_check"] is None for r in cov.values()))
    return result


def contrasts(rows):
    lookup={(r["arm"],r["mode"],r["id"]):r for r in rows}
    results={}
    for left,right in (("start","A"),("start","B"),("priorpaired","B"),("A","B")):
        cmp={}
        for mode,field in (("natural","answer"),("natural","joint"),("common","repair"),("common","retain")):
            pairs=[]
            for (arm,m,rid),r in lookup.items():
                other=lookup.get((right,m,rid))
                if arm!=left or m!=mode or other is None:continue
                if mode=="common" and r["supplied_right"]!=(field=="retain"):continue
                pairs.append((r["chart"],r[field],other[field]))
            if not pairs:continue
            cc=Counter();clusters=defaultdict(list)
            for family,a,b in pairs:
                if a is None or b is None:cc["unresolved"]+=1
                else:cc[("kept" if b else "lost") if a else ("gained" if b else "both_wrong")]+=1
                clusters[family].append((a,b))
            # Only a fully resolved matched sample supports the following interval.
            ci=None
            if not cc["unresolved"]:
                rng=random.Random(3407); diffs=[]; fam=list(clusters)
                for _ in range(1000):
                    sample=[pair for _ in fam for pair in clusters[rng.choice(fam)]]
                    diffs.append(100*sum(int(b)-int(a) for a,b in sample)/len(sample))
                diffs.sort();ci=[round(diffs[25],2),round(diffs[974],2)]
            cmp[mode+"_"+field]=dict(n=len(pairs),transitions=dict(cc),difference_pp_ci95=ci)
        results[left+"-to-"+right]=cmp
    return results


def show(count):
    return f"{count['yes']}/{count['n']}" if not count["unresolved"] else f"{count['yes']}–{count['yes']+count['unresolved']}/{count['n']}"


def make_report(summary_by_arm,complete,nqueue,fingerprint):
    txt=["# ERA 1k A/B 평가", "",
         f"평가 실행 {'완료' if complete else '진행 중'} · 익명 원문 검수 대기 {nqueue}건. 이 자동 보고서는 확대를 승인하지 않는다.", "",
         "범위 표기는 확정 성공 수부터 미판정 사례를 모두 성공으로 가정한 수까지다. 근거 연결과 수정은 실제 차트 자료와 대조한 정형 표현만 자동 확정했다. 다른 표현은 익명 검수 대상이다. 단일 시드, 동일 수치표의 변형이 함께 들어간 개발 세트다.","",
         "| 모델 | 자연 정답 | 자연 정답+ERA 연결 | 처음 틀린 E 중 이미지 근거 수정 | 첫 후보 미판정 | 동일 오답 초안 교체 | 맞는 초안 유지 |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for arm,res in summary_by_arm.items():
        n=res["natural"];common=res.get("common")
        txt.append("| "+" | ".join([arm,show(n["answer"]),show(n["joint"]),show(n["repair_if_wrong"]),str(n["initial"]["unresolved"]),
                    show(common["repair"]) if common else "–",show(common["retain"]) if common else "–"])+" |")
    txt+=["","자연 수정의 분모에는 첫 E가 명확하게 틀렸다고 판정된 사례만 들어간다. 처음 후보가 미판정인 사례는 별도 열에 남아 있다. 직접 판독과 비교 문항을 합쳐 집계하며, 고정 초안 진단은 별도 분모다.",
          "", "## 같은 모델의 자기 초안 재검토", "",
          "| 모델 | 숫자 초안 복사 가능 / 순위 전체 | 원래 답 | 강제 재검토 후 답 | 선택적 재검토 후 답 | 선택적 추가 호출 | 명시적 결정 없음 |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for arm,res in summary_by_arm.items():
        s=res.get("replay_strategy")
        if not s:continue
        f,z=s["forced"],s["selected"]
        txt.append(f"| {arm} | {s['rank_coverage']}/{s['rank_n']} | {show(res['natural']['answer'])} | {f['correct']}/{f['n']} | {z['correct']}/{z['n']} | {z['extra_calls']}/{z['n']} | {s['policy_unknown']} |")
    txt+=["","재검토 정책은 첫 답의 명시적 접두어를 이용해 재구성했다. 복사할 수 없는 첫 후보와 직접 판독은 첫 답을 유지했다. 선택적 조건에서는 명시적인 재확인 결정이 있는 경우에만 2차 답으로 대체했다. 암묵적인 확인은 호출로 추정하지 않았다. 초안 추출 불가와 첫 선택의 의미 미판정은 서로 다른 기준이다.",
          "", "## 판정과 기록", "",
          "metrics.json에 자연 100문항당 확인된 수정 수, 동일 초안 가족의 양쪽 성공, 확정·손실 전이, 평균 토큰, 모델 간 짝 비교와 가족 단위 95% 부트스트랩 구간이 있다. 미판정이 있는 지표에서는 짝 차이 구간을 보류한다. 근거 기반 양성, 미판정 및 일부 음성은 blind_queue.jsonl에서 원문을 독립 검수한다. 1k A/B 게이트는 검수 후 데이터 규모·첫 오류 기회·보존 손실을 함께 판단해야 한다.",
          "", "실행 지문 `"+fingerprint+"`.", ""]
    return "\n".join(txt)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=ROOT)
    ap.add_argument("--partial",action="store_true",help="report complete modes so far without issuing final marker")
    args=ap.parse_args();root=args.root.resolve()
    out=root/("analysis_partial" if args.partial else "analysis");out.mkdir(parents=True,exist_ok=True)
    scores,queue,done,cov,receipts=load(root,out,args.partial)
    arms={}
    for arm in ARMS:
        rows=[r for r in scores if r["arm"]==arm]
        if not rows:continue
        arms[arm]=summary(rows)
        if arm in cov:arms[arm]["replay_strategy"]=strategies(rows,cov[arm])
    source_hashes={f:sha(root/f) for f in ("holdout/gold.json","holdout/inputs.json")}
    fingerprint=hashlib.sha256(json_text([receipts,source_hashes,sha(Path(__file__)),sha(Path(__file__).with_name("semantic_check.py")),
                                        sha(out/"adjudication.json") if (out/"adjudication.json").exists() else None]).encode()).hexdigest()
    complete=all(done.values())
    metrics=dict(fingerprint=fingerprint,completed=done,inputs=source_hashes,predictions=receipts,n_responses=len(scores),
                 n_queue=len(queue),arms=arms,contrasts=contrasts(scores),promotion="requires_review")
    write(out/"scores.jsonl",jsonl_text(scores));write(out/"blind_queue.jsonl",jsonl_text(queue));write(out/"metrics.json",json_text(metrics))
    report=make_report(arms,complete,len(queue),fingerprint)
    write(out/"report.md",report)
    if not args.partial:
        write(root/"report.md",report)
        write(out/"analysis.complete",json_text(dict(fingerprint=fingerprint,metrics_sha256=sha(out/"metrics.json"),n_responses=len(scores),
                                                    n_queue=len(queue),promotion="requires_review")))
    print(f"Analysis {'complete' if complete else 'partial'}: {len(scores)} responses, {len(queue)} blinded review rows. {out/'report.md'}",flush=True)


if __name__=="__main__":main()
