"""Pair the three unprompted predictions by frozen chart ID; preserve exceptional outputs."""
import csv
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import html
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent
ARMS=("baseline","standard","checked")
LABEL={"baseline":"기존 ERA 3 epoch","standard":"일반 ERA","checked":"E 확인 질문 ERA"}
def read(path):return json.loads(path.read_text(encoding="utf8"))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
def load_rows(path):return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines()]
def escape(x):return html.escape(str(x))

def answer(r,opts,gold):
    text=r["raw"].strip()
    fields=re.findall(r"(?im)(?:^|\b)Final\s+answer\s*:\s*([^\r\n]*)",text)
    field=fields[-1].strip().strip("*").strip() if fields else ""
    m=re.match(r"^(?:option\s+)?[\[(]?([A-D])\b",field,re.I)
    mode="letter";letter=m[1].upper() if m else None
    if letter is None and field:
        numeric=re.fullmatch(r"[\[(]?([-+]?\d[\d,]*(?:\.\d+)?)[\])% .]*",field)
        if numeric:
            val=Decimal(numeric[1].replace(",",""));matches=[i for i,s in enumerate(opts) if Decimal(s)==val]
            if len(matches)==1:letter="ABCD"[matches[0]];mode="number"
    if not field and text:
        last_line=text.splitlines()[-1].strip()
        m=re.fullmatch(r"(?:option\s+)?[\[(]?([A-D])[\]).!\s]*",last_line,flags=re.I)
        if m:letter=m[1].upper();mode="bare_letter"
        else:
            m=re.search(r"(?:\bthe answer is\s+(?:option\s+)?|\bselect(?:ing)?\s+(?:option\s+)?|\b(?:matches|matching)\s+option\s+)([A-D])\b[\s.!*)]*$",last_line,flags=re.I)
            if m:letter=m[1].upper();mode="terminal_statement"
    e=re.search(r"(?ims)^\s*(?:\*\*)?Evidence\s*:(?:\*\*)?\s*(.*?)(?=^\s*(?:\*\*)?(?:Rule|Application|Final answer)\s*:|\Z)",text)
    evidence=e[1].strip() if e else ""
    questions=[s.strip() for s in re.findall(r"[^.?!\n]*\?",evidence) if s.strip("? ")]
    return dict(letter=letter,value=opts["ABCD".index(letter)] if letter else None,
                correct=letter==gold and r["finish_reason"]=="stop",mode=mode if letter else "unmapped",
                finish=r["finish_reason"],tokens=r["output_tokens"],evidence=evidence,
                E_question_mark=bool(questions),E_questions=questions,final_text=field[:200],
                standalone=bool(re.fullmatch(r"Final answer:\s*[A-D]",text.splitlines()[-1],re.I)))

def compare(rows,left,right):
    groups={k:[] for k in ("kept","lost","recovered","still_wrong")}
    for r in rows:
        a,b=r[left]["correct"],r[right]["correct"]
        groups[("kept" if b else "lost") if a else ("recovered" if b else "still_wrong")].append(r["id"])
    return groups

def main():
    fresh=ROOT/"fresh";inputs=read(fresh/"inputs.json");items=read(fresh/"items.json")
    receipt=read(fresh/"receipt.json")
    assert sha(fresh/"inputs.json")==receipt["inputs_sha256"] and sha(fresh/"items.json")==receipt["items_sha256"]
    inputs_by={j["id"]:j for j in inputs}; assert len(inputs_by)==64
    pred={}; manifest={}
    for arm in ARMS:
        raw=load_rows(ROOT/"eval"/arm/"predictions.jsonl")
        assert [r["id"] for r in raw]==[j["id"] for j in inputs]
        for r in raw:
            iid=r["id"];assert r["image_sha256"]==inputs_by[iid]["image_sha256"]==sha(fresh/inputs_by[iid]["image"])
        m=read(ROOT/"eval"/arm/"manifest.json"); assert m["inputs_sha256"]==receipt["inputs_sha256"]
        assert m["config_sha256"]==sha(ROOT/"eval_config.json") and m["eval_script_sha256"]==sha(ROOT/"eval.py")
        if arm!="baseline":
            assert m["weights"]==read(ROOT/"train"/arm/"merge.json")["weights"]
            assert all(m[k]==manifest["baseline"][k] for k in ("config_sha256","runner_frozen_prompt_sha256"))
        pred[arm]={r["id"]:r for r in raw};manifest[arm]=m
    receipt_server=read(ROOT/"server_receipt.json")
    for arm in ARMS:assert sha(ROOT/"eval"/arm/"predictions.jsonl")==receipt_server[arm]
    records=[]
    for it in items:
        for condition,cell in it["cells"].items():
            rid=cell["id"];row=dict(id=rid,chart=it["chart"],condition=condition,aligned=cell["aligned"],
                options=it["options"],truth=cell["truth"],bottom=cell["bottom"])
            assert len({pred[a][rid]["prompt"] for a in ARMS})==1
            assert row["truth"]["total"]==format(sum(sorted(map(Decimal,cell["rows"].values()))[:3]).normalize(),"f")
            for arm in ARMS:row[arm]=answer(pred[arm][rid],it["options"],cell["truth"]["letter"])
            records.append(row)
    summary={}
    for arm in ARMS:
        summary[arm]=dict(correct=sum(r[arm]["correct"] for r in records),
            aligned_correct=sum(r[arm]["correct"] for r in records if r["aligned"]),
            conflict_correct=sum(r[arm]["correct"] for r in records if not r["aligned"]),
            conflict_bottom=sum(r[arm]["letter"]==r["bottom"]["letter"] and r[arm]["finish"]=="stop" for r in records if not r["aligned"]),
            explicit_E_question=sum(r[arm]["E_question_mark"] for r in records),
            strict_final_line=sum(r[arm]["standalone"] for r in records),
            length_limited=sum(r[arm]["finish"]=="length" for r in records),
            unmapped=sum(r[arm]["letter"] is None for r in records),
            median_tokens=statistics.median(r[arm]["tokens"] for r in records))
        per_case={r["id"]:r[arm] for r in records}
        summary[arm]["same_values_both_orders_correct"]=sum(
            all(per_case[f'{it["chart"]}_v{v}o{o}']["correct"] for o in (0,1))
            for it in items for v in (0,1))
        summary[arm]["all_four_correct_charts"]=sum(all(per_case[c["id"]]["correct"] for c in it["cells"].values()) for it in items)
        if arm!="baseline":summary[arm]["vs_baseline"]=compare(records,"baseline",arm)
    summary["standard_to_checked"]=compare(records,"standard","checked")
    # Count improvement per family to avoid treating the four variants as independent charts.
    net={it["chart"]:sum(int(r["checked"]["correct"])-int(r["standard"]["correct"]) for r in records if r["chart"]==it["chart"]) for it in items}
    summary["per_chart_checked_minus_standard"]=net
    dump(ROOT/"comparisons.json",records);dump(ROOT/"summary.json",summary)
    with (ROOT/"comparisons.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(("id","aligned","correct_value","bottom_sum",*ARMS,"standard_correct","checked_correct","baseline_correct","checked_E_question"))
        for r in records:w.writerow((r["id"],r["aligned"],r["truth"]["total"],r["bottom"]["total"],
            *[r[a]["value"] for a in ARMS],r["standard"]["correct"],r["checked"]["correct"],r["baseline"]["correct"],r["checked"]["E_question_mark"]))
    tags=lambda x:'정답' if x else '오답'
    grid=[]
    for it in items:
        links=[]
        for r in (r for r in records if r["chart"]==it["chart"]):
            loc=r["id"];results=" / ".join("✓" if r[a]["correct"] else "✗" for a in ARMS)
            links.append(f'<td><a href="#{loc}">{results}</a></td>')
        grid.append('<tr><th>'+it["chart"]+'</th>'+''.join(links)+'</tr>')
    details=[]
    for r in records:
        rid=r["id"];cards=[]
        for arm in ARMS:
            result=r[arm];cards.append(f'<article><h3>{LABEL[arm]}　{tags(result["correct"])}</h3><p>선택 {escape(result["letter"])}　종료 {result["finish"]}</p><pre>{escape(pred[arm][rid]["raw"])}</pre></article>')
        details.append(f'<section id="{rid}"><h2>{rid}</h2><p>최소 세 값 {escape(r["truth"]["values"])}　정답 {r["truth"]["total"]}　하단 세 행의 합 {r["bottom"]["total"]}</p><p>'+escape("　".join(f'{a}. {v}' for a,v in zip("ABCD",r["options"])))+f'</p><img src="fresh/{inputs_by[rid]["image"]}" alt="평가 차트"><div class="cards">'+''.join(cards)+'</div></section>')
    css='body{max-width:1440px;margin:35px auto;padding:0 18px;font:16px/1.55 system-ui;color:#183639}table{border-collapse:collapse}td,th{padding:4px 9px;border:1px solid #bbcacc}img{width:600px;max-width:100%}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}article{padding:8px;background:#f0f5f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:34em;overflow:auto}section{border-top:1px solid #b2c6ca;margin-top:28px}@media(max-width:800px){.cards{display:block}}'
    (ROOT/"review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>D 226 새 차트 평가</title><style>'+css+'</style><h1>D 226 학습 후 새 차트 비교</h1><p>표 순서: 기존 ERA / 일반 ERA / E 확인 질문 ERA. 문항을 누르면 차트와 세 응답을 함께 볼 수 있다.</p><table><tr><th>차트</th><th>V0O0</th><th>V0O1</th><th>V1O0</th><th>V1O1</th></tr>'+''.join(grid)+'</table>'+''.join(details)+'</html>',encoding="utf8")

if __name__=="__main__":main()
