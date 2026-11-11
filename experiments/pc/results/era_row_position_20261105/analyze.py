"""Score frozen predictions against both hypotheses, with separately reviewed E and A."""
import csv
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(path): return json.loads(path.read_text(encoding="utf8"))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def jsonl(path): return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines()]
def dump(path, obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
def esc(x): return html.escape(str(x))

def analyze():
    items=read(ROOT/"items.json")
    frozen=read(ROOT/"design_receipt.json")
    inputs={r["id"]:r for r in read(ROOT/"inputs.json")}
    rows=jsonl(ROOT/"run/predictions.jsonl")
    responses={r["id"]:r for r in rows}
    assert set(responses)==set(inputs) and len(rows)==len(responses)==48
    man=read(ROOT/"run/manifest.json")
    assert man["inputs_sha256"]==frozen["inputs_sha256"]==sha(ROOT/"inputs.json")
    assert sha(ROOT/"items.json")==frozen["items_sha256"]
    assert man["eval_script_sha256"]==sha(ROOT/"run_eval.py")
    prev=ROOT.parent/"era_visual_pairs_20261105"
    old_man=read(prev/"run/manifest.json")
    assert all(man[k]==old_man[k] for k in ("weights","config_sha256","runner_frozen_prompt_sha256","model"))
    old_rows={r["id"]:r for r in jsonl(prev/"run/predictions.jsonl")}
    for cur,past in [("F01_v0o0","P08_control"),("F01_v1o0","P08_changed")]:
        assert all(responses[cur][k]==old_rows[past][k] for k in ("raw","image_sha256","prompt","finish_reason"))
    with (ROOT/"content_review.tsv").open(encoding="utf8") as f:
        labels={r["id"]:r for r in csv.DictReader(f,delimiter="\t")}
    assert set(labels)==set(responses)
    grouped={"aligned":Counter(),"conflict":Counter()}
    metrics=Counter(); charts=[]; all_rows=[]; blocks=[]
    for item in items:
        data={}
        assert len({responses[c["id"]]["prompt"] for c in item["cells"].values()})==1
        for cond,cell in item["cells"].items():
            rid=cell["id"]; resp=responses[rid]; inp=inputs[rid]
            assert inp["image_sha256"]==resp["image_sha256"]==sha(ROOT/inp["image"])
            assert resp["finish_reason"]=="stop"
            found=re.findall(r"(?i)Final answer:\s*([A-D])\b",resp["raw"])
            assert len(set(found))==1,(rid,found)
            pred=found[-1]; selected=inp["options"]["ABCD".index(pred)]
            correct=pred==cell["truth"]["letter"]; bottom=pred==cell["bottom"]["letter"]
            lab=labels[rid]; e=lab["E"]; assert e in {"both","rank","bottom","other","unclear"}
            assert (e!="both" or cell["aligned"]) and (e not in {"rank","bottom"} or not cell["aligned"])
            ev_correct=e in {"both","rank"};ev_bottom=e in {"both","bottom"}
            rec=dict(id=rid,aligned=cell["aligned"],truth=cell["truth"],bottom=cell["bottom"],answer=selected,
                     letter=pred,correct=correct,bottom_prediction=bottom,E=e,ev_correct=ev_correct,ev_bottom=ev_bottom,
                     rule_valid=lab["R"]=="1",arithmetic_on_chosen_operands=lab["A"]=="1",note=lab["note"])
            data[cond]=rec;all_rows.append(rec)
            cnt=grouped["aligned" if cell["aligned"] else "conflict"]
            for k,v in dict(n=True,correct=correct,bottom=bottom,e_correct=ev_correct,e_bottom=ev_bottom,
                            e_and_answer_bottom=ev_bottom and bottom,e_and_answer_rank=ev_correct and correct,
                            rule=rec["rule_valid"],arithmetic=rec["arithmetic_on_chosen_operands"]).items(): cnt[k]+=int(v)
        diff=0
        for v in (0,1):
            aligned=data[f"v{v}o{v}"];conflict=data[f"v{v}o{1-v}"]
            assert aligned["truth"]==conflict["truth"]
            key=("correct" if aligned["correct"] else "wrong")+"_to_"+("correct" if conflict["correct"] else "wrong")
            metrics[key]+=1
            metrics["changed_final"]+=int(aligned["letter"]!=conflict["letter"])
            diff+=int(conflict["ev_bottom"] and conflict["bottom_prediction"])
            for cat in ("correct","ev_correct","bottom_prediction","ev_bottom"):
                metrics[f"v{v}_conflict_{cat}"]+=int(conflict[cat])
        charts.append(dict(id=item["chart"],both_E_and_answer_bottom_in_n_conflicts=diff,
                           four_correct=all(x["correct"] for x in data.values())))
        cards=[]
        for cond in ("v0o0","v0o1","v1o0","v1o1"):
            rec=data[cond];ri=rec["id"];src=inputs[ri]["image"]
            alignment="일치" if rec["aligned"] else "충돌"
            cards.append(f'<article><h3>{esc(cond.upper())}　·　하단 세 행과 정답 {alignment}</h3><a href="{esc(src)}"><img src="{esc(src)}" alt="{esc(ri)} 차트"></a><p>정답 {esc(rec["truth"]["total"])}　·　하단 합 {esc(rec["bottom"]["total"])}　·　모델 {esc(rec["answer"])}</p><pre>{esc(responses[ri]["raw"])}</pre><p>{esc(rec["note"])}</p></article>')
        blocks.append(f'<section id="{item["chart"]}"><h2>{item["chart"]}　{esc(item["changed"])} {esc(item["old"])} → {esc(item["new"])}</h2><span>'+esc("　".join(f"{k}. {v}" for k,v in zip("ABCD",item["options"])))+'</span><div class="grid">'+''.join(cards)+'</div></section>')
    summary=dict(charts=12,conditions=48,groups={k:dict(v) for k,v in grouped.items()},same_values_pairs=dict(metrics),
                 conflicts_bottom_counts_per_chart=dict(Counter(x["both_E_and_answer_bottom_in_n_conflicts"] for x in charts)),
                 charts_all_four_correct=sum(x["four_correct"] for x in charts),prediction_sha256=sha(ROOT/"run/predictions.jsonl"),
                 p08_replays_byte_identical_text=2)
    dump(ROOT/"summary.json",summary); dump(ROOT/"reviews.json",all_rows)
    css="body{max-width:1350px;margin:35px auto;padding:0 18px;font:16px/1.55 system-ui;color:#17343a}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid article{min-width:0;background:#f4f7f7;padding:12px}img{width:100%}pre{white-space:pre-wrap}nav a{margin-right:15px}@media(max-width:760px){.grid{grid-template-columns:1fr}}"
    a,b=grouped["aligned"],grouped["conflict"]
    intro=f'<h1>차트 행 위치와 최소 세 값의 선택</h1><p>12개 차트에서 최소 세 값과 하단 세 행이 일치할 때 {a["correct"]}/24 정답, 충돌할 때 {b["correct"]}/24 정답. 충돌 시 모델이 하단 세 수치를 E로 제시하고 그 합을 선택한 사례는 {b["e_and_answer_bottom"]}/24.</p><p>V0은 원 자료, V1은 수치 하나 변경. O0은 기존 행 순서, O1은 경계의 두 행을 바꾼 순서.</p>'
    nav='<nav>'+''.join(f'<a href="#{it["chart"]}">{it["chart"]}</a>' for it in items)+'</nav>'
    (ROOT/"review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ERA 행 위치 평가</title><style>'+css+'</style>'+intro+nav+''.join(blocks)+'</html>',encoding="utf8")

if __name__=="__main__": analyze()
