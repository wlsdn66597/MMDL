"""Match responses by ID; check the exact prompts and score all treatments against the frozen gold."""
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/"era_row_position_20261105"
ARMS=("generic","boundary")
NAMES={"baseline":"기존 무지시","generic":"일반 재확인","boundary":"경계값 비교"}

def read(path): return json.loads(path.read_text(encoding="utf-8"))
def load_rows(path): return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
def by_id(rows):
    indexed={r["id"]:r for r in rows}
    assert len(indexed)==len(rows)==48
    return indexed
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def score(row,options,truth,bottom):
    raw=row["raw"].strip()
    fields=re.findall(r"(?im)Final answer:[ \t]*([^\r\n]*)",raw)
    marks=[]
    for field in fields:
        marker=re.match(r"\s*\**\s*([A-D])\b",field,re.I)
        marks.append(marker[1].upper() if marker else "unmapped")
    selected=marks[-1] if marks else "unmapped"
    ok=row["finish_reason"]=="stop" and selected==truth["letter"]
    return dict(letter=selected,value=options["ABCD".index(selected)] if selected in list("ABCD") else "미선택",
                correct=ok,answer_markers=marks,unmapped_last=selected=="unmapped",
                conflicting_markers=len(set(marks)-{"unmapped"})>1,era_cycles=len(re.findall(r"(?im)^\s*Evidence:",raw)),
                bottom=selected==bottom["letter"],
                final_line=bool(re.fullmatch(r"Final answer:\s*[A-D]",raw.splitlines()[-1],re.I)),
                tokens=row["output_tokens"])

def transition(before,after):
    if before: return "kept" if after else "lost"
    return "recovered" if after else "still_wrong"

def main():
    instructions=read(ROOT/"instructions.json")
    base_manifest=read(PREV/"run/manifest.json")
    inputs=by_id(read(PREV/"inputs.json"))
    predictions={"baseline":by_id(load_rows(PREV/"run/predictions.jsonl"))}
    assert digest(PREV/"run/predictions.jsonl")=="f6c7d4cfbf6384f18a5ca82fcfa62bec281c5d80d6ddd93196b7a6c8848f8a0e"
    old_reviews=by_id(read(PREV/"reviews.json"))
    receipt=read(ROOT/"server_hashes.json")
    for arm in ARMS:
        manifest=read(ROOT/arm/"manifest.json")
        assert manifest==base_manifest
        meta=read(ROOT/arm/"arm.json")
        assert meta["arm"]==arm and meta["instruction"]==instructions[arm] and meta["wrapper_sha256"]==digest(ROOT/"run_arm.py")
        assert meta["instructions_sha256"]==digest(ROOT/"instructions.json")
        assert digest(ROOT/arm/"predictions.jsonl")==receipt[arm]["predictions"]
        assert digest(ROOT/arm/"manifest.json")==receipt[arm]["manifest"]
        assert digest(ROOT/arm/"arm.json")==receipt[arm]["arm"]
        pred=by_id(load_rows(ROOT/arm/"predictions.jsonl"))
        assert list(pred)==list(predictions["baseline"])==list(inputs)
        for rid,row in pred.items():
            template=predictions["baseline"][rid]["prompt"].rsplit("\n",1)
            assert row["prompt"]=="\n".join((template[0],instructions[arm],template[1]))
            assert row["image_sha256"]==inputs[rid]["image_sha256"]==predictions["baseline"][rid]["image_sha256"]
        predictions[arm]=pred
    items=read(PREV/"items.json")
    results=[]
    for item in items:
        for cond,cell in item["cells"].items():
            rid=cell["id"]
            res=dict(id=rid,chart=item["chart"],condition=cond,aligned=cell["aligned"],truth=cell["truth"],bottom=cell["bottom"])
            for arm in predictions:
                res[arm]=score(predictions[arm][rid],item["options"],cell["truth"],cell["bottom"])
            assert res["baseline"]["correct"]==old_reviews[rid]["correct"]
            for arm in ARMS:
                res[arm]["transition"]=transition(res["baseline"]["correct"],res[arm]["correct"])
            results.append(res)
    assert sum(r["baseline"]["correct"] for r in results)==38
    summary={}
    for arm in predictions:
        out={"correct":sum(r[arm]["correct"] for r in results),"aligned_correct":sum(r[arm]["correct"] for r in results if r["aligned"]),
             "conflict_correct":sum(r[arm]["correct"] for r in results if not r["aligned"]),
             "conflict_bottom":sum(r[arm]["bottom"] for r in results if not r["aligned"]),
             "final_line":sum(r[arm]["final_line"] for r in results),"normal_stop":sum(row["finish_reason"]=="stop" for row in predictions[arm].values()),
             "mean_tokens":round(statistics.mean(r[arm]["tokens"] for r in results),1),
             "median_tokens":statistics.median(r[arm]["tokens"] for r in results),
             "max_tokens":max(r[arm]["tokens"] for r in results),
             "first_marker_correct_ignoring_later":sum(bool(r[arm]["answer_markers"]) and r[arm]["answer_markers"][0]==r["truth"]["letter"] for r in results),
             "repeated_era":sum(r[arm]["era_cycles"]>1 for r in results),
             "multiple_final_letters_disagree":sum(r[arm]["conflicting_markers"] for r in results),
             "unmapped_last":sum(r[arm]["unmapped_last"] for r in results)}
        if arm in ARMS:
            for k in ("recovered","lost","still_wrong","kept"):
                out[k]=[r["id"] for r in results if r[arm]["transition"]==k]
            assert len(out["recovered"])+len(out["still_wrong"])==10 and len(out["kept"])+len(out["lost"])==38
        summary[arm]=out
    summary["generic_to_boundary"]=dict(Counter(transition(r["generic"]["correct"],r["boundary"]["correct"]) for r in results))
    for group in ("recovered","lost"):
        summary[group+"_overlap"]=sorted(set(summary["generic"][group])&set(summary["boundary"][group]))
    save(ROOT/"summary.json",summary);save(ROOT/"comparisons.json",results)
    with (ROOT/"comparisons.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(("id","aligned","gold_answer","bottom_sum","baseline","generic","boundary","generic_transition","boundary_transition"))
        for r in results:
            w.writerow((r["id"],r["aligned"],r["truth"]["total"],r["bottom"]["total"],r["baseline"]["value"],r["generic"]["value"],r["boundary"]["value"],r["generic"]["transition"],r["boundary"]["transition"]))
    sections=[]
    for r in results:
        rid=r["id"];inp=inputs[rid]
        cards=[]
        for arm in predictions:
            pred=predictions[arm][rid];correct="정답" if r[arm]["correct"] else "오답"
            cards.append(f'<article><h3>{NAMES[arm]} · {correct} ({html.escape(r[arm]["value"])})</h3><pre>{html.escape(pred["raw"])}</pre></article>')
        sections.append(f'<section id="{rid}"><h2>{rid}</h2><p>정답 {r["truth"]["total"]}　하단 세 행의 합 {r["bottom"]["total"]}　|　'+html.escape("　".join(f"{a}. {v}" for a,v in zip("ABCD",inp["options"])))+'</p><img src="../era_row_position_20261105/'+html.escape(inp["image"])+'" alt="평가 차트"><div class="cards">'+''.join(cards)+'</div></section>')
    changed=[r["id"] for r in results if any(r[arm]["transition"] in ("lost","recovered") for arm in ARMS)]
    nav="　".join(f'<a href="#{rid}">{rid}</a>' for rid in changed)
    transitions={"lost":"정답→오답","recovered":"오답→정답","kept":"정답 유지","still_wrong":"오답 유지"}
    grids=[]
    for arm in ARMS:
        lines=[]
        for chart in items:
            cells=[r for r in results if r["chart"]==chart["chart"]]
            cols="".join(f'<td class="{r[arm]["transition"]}"><a href="#{r["id"]}">{transitions[r[arm]["transition"]]}</a></td>' for r in cells)
            lines.append(f'<tr><th>{chart["chart"]}</th>{cols}</tr>')
        grids.append(f'<div><h3>{NAMES[arm]}</h3><table><thead><tr><th>차트</th><th>V0O0</th><th>V0O1</th><th>V1O0</th><th>V1O1</th></tr></thead><tbody>'+''.join(lines)+'</tbody></table></div>')
    css="body{max-width:1470px;margin:36px auto;padding:0 18px;font:16px/1.55 system-ui;color:#163335}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.grids{display:flex;flex-wrap:wrap;gap:25px}table{border-collapse:collapse;font-size:.9rem}td,th{padding:4px 8px;border:1px solid #c4d5d6}td a{color:inherit}.lost{background:#f6d9d7}.recovered{background:#c3ecd6}.kept{background:#edf3f3}.still_wrong{background:#f2e5d4}article{background:#f1f6f7;padding:0 12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:35em;overflow-y:auto}img{width:560px;max-width:100%}section{border-top:1px solid #c4d5d6;margin:26px 0}@media(max-width:850px){.cards{display:block}}"
    desc=f'무지시 38/48, 일반 재확인 {summary["generic"]["correct"]}/48, 경계값 비교 {summary["boundary"]["correct"]}/48.'
    (ROOT/"review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>지시별 ERA 응답 대조</title><style>'+css+'</style><h1>검증 지시별 ERA 응답</h1><p>'+desc+'</p><p>V0는 원본 수치, V1은 한 값 변경. O0는 기존 행 순서, O1은 경쟁하는 두 행의 순서를 교환한 조건.</p><div class="grids">'+''.join(grids)+'</div><nav>정오가 바뀐 문항　'+nav+'</nav>'+''.join(sections)+'</html>',encoding="utf-8")

if __name__=="__main__":main()
