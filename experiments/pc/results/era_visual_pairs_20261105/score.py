"""Audit the 30 collected responses; content judgments are recorded after full-text review."""
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUN=ROOT/"run"
ORDER=("original","control","changed")
NOTES={
    "P01":"모리셔스 값과 키프로스 값을 더한다.",
    "P02":"스페인과 캐나다의 비율을 합산한다.",
    "P03":"Incidence에서 Prevalence를 뺀다.",
    "P04":"최댓값과 최솟값의 차를 구한다. 변형에서 최댓값이 다른 항목으로 바뀐다.",
    "P05":"24시간과 48시간 두 항목만 평균한다.",
    "P06":"50–69세 행을 판독한다.",
    "P07":"가장 긴 막대의 값과 나라를 찾는다.",
    "P08":"전체에서 작은 세 값을 다시 골라 합산한다.",
    "P09":"작은 두 값의 합을 2로 나눈다.",
    "P10":"저소득 경제권과 사하라 이남 아프리카 두 항목을 더한다.",
}
SPECIAL={
    "P04_changed":"실제 새 최댓값 1652와 최솟값 61을 골라 1591을 계산했다. E에도 계산을 먼저 썼다.",
    "P08_changed":"변경된 Suriname 4.7을 읽었지만 Armenia 4.5를 누락하고 4.7을 세 번째 최솟값으로 골랐다. 0.2+0.3+4.7=5.2는 조건부로 맞으나 올바른 합은 0.2+0.3+4.5=5다.",
    "P09_control":"식 (7+19)/2=13과 답 A가 맞다. 계산은 Rule에만 있고 Application은 ‘Final answer: A’만 담았다. 의미상 정답, 엄격한 단락 연결 및 독립 최종 줄 실패.",
    "P09_changed":"Rule에 식과 계산이 있고 Application에서 14를 D로 연결한다.",
    "P10_original":"수치·식·답 C가 맞다. 최종 문자가 Application 문장 끝에 있어 독립된 마지막 답 줄 기준은 실패.",
}
def read(path): return json.loads(path.read_text(encoding="utf8"))
def write(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def esc(t): return html.escape(str(t))
def image_box(title,src): return f'<div><h3>{esc(title)}</h3><a href="{esc(src)}"><img src="{esc(src)}" alt="{esc(title)}"></a></div>'

def main():
    pairs=read(ROOT/"pairs.json"); inputs={r["id"]:r for r in read(ROOT/"inputs.json")}
    rows=[json.loads(line) for line in (RUN/"predictions.jsonl").read_text(encoding="utf8").splitlines()]
    assert len(rows)==len({r["id"] for r in rows})==30 and set(inputs)=={r["id"] for r in rows}
    assert sha(RUN/"predictions.jsonl")=="f069fa3e74ce7fad59baf0ccfb9f7c4f184e99de104337871c053926d0f6861b"
    manifest=read(RUN/"manifest.json"); prior=read(ROOT.parent/"mmmu900_epochs_20261031"/"plan.json")["exports"]["3"]["hashes"]
    assert len(manifest["weights"])==3 and all(prior[n]==v for n,v in manifest["weights"].items())
    assert manifest["inputs_sha256"]==sha(ROOT/"inputs.json") and manifest["config_sha256"]==sha(ROOT/"eval_config.json") and manifest["eval_script_sha256"]==sha(ROOT/"run_eval.py")
    by_id={r["id"]:r for r in rows}
    reviews=[]; stats={cond:Counter() for cond in ORDER}; paired=Counter()
    panels=[]
    for p in pairs:
        pid=p["pair"]; group={}
        assert len({by_id[f"{pid}_{k}"]["prompt"] for k in ORDER})==1
        for cond in ORDER:
            id_=pid+"_"+cond; pred=by_id[id_]; gold=p["gold"][cond]; raw=pred["raw"].strip()
            assert pred["image_sha256"]==inputs[id_]["image_sha256"]==sha(ROOT/inputs[id_]["image"])
            assert pred["finish_reason"]=="stop"
            marker=re.findall(r"Final answer:\s*([A-D])\b",raw,re.I)
            assert len(marker)==1
            selected=marker[0].upper()
            strict=bool(re.fullmatch(r"Final answer:\s*[A-D]",raw.splitlines()[-1],re.I))
            headings=re.findall(r"(?m)^(Evidence|Rule|Application):",raw)
            assert headings==["Evidence","Rule","Application"]
            correct=selected==gold["letter"]
            # Full-text semantic review, independent of numerical answer extraction.
            e_ok=id_!="P08_changed"
            r_ok=True
            a_ok=id_ not in {"P08_changed","P09_control"}
            rec={"id":id_,"condition":cond,"gold":gold,"selected":selected,"answer_semantic":correct,
                 "standalone_final":strict,"answer_strict":strict and correct,"format":strict,
                 "E":e_ok,"R":r_ok,"A_in_section":a_ok,"application_anywhere":id_!="P08_changed",
                 "strict_complete":e_ok and r_ok and a_ok and correct and strict,
                 "note":SPECIAL.get(id_,NOTES[pid]+" 근거·규칙·계산·최종 선택이 맞다.")}
            reviews.append(rec); group[cond]=rec
            for key in ("answer_semantic","answer_strict","format","E","R","A_in_section","application_anywhere","strict_complete"):
                stats[cond][key]+=int(rec[key])
        for key in ("answer_semantic","answer_strict","format","E","R","A_in_section","application_anywhere","strict_complete"):
            paired[key]+=int(all(group[k][key] for k in ("control","changed")))
        figures=''.join(image_box(t,inputs[pid+"_"+k]["image"]) for k,t in zip(ORDER,("공개 원본","같은 수치로 다시 그린 차트","한 값 변경")))
        detail=[]
        for k,t in zip(ORDER,("공개 원본","대조 차트","수치 변경 차트")):
            rec=group[k]; raw=by_id[rec["id"]]["raw"]
            detail.append(f'<h4>{t} · 정답 {esc(rec["gold"]["letter"])} ({esc(rec["gold"]["answer"])}) / 응답 {esc(rec["selected"])}</h4><p>{esc(rec["note"])}</p><pre>{esc(raw)}</pre>')
        panels.append(f'<section id="{pid}"><h2>{pid}　{esc(p["source_question"])}</h2><p>{esc(p["changed_cell"])}　{esc(p["original_value"])} → {esc(p["changed_value"])}　|　정답 수치 {esc(group["control"]["gold"]["answer"])} → {esc(group["changed"]["gold"]["answer"])}</p><p>'+esc("　".join(f"{x}. {y}" for x,y in zip("ABCD",p["options"])))+'</p><div class="fig">'+figures+'</div>'+''.join(detail)+'</section>')
    write(ROOT/"reviews.json",reviews)
    summary={"items":30,"pairs":10,"conditions":{k:dict(v) for k,v in stats.items()},"both_control_and_changed":dict(paired),
             "inference_host":"AIX408-40","prediction_sha256":sha(RUN/"predictions.jsonl"),"model_shard_hashes_verified":True}
    write(ROOT/"summary.json",summary)
    css='body{max-width:1250px;margin:40px auto;padding:0 20px;font:16px/1.55 system-ui;color:#18343a}a{color:#096479}section{border-top:1px solid #b2c6ca;margin:24px 0}.fig{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.fig img{width:100%;border:1px solid #cedadd}pre{white-space:pre-wrap;background:#f2f5f6;padding:12px}h4{margin-bottom:0}h3{font-size:16px}nav a{margin-right:14px}@media(max-width:750px){.fig{grid-template-columns:1fr}}'
    intro=f'<h1>이미지 근거 변경 평가 · ERA 3 epoch</h1><p>대조·변형 양쪽의 선택이 맞은 쌍은 {paired["answer_semantic"]}/10, 근거·규칙·적용 및 독립 최종 답 줄까지 모두 충족한 쌍은 {paired["strict_complete"]}/10이다. 세 조건의 전체 응답을 확인할 수 있다.</p>'
    nav='<nav>'+''.join(f'<a href="#{p["pair"]}">{p["pair"]}</a>' for p in pairs)+'</nav>'
    (ROOT/"review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ERA 이미지 근거 변경 평가</title><style>'+css+'</style>'+intro+nav+''.join(panels)+'</html>',encoding="utf8")

if __name__=="__main__": main()
