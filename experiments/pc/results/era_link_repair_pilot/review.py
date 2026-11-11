from score import *
def main():
    gold=read(R/"holdout/gold.json");scores={(x["arm"],x["id"]):x for x in read(R/"scores.json")}
    names={"baseline":"출발 모델","statement":"평서형 검증","question":"질문형 검증"}
    preds={a:{x["id"]:x for x in map(json.loads,(R/"eval"/a/"predictions.jsonl").read_text(encoding="utf8").splitlines())} for a in ARMS}
    chart=[];changes=[]
    for it in gold:
        rid=it["id"];cards=[]
        for arm in ARMS:
            row=scores[arm,rid]
            cards.append(f'<article><h3>{names[arm]}</h3><p>정답 {"✓" if row["answer"] else "✗"} · 연결 {"✓" if row["joint"] else "✗"} · 첫 선택 {html.escape(row["init"])}</p><pre>{html.escape(preds[arm][rid]["raw"])}</pre><p>{html.escape(row["reason"])}</p></article>')
        if scores["question",rid]["repair"] or scores["question",rid]["joint"]!=scores["baseline",rid]["joint"]:changes.append(f'<a href="#{rid}">{rid}</a>')
        chart.append(f'<section id="{rid}"><h2>{rid}</h2><p>{html.escape(it["question"])}</p><p>정답 {it["letter"]} ({it["truth"]}) · '+html.escape("　".join(f"{a}. {b}" for a,b in zip("ABCD",it["options"])))+f'</p><img src="holdout/{it["image"]}" alt="평가 차트"><div class="grid">'+''.join(cards)+'</div></section>')
    css='body{max-width:1450px;margin:36px auto;padding:0 16px;font:16px/1.55 system-ui;color:#193739}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}article{padding:10px;background:#f1f5f5}section{border-top:1px solid #bdcccc;margin:20px 0}img{width:540px;max-width:100%}pre{white-space:pre-wrap;max-height:26em;overflow:auto}a{margin-right:12px}@media(max-width:750px){.grid{display:block}}'
    (R/"review.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>근거 연결 및 수정 평가</title><style>'+css+'</style><h1>근거 연결 및 수정 평가</h1><p>r은 자연 순위, read는 직접 판독, dkeep/dreplace는 명시된 초안이 있는 별도 검사다.</p><p>질문형에서 연결 판정이 달라지거나 실제로 수정된 사례　'+''.join(changes)+'</p>'+''.join(chart)+'</html>',encoding="utf8")
    with (R/"scores.csv").open("w",encoding="utf-8-sig",newline="") as f:
        keys=["id","arm","mode","letter","answer","init","E","R","A","joint","repair","draft_supported","format_ok","explicit_question","reason"]
        w=csv.DictWriter(f,fieldnames=keys,extrasaction="ignore");w.writeheader();w.writerows(scores.values())
if __name__=="__main__":main()
