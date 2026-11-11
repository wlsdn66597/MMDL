"""Verify the ScienceQA intake and document eight answer-independent visual checks."""
import hashlib
import io
import json
from pathlib import Path
import sys

from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/".runtime/era_review_deps"))
import pyarrow.parquet as pq

OBS={
    "sqa-13942":("보조·힌트 충분","어두운 수중의 물고기가 보인다. 힌트가 이미 'deep sea'를 지정하고 두 보기가 밝은 햇빛 여부에서 갈리므로 A를 고를 때 사진에만 의존할 필요는 없다."),
    "sqa-571":("보조·힌트 충분","나무줄기와 녹색 숲 바닥은 보인다. 한 장의 사진으로 여름 기후와 전체 나무 종수까지 판독할 수 없고, 온대 낙엽수림이라는 힌트와 출처 해설로 B를 정한다."),
    "sqa-3364":("보조·명칭 충분","잔 속 음료 사진은 액체와 일치한다. 질문이 물질을 커피라고 알려주므로 사진 없이도 B를 고를 수 있다."),
    "sqa-2606":("보조·힌트 충분","서로 다른 입자가 보이는 표본이다. 힌트가 고정 결정구조가 없고 순물질이 아니라는 성질을 열거하여 그림 없이도 암석 A로 분류할 수 있다."),
    "sqa-19687":("그림에서 직접 확인","배의 갑판에 작은 인물들이 옆으로 빽빽하게 배열되어 있다. D의 과밀을 지지하고 개별 선실이나 승무원의 식사를 보여주지 않는다."),
    "sqa-18974":("그림에서 직접 확인","타임라인의 제헌회의는 1787년, 남북전쟁 시작은 1861년이다. 1861−1787=74로 A가 맞다. 역사 지식으로도 가능하지만 두 수가 그림에 명시된다."),
    "sqa-8097":("그림에서 직접 확인","Mermaids' Lagoon 전투 상자에서 바로 아래의 Wendy remembers her parents로 화살표가 이어져 A다. Hook을 물리치는 상자는 몇 단계 뒤에 있다."),
    "sqa-8794":("그림에서 직접 확인","왼쪽 nice에서 오른쪽 mean으로 가는 척도에서 Big Bad Wolf가 Rumpelstiltskin 오른쪽, Goldilocks는 왼쪽이다. B다.")
}


def sha(b):return hashlib.sha256(b).hexdigest()


def main():
    rows=[json.loads(s) for s in (HERE/"items.jsonl").read_text(encoding="utf8").splitlines()]
    sqa=[r for r in rows if r["source"]=="ScienceQA/validation"]
    assert len(sqa)==len({r["source_id"] for r in sqa})==72
    meta=ROOT/"reports/era_dataset_examples_20261014"
    ids=list(map(str,json.loads((meta/"scienceqa_splits.json").read_text(encoding="utf8"))["val"]))
    problems=json.loads((meta/"scienceqa_problems.json").read_text(encoding="utf8"))
    parquet=ROOT/"tmp/er_r7_sources/scienceqa_val.parquet"
    assert sha(parquet.read_bytes())=="e66b5215ce71e748ade4ee629bc14fbf86762071d355a4fcd831581cc04d72d8"
    source=pq.read_table(parquet).to_pylist()
    assert len(source)==len(ids)==4241
    for item in sqa:
        sid=item["source_id"];p=problems[sid];q=source[ids.index(sid)]
        for key in ("question","hint"):
            assert item[key]==p[key]==q[key],(sid,key)
        assert item["options"]==p["choices"]==q["choices"]
        assert item["gold"]==chr(65+p["answer"])==chr(65+q["answer"])
        assert item["reference"]["source_solution"]==p["solution"]==q["solution"]
        assert p["split"]=="val" and item["id"]=="sqa-"+sid
        b=(HERE.parent/item["image"]).read_bytes()
        assert sha(b)==item["image_sha256"]
        with Image.open(io.BytesIO(b)) as a, Image.open(io.BytesIO(q["image"]["bytes"])) as original:
            assert a.size==original.size and a.convert("RGB").tobytes()==original.convert("RGB").tobytes()
    selected=[]
    for k,n in (("natural science",4),("social science",2),("language science",2)):
        selected.extend(sorted((r for r in sqa if r["kind"]==k),key=lambda r:sha(("r7-source-audit-v1/"+r["id"]).encode()))[:n])
    assert {r["id"] for r in selected}==set(OBS)
    receipt=dict(checked_id_question_hint_choice_order_gold_solution_and_rgb=72,
                 source_validation_size=4241,source_sha256=sha(parquet.read_bytes()),items_sha256=sha((HERE/"items.jsonl").read_bytes()),
                 sample_selection="SHA256(r7-source-audit-v1/id), first 4 natural, 2 social and 2 language",
                 sample=[dict(id=r["id"],decision=OBS[r["id"]][0],note=OBS[r["id"]][1],gold=r["gold"]) for r in selected])
    (HERE/"science_source_audit.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    md=["# 독립 일반 평가의 ScienceQA 원본 검수","",
        "72문항 전체에서 validation ID, 질문, 힌트, 보기의 원래 순서, 정답, 원본 해설, 이미지의 RGB·크기를 대조했다. validation 원본 4,241개의 순서와 별도 ScienceQA 메타데이터를 함께 확인했다. [검수 기록](science_source_audit.json)","",
        "모델 출력과 무관한 고정 해시 순서로 자연과학 4개, 사회과학 2개, 언어 2개를 뽑았다. 이 8개는 각 원본 사진·도표를 열고 출처 해설과 정답을 직접 대조했다.","",
        "| 문항 | 정답 | 이미지의 역할 | 확인한 근거 |","|---|---|---|---|"]
    for r in selected:
        i=r["id"]; tag,reason=OBS[i]
        md.append(f"| [{i}](images/{i}.png) | {r['gold']} | {tag} | {reason} |")
    md+=["","이 표본에서는 네 문항이 그림에 표시된 날짜·순서·배치로 풀리며, 네 문항은 이름 또는 힌트만으로도 답을 낼 수 있다. 표본 8개로 나머지 64개의 이미지 필요도를 추정하거나, 이미지가 있는 일반 정확도를 이미지 의존성 향상으로 해석할 수 없다. 다른 사회과학 문항처럼 그림에서도 뒷받침되고 사전 지식으로도 풀리는 경우가 있다."]
    (HERE/"science_source_audit.md").write_text("\n".join(md)+"\n",encoding="utf8")
    print(72,len(selected),sha((HERE/"science_source_audit.json").read_bytes()))


if __name__=="__main__":main()
