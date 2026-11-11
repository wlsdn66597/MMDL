"""Combine the four exhaustive review shards and explicit second-pass decisions."""
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAMES = {"E":"근거 선택·판독", "R":"규칙", "A_INV":"A에서 관계 반전",
         "A_OTHER":"A의 다른 적용 오류", "FINAL":"최종 답 대응",
         "BORDER":"A 반전의 경계 사례", "UNCERTAIN":"특정 보류"}


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x]


def main():
    cards = lines(HERE/"cards.jsonl")
    shards = sorted(HERE.glob("review_???_???.jsonl"))
    primary = [r for p in shards for r in lines(p)]
    assert len(shards) == 4 and len(primary) == 148
    assert [r["idx"] for r in primary] == list(range(1,149))
    edits = json.loads((HERE/"adjudications.json").read_text(encoding="utf8"))
    corrected = []
    table = []
    for card,row in zip(cards,primary):
        i=card["idx"]
        assert card["id"] == row["id"] and i == row["idx"]
        patch=edits.get(str(i),{})
        result=row | patch
        assert result["code"] in NAMES and result["e"] in {"ok","unknown","wrong"} and result["r"] in {"ok","unknown","wrong"} and result["witness"].strip()
        if result["code"] in {"A_INV","A_OTHER","FINAL"}:
            assert result["e"] == result["r"] == "ok"
        files=[Path(os.path.relpath(ROOT/f,HERE)).as_posix() for f in card["images"]]
        assert all((HERE/f).is_file() for f in files)
        corrected.append(result | dict(primary_code=row["code"],gold=card["gold"],parsed=card["choice"],grading_method=card["method"],images=files))
        links=" ".join(f"[그림{j}]({f})" for j,f in enumerate(files,1))
        table.append(f"| {i} | {card['id'].removeprefix('validation_')} | {NAMES[result['code']]} | {result['e']} / {result['r']} | {result['witness'].replace('|','／')} {links} |")
    assert all(str(int(k))==k and 1<=int(k)<=148 for k in edits)
    target=HERE/"adjudicated.jsonl"
    target.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in corrected),encoding="utf8")
    report = ["# MMMU 기존 정답 손실 148개 판정 원장","",
              "문항의 공식 정오를 유지하고 방향 반전 ERA 응답의 첫 실질 오류를 기록했다. E/R의 ok·wrong·unknown은 각각 부합·오류 또는 불충분·확인 불가를 뜻한다. 한 문항에 후속 오류가 여럿 있어도 빈도에는 한 번만 포함했다. [질문·보기·ERA 원문](cards.jsonl) · [정답과 실제 추출 답을 포함한 원장](adjudicated.jsonl) · [교차검수 결정](adjudications.json)","",
              "| 순번 | 문항 | 첫 오류 분류 | E / R | 검수 근거 |","|---:|---|---|---|---|",*table]
    (HERE/"item_review.md").write_text("\n".join(report)+"\n",encoding="utf8")
    counts=dict(Counter(r["code"] for r in corrected))
    ids={c:[r["id"] for r in corrected if r["code"]==c] for c in NAMES}
    inputs=[*shards,HERE/"cards.jsonl",HERE/"adjudications.json",target]
    out=dict(n=len(corrected),counts=counts,ids=ids,sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (HERE/"summary.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(counts)


if __name__ == "__main__":
    main()
