"""Independently solve 48 choices and check that only 24 Application fields differ."""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import sys
import json

from PIL import Image
from build import HERE, ROOT, sha, render, js
from tokenizers import Tokenizer


def answer(row, source, op):
    data=source["source"]["rows"]
    if op in ("net","travel"):
        vals=[v for _,v in data]
        return str(vals[-1]-vals[0] if op=="net" else sum(abs(b-a) for a,b in zip(vals,vals[1:])))
    if op in ("equal","pooled"):
        (_,v,p),(_,w,q)=data
        return str(Fraction(p+q,2) if op=="equal" else Fraction(v*p+w*q,v+w))
    if op in ("count","rate"):
        values=[x if op=="count" else Fraction(x,y) for (_,x,y) in data]
        winner=max(values);assert values.count(winner)==1
        return data[values.index(winner)][0]
    if op=="mark":
        hits=[mark for name,mark in data if name.lower() in row["question"].lower()]
    elif op=="direct_number":
        hits=[str(num) for name,num,_ in data if name in row["question"]]
    elif op=="restricted":
        hits=[name for name,_,tag in data if f"labeled {tag} " in row["question"]]
    else:
        raise ValueError(op)
    assert len(hits)==1
    return hits[0]


def main():
    groups={s["group"]:s for s in json.loads((HERE/"sources.json").read_text(encoding="utf8"))}
    rows=[json.loads(line) for line in (HERE/"rows.jsonl").read_text(encoding="utf8").splitlines()]
    by_group=defaultdict(list);positions=Counter();tokens=defaultdict(Counter)
    tok=Tokenizer.from_file(str(ROOT/".runtime/qwen_tokenizer/tokenizer.json"))
    changed=[];image_digests={};rgb_digests=set()
    for row in rows:
        s=groups[row["group"]]
        assert row["kind"]==s["kind"] and row["family"]==s["family"]
        pair=by_group[row["group"]];ref=s["oracle"][len(pair)];pair.append(row)
        path=HERE.parent/row["image"]
        assert sha(path)==row["image_sha256"]==s["image_sha256"]
        if path not in image_digests:
            import hashlib
            with Image.open(path) as im:
                assert list(im.size)==s["image_size"]
                rgb_digests.add(hashlib.sha256(im.convert("RGB").tobytes()).hexdigest())
            image_digests[path]=sha(path)
        truth=answer(row,s,ref["op"])
        assert len(row["options"])==len(set(row["options"]))==4
        assert truth==row["options"][ord(row["gold"])-65]==ref["answer"]
        usual,enhanced=row["ordinary"],row["enhanced"]
        assert set(usual)==set(enhanced)=={"e","r","a"}
        assert usual["e"]==enhanced["e"] and usual["r"]==enhanced["r"]
        changed_a=usual["a"]!=enhanced["a"]
        assert changed_a==(s["kind"]=="A")
        if changed_a: changed.append(row["id"])
        positions[row["gold"]]+=1
        for arm in ["ordinary","enhanced"]:
            n=len(tok.encode(render(row[arm],row["gold"]),add_special_tokens=False).ids)+1
            tokens["new48"][arm]+=n
            if changed_a:
                tokens["changed24"][arm]+=n;tokens[s["family"]][arm]+=n
    for pair in by_group.values():
        a,b=pair;assert a["image"]==b["image"] and a["question"]!=b["question"]
        if a["kind"]=="A":assert a["options"]==b["options"] and a["gold"]!=b["gold"]
    assert len(groups)==len(by_group)==len(image_digests)==len(rgb_digests)==24
    assert len(rows)==len({r["id"] for r in rows})==len({r["question"] for r in rows})==48
    assert sorted(positions.values())==[12]*4 and len(changed)==24
    diff=tokens["changed24"]["enhanced"]-tokens["changed24"]["ordinary"]
    assert abs(diff)<=0.02*tokens["changed24"]["ordinary"]
    assert Counter(s["kind"] for s in groups.values())=={"A":12,"SYMBOL":6,"GENERAL":6}
    output=dict(n=48,groups=24,kinds={"A":12,"SYMBOL":6,"GENERAL":6},changed=changed,
                unchanged_new_ids=[r["id"] for r in rows if r["id"] not in changed],
                outside_A_identical=True,evidence_and_rules_fixed=True,solver_answer_fixed=True,
                answers_recomputed=48,positions=dict(positions),distinct_rgb_images=len(rgb_digests),
                rows_sha256=sha(HERE/"rows.jsonl"),sources_sha256=sha(HERE/"sources.json"),
                tokenizer_sha256=sha(ROOT/".runtime/qwen_tokenizer/tokenizer.json"),
                token_estimates={name:dict(c) for name,c in tokens.items()},
                changed_token_relative_delta=diff/tokens["changed24"]["ordinary"])
    (HERE/"receipt.json").write_text(js(output),encoding="utf8")
    print(output["rows_sha256"],output["token_estimates"]["changed24"],output["changed_token_relative_delta"])


if __name__=="__main__":
    main()
