"""Apply the predeclared missing-evidence rule to all option-only answers, blinded."""
import json
from pathlib import Path
import re

REVIEW=Path(__file__).resolve().parent/"review"


def normal(s):
    return " ".join(s.casefold().strip().strip(" .").split())


def only_options(raw,opts):
    allowed={normal(v) for v in opts}
    result=[]
    for line in raw.strip().splitlines():
        v=line.strip().replace("*","").replace("`","")
        if not v:continue
        v=re.sub(r"(?i)^(?:(?:final\s+)?answer)\s*:\s*","",v)
        m=re.fullmatch(r"(?i)(?:option\s+|choice\s+)?\(?([a-h])\)?(?:[.)\s:]+(.*))?",v)
        if m and ord(m[1].upper())<65+len(opts) and (not m[2] or normal(m[2]) in allowed):
            result.append(v);continue
        if normal(v) in allowed:result.append(v);continue
        return False
    return bool(result)


def main():
    rows=list(map(json.loads,(REVIEW/"packet.jsonl").read_text(encoding="utf8").splitlines()))
    covered=[]
    for card in rows:
        if only_options(card["raw"],card["options"]):
            covered.append(dict(key=card["key"],id=card["id"],fields=dict(e=False,r=False,a=False,first=None,corrected=None),
                                reason="응답에는 보기 글자 또는 보기 내용만 있고 근거 판독·규칙·대입이 없다.",raw=card["raw"]))
    buf="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in covered).encode()
    path=REVIEW/"answer_only.jsonl"
    if path.exists():assert path.read_bytes()==buf
    else:path.write_bytes(buf)
    print(len(covered))


if __name__=="__main__":main()
