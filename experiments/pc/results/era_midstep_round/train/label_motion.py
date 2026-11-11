"""Publish the sealed manual ratings from the blind motion packet after reviewing every raw answer."""
import hashlib
import json
from pathlib import Path


HERE=Path(__file__).resolve().parent
REVIEW=HERE.parent/"eval/review"
PER_ENDPOINT={"c10e95c0638132250b4e","5311c2e94246a4f96233","fa40bea41ed57ef0a92f"}
SIGNED_INSTEAD="c8b92435e6138601c438"
CONTRADICTION="578ba40365033cb964b2"


def main():
    packet=REVIEW/"motion.jsonl"
    digest=hashlib.sha256(packet.read_bytes()).hexdigest()
    assert digest.startswith("9d13ae62")
    cards=[json.loads(x) for x in packet.read_text(encoding="utf8").splitlines()]
    assert len(cards)==len({x["key"] for x in cards})==92
    assert {SIGNED_INSTEAD,CONTRADICTION,*PER_ENDPOINT}<={r["key"] for r in cards}
    labels=[]
    for card in cards:
        ref=card["reference"]
        assert ref["domain"]=="motion" and card["finish"]=="stop"
        src=ref["source"]
        xi,xm,xf=src["start"],src["turn"],src["end"]
        key=card["key"]
        er=rr=ar=ans=consistent=first=True
        corrected=None
        if key in PER_ENDPOINT:
            er=rr=ar=ans=first=False
            corrected=False
            reason=(f"The first evidence set correctly reads {xi} and {xf} but omits the required turning point {xm}. "
                    f"It uses the absolute net change 2 and selects the matching option instead of tracing both legs "
                    f"({abs(xm-xi)}+{abs(xf-xm)}={ref['value']}). No subsequent source-based correction.")
        elif key==SIGNED_INSTEAD:
            rr=ar=ans=False
            reason=(f"E reads {xi}, {xm}, {xf}. It chooses signed displacement and computes {xf}-({xi})=-2 "
                    f"even though the question asks for distance; total distance is {abs(xm-xi)}+{abs(xf-xm)}. "
                    "Its selected letter is consistent with the wrong signed value.")
        elif key==CONTRADICTION:
            consistent=False
            reason=("Positions -5, -16 and -1 are read correctly. It explicitly asserts +17 on the second leg, "
                    "then uses +15 in the distance calculation without addressing the conflicting +17. "
                    "The final application and choice C, 11+15=26 m, are correct.")
        elif ref["rule_index"]==0:
            reason=(f"Reads initial {xi} and final {xf}, uses final minus initial and obtains "
                    f"{xf}-({xi})={ref['value']} m, matching choice {ref['gold']}.")
        else:
            reason=(f"Uses all three positions {xi}, {xm}, {xf}; the two leg magnitudes "
                    f"{abs(xm-xi)} and {abs(xf-xm)} sum to {ref['value']} m, choice {ref['gold']}.")
        labels.append(dict(key=key,id=ref["id"],E=er,R=rr,A=ar,consistent=consistent,answer=ans,
                           first_E_correct=first,image_corrected=corrected,reason=reason))
    target=REVIEW/"motion_labels.jsonl"
    payload="".join(json.dumps(x,ensure_ascii=False)+"\n" for x in labels)
    if target.exists():
        assert target.read_bytes()==payload.encode()
    else:
        target.write_bytes(payload.encode("utf8"))
    print(digest,hashlib.sha256(target.read_bytes()).hexdigest())


if __name__=="__main__":
    main()
