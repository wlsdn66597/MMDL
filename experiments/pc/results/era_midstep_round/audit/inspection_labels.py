"""Transcribe the full-text, image-backed decisions on the sealed anonymous inspection packet."""
import hashlib
import json
from pathlib import Path
import re

from inspection_view import cards, FILE, ROOT

PACKET_SHA="6ccfcfa3"

# Index is the fixed SHA-key sort order printed by inspection_view.py. Every packet
# was read, including the long response ending mid-sentence after its division check.
CONFIRMED=list(map(int,("""
1 2 3 4 6 7 8 9 10 11 12 13 14 15 16 18 20 21 22 23 24 25 26 27 28 29 30
31 32 33 34 35 39 40 41 42 43 44 47 48 49 52 53 55 56 57 58 59 60 61 62
63 64 65 66 67 68 69 70 71 75 76 77 78 80 82 83 84 85 86 87 88 89 90
""").split()))
WRONG_BINDING={0,5,17,19,36,37,45,79}
BAD_RATE={38,54,72,81}
EARLY_ROUND={46,50,51,74}
TRUNCATED={73}


def main():
    digest=hashlib.sha256(FILE.read_bytes()).hexdigest()
    assert digest.startswith(PACKET_SHA)
    seq=cards()
    groups=[set(CONFIRMED),WRONG_BINDING,BAD_RATE,EARLY_ROUND,TRUNCATED]
    assert len(seq)==91==sum(map(len,groups)) and set.union(*groups)==set(range(91))
    labels=[]
    for n,card in enumerate(seq):
        ref=card["reference"]; src=ref["source"]
        a,b,n1,n2=[src[k] for k in ("a","b","n1","n2")]
        raw=card["raw"]
        answer=card["finish"]=="stop"
        if answer:
            m=re.search(r"(?i)Final answer\s*:\s*([A-D])[\s*]*$",raw.strip().replace("*",""))
            assert m and m[1]==ref["gold"]
        label=dict(key=card["key"],id=card["id"],E=True,R=True,A=True,consistent=True,
                   answer=answer,first_E_correct=True,image_corrected=None)
        if n in WRONG_BINDING:
            label.update(E=False,consistent=False,first_E_correct=False,image_corrected=False)
            if ref["scene"]=="inspection-elara":
                text=f'E states the {n1} and {n2} inspected units passed. A uses the actual passes {a},{b} and correct pooled denominator without withdrawing or explaining that mistaken binding.'
            else:
                text=f'E describes {n1} East units as passing and {n2} as its inspected count. A uses the correct 18 and 37 without explicitly resolving the original mismatch.'
        elif n in BAD_RATE:
            label.update(A=False,consistent=False)
            text="Correct raw counts, but 61/73 is 83.5616% (83.6% to one decimal); A relies on E's incorrect 83.5% intermediate when averaging. Its 72.2% matches the listed answer by coincidence."
        elif n in EARLY_ROUND:
            label.update(A=False,consistent=False)
            if n==46:
                text="A incorrectly rounds its own exact displayed 63.55% to 63.5%. The unrounded source ratios really average to 63.52%, but that derivation is missing."
            else:
                text="A averages prematurely rounded percentages to 63.55%, obtains 63.6%, then selects the 63.5% option without recomputing from the two original fractions."
        elif n in TRUNCATED:
            text="Reads all four cells and eventually corrects the division to 38/59 ≈ 0.64407 = 64.4% before length termination. This is arithmetic correction, not correction of E; the final answer is uncommitted."
            assert not answer
        elif ref["rule_index"]==0:
            text=f"Correct batch bindings lead to pooled passes ({a}+{b})/({n1}+{n2}) and the stated {ref['value']} choice."
            if n==11:
                text+=" Its extra quotient digits (0.64399 versus 0.64407) do not change the requested one-decimal result."
        else:
            text=f"Uses both batch fractions {a}/{n1} and {b}/{n2} and averages their percentages equally, obtaining {ref['value']} and its matching option."
        label["reason"]=text;labels.append(label)
    out=ROOT/"eval/review/inspection_labels.jsonl"
    result="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in labels).encode()
    if out.exists() and out.read_bytes()!=result:
        raise ValueError("Blinded labels already frozen")
    out.write_bytes(result)
    print("source",digest,"labels",hashlib.sha256(result).hexdigest(),"rows",len(labels))


if __name__=="__main__":
    main()
