"""Encode the independent image/transcript judgments and bind every kept excerpt to the exact stored answer."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent/"data"

# The graph labels are cropped in the first two pie charts. The source table for the third
# disagrees with its displayed decimal readings (the teacher independently notes the mismatch).
REJECT = {
    "chart-two_col_102574": "Seafood preparation has no displayed percentage and a second slice also has an ellipsis; the teacher guesses and reaches the token cap.",
    "chart-two_col_104864": "The teacher uses 50 and 45 as exact readings; only 5 is printed. The two other percentages are absent from the labels in this printed-reading question.",
    "chart-11925": "The actual image gives 50.7, 22.6, and 35.7. The provided table uses 7, 22, 35; the correct chart calculation 0.95 is missing from the options and the teacher output ends by length.",
}

# Explicit intermediate results retained along the actual path of the model. Several solvers
# propagate velocity or momentum across intervals without ever forming a total impulse.
IMPULSE = {
    "force-multi-015": ("30,-60", "The initial momentum is -21 kg m/s. The first impulse is 15 × 2 = 30 N s, giving +9 kg m/s. The second is -12 × (7 - 2) = -60 N s, giving -51 kg m/s; -51 / 3 = -17 m/s."),
    "force-multi-026": ("63,-168,-105", "The impulses are 21 × 3 = 63 and -42 × (7 - 3) = -168 N s. The net is -105 N s. Starting at -7 m/s and adding -105 / 7 gives -22 m/s."),
    "force-multi-017": ("48,-30", "The first impulse is 12 × 4 = 48 N s, so the velocity rises from 5 to 29 m/s. The next impulse is -6 × (9 - 4) = -30 N s, giving 29 - 15 = 14 m/s."),
    "force-multi-016": ("128,-64,64", "The first impulse is 64 × 2 = 128 N s and the next is -16 × (6 - 2) = -64 N s. The net impulse 64 gives a velocity change of 8 m/s, so -4 + 8 = 4 m/s."),
    "force-multi-023": ("100,-60", "The first impulse 25 × 4 = 100 N s increases the velocity from -6 to 14 m/s. The second impulse -30 × (6 - 4) = -60 N s reduces it by 12 m/s, giving 2 m/s."),
    "force-multi-011": ("120,-168,-48", "The two impulses are 24 × 5 = 120 and -24 × (12 - 5) = -168 N s. Their sum is -48 N s. The velocity change is -48 / 6 = -8 m/s, and -5 - 8 = -13 m/s."),
    "force-multi-031": ("35,-210", "Starting momentum is 35 kg m/s. The first impulse 35 × 1 = 35 N s increases it to 70. The second impulse -42 × (6 - 1) = -210 N s gives -140 kg m/s, so -140 / 7 = -20 m/s."),
    "force-multi-010": ("40,-84", "The first impulse is 8 × 5 = 40 N s, bringing velocity from -1 to 19 m/s. The second impulse -12 × (12 - 5) = -84 N s gives a change of -42 m/s and final velocity 19 - 42 = -23 m/s."),
    "force-multi-002": ("75,-80", "The first impulse 25 × 3 = 75 N s raises velocity from 3 to 18 m/s. The next impulse -20 × (7 - 3) = -80 N s gives a change of -16 m/s, so the final velocity is 2 m/s."),
    "force-multi-014": ("24,-36", "The first impulse 8 × 3 = 24 N s raises the velocity from -6 to 0 m/s. The second impulse -12 × (6 - 3) = -36 N s gives -36 / 4 = -9 m/s."),
}


def main():
    cards=json.loads((HERE/"cards.json").read_text(encoding="utf8"))
    judgments={}
    for r in cards:
        rid=r["id"]
        raw=r["raw"]
        if rid in REJECT:
            accept=False
            reason=REJECT[rid]
            witness=[]
            quote=raw[:min(420,len(raw))]
        else:
            accept=True
            if r["schema"]=="impulse":
                witness=IMPULSE[rid][0].split(",")
                reason="Image force plateaus, durations, mass and starting speed match the response, and the two signed updates lead to its correct final velocity and option."
            else:
                witness=r["intermediate"]
                reason=f"All three plotted category/number bindings, the intermediate {witness[0]}, the final value {r['answer_text']} and option {r['answer']} match the source."
            quote=raw[:raw.rfind("Final answer:")].strip()
            assert r["finish_reason"]=="stop" and raw.splitlines()[-1].startswith(f"Final answer: {r['answer']}")
        assert quote in raw
        judgments[rid]=dict(accept=accept,teacher_quote=quote,witness_values=witness,reason=reason,
                            quote_start=raw.index(quote),quote_end=raw.index(quote)+len(quote),
                            raw_sha256=hashlib.sha256(raw.encode()).hexdigest())
        if rid in IMPULSE:
            judgments[rid]["preserved_application"]=IMPULSE[rid][1]
    assert len(judgments)==30 and sum(r["accept"] for r in judgments.values())==27
    dest=DATA/"review_quant_trainer.json"
    dest.write_text(json.dumps(judgments,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(dest, hashlib.sha256(dest.read_bytes()).hexdigest())


if __name__=="__main__":
    main()
