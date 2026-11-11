"""Prepare train-only second-pass examples where one chart value reverses the right decision."""
import json
import random
from decimal import Decimal

from PIL import Image, ImageChops
from score import assess, read, save, sha
from targets import make
from verify import input_prompt
from build import R, paired_chart, fmt


def main():
    out = R / "next_train"
    (out / "images").mkdir(parents=True, exist_ok=True)
    train = read(R / "train/gold.json")
    original = [r for r in train if r["id"].endswith("_r1")]
    heldout_hashes = {r["image_sha256"] for r in read(R / "holdout/gold.json")}
    rng = random.Random(519)
    examples, references = [], []
    for item in original:
        values = {n: Decimal(v) for n,v in item["rows"].items()}
        ordered = sorted(values, key=values.get)
        selected, competitor = ordered[:3], ordered[3]
        low, cutoff = values[selected[1]], values[selected[2]]
        drop = min(values[competitor] - cutoff, (cutoff - low) / 2)
        new = cutoff - drop
        assert low < new < cutoff < values[competitor]
        draft = [values[n] for n in selected]
        rng.shuffle(draft)
        old_total = sum(draft)
        options = [fmt(old_total), fmt(old_total - cutoff + new), fmt(sum(sorted(draft)[:2])), fmt(old_total + values[competitor])]
        rng.shuffle(options)
        assert len(set(options)) == 4
        question = "What is the sum of the three smallest values in the chart? A provisional selection is " + ", ".join(map(fmt, draft)) + "."
        limit = (max(values.values())/10).to_integral_value(rounding="ROUND_CEILING")*10
        family = item["chart"]
        variants = []
        for action in ("keep", "replace"):
            rows = dict(values)
            if action == "replace":
                rows[competitor] = new
            best = sorted(rows, key=rows.get)[:3]
            total = fmt(sum(rows[n] for n in best))
            rel = "images/" + family + "_" + action + ".png"
            im = paired_chart("Station observations — series " + str(int(family[1:])), rows, limit, "Value")
            im.save(out / rel)
            variants.append(im)
            reference = dict(id=family+"_"+action, mode="draft", image=rel, image_sha256=sha(out/rel),
                rows={n:fmt(v) for n,v in rows.items()}, question=question, options=options, truth=total,
                letter="ABCD"[options.index(total)], truth_vals=[fmt(rows[n]) for n in best], draft=list(map(fmt,draft)))
            assert reference["image_sha256"] not in heldout_hashes
            _, e, rule, app, decision = make(reference,draft)
            assert decision == action
            target = f"Evidence: {e}\nRule: {rule}\nApplication: {app}\nFinal answer: {reference['letter']}"
            result = assess(dict(reference,chart=family,layout=1), dict(raw=target,output_tokens=0,finish_reason="stop"))
            assert result["E"] and result["R"] and result["A"] and result["answer"] and result["draft_supported"]
            examples.append(dict(id=reference["id"],image="next_train/"+rel,input_prompt=input_prompt(reference),target=target))
            references.append(reference)
        assert examples[-1]["input_prompt"] == examples[-2]["input_prompt"]
        assert references[-1]["letter"] != references[-2]["letter"]
        assert [n for n in values if references[-1]["rows"][n] != references[-2]["rows"][n]] == [competitor]
        bbox = ImageChops.difference(*variants).getbbox()
        assert bbox is not None
        with Image.open(R / "train" / item["image"]) as source:
            assert ImageChops.difference(source.convert("RGB"), variants[0].convert("RGB")).getbbox() is None
    assert len(examples) == 48
    rng.shuffle(examples)
    output = out/"second_pass.jsonl"
    output.write_text("".join(json.dumps(i, ensure_ascii=False)+"\n" for i in examples),encoding="utf8")
    save(out/"reference.json", references)
    save(out/"receipt.json",dict(pairs=len(original),rows=len(examples),hash=sha(output),source="train only",
        status="Prepared and validated. These examples were not used to train or score the reported models."))


if __name__ == "__main__":
    main()
