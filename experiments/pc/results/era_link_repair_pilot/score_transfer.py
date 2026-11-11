"""Use the original 64-item study's answer rubric for the secondary transfer check."""
import importlib.util
from score import ARMS, R, read, save, sha


def main():
    old = R.parent / "d226_echeck_20261105"
    spec = importlib.util.spec_from_file_location("previous_d226_evaluation", old / "analyze.py")
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    inputs = {row["id"]: row for row in read(old / "fresh/inputs.json")}
    cells = {
        cell["id"]: (chart["options"], cell)
        for chart in read(old / "fresh/items.json") for cell in chart["cells"].values()
    }
    prompts = {}
    details = []
    result = {}
    baseline = {}
    for arm in ARMS:
        directory = old / "eval/checked" if arm == "baseline" else R / "transfer" / arm
        predictions = prior.load_rows(directory / "predictions.jsonl")
        assert len(predictions) == len(cells) == len({x["id"] for x in predictions}) == 64
        rows = []
        for pred in predictions:
            rid = pred["id"]
            options, cell = cells[rid]
            assert pred["image_sha256"] == inputs[rid]["image_sha256"] == sha(old / "fresh" / inputs[rid]["image"])
            assert pred["prompt"] == prompts.setdefault(rid, pred["prompt"])
            a = prior.answer(pred, options, cell["truth"]["letter"])
            rec = dict(arm=arm, id=rid, aligned=cell["aligned"], answer=a["correct"], letter=a["letter"],
                       truth=cell["truth"]["letter"], finish=a["finish"], E_questions=a["E_question_mark"])
            rows.append(rec)
            if arm == "baseline":
                baseline[rid] = a["correct"]
        details.extend(rows)
        result[arm] = dict(n=len(rows), correct=sum(r["answer"] for r in rows),
            kept=sum(baseline[r["id"]] and r["answer"] for r in rows),
            gained=sum(not baseline[r["id"]] and r["answer"] for r in rows),
            lost=sum(baseline[r["id"]] and not r["answer"] for r in rows),
            aligned=sum(r["answer"] and r["aligned"] for r in rows),
            conflict=sum(r["answer"] and not r["aligned"] for r in rows),
            E_questions=sum(r["E_questions"] for r in rows), length=sum(r["finish"]=="length" for r in rows))
    assert result["baseline"]["correct"] == 53
    save(R / "transfer_metrics.json", result)
    save(R / "transfer_scores.json", details)


if __name__ == "__main__":
    main()
