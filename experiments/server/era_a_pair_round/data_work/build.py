"""Build A-only preference pairs from frozen train-side teaching records and numeric source tables."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIOR = ROOT / "results/era_single_call_campaign/data"
ANCHORS = ROOT / "results/era_self_1k_v1/anchors"
HOLD = HERE.parent / "connection_work/spec"
OFFICIAL = ROOT / "results/mmmu_first_baseline_20261009/baseline_seed3407/results.json"
FINGERPRINTS = ROOT / "reports/era_abcd_review_20261020/mmmu_reference_fingerprints.json"
sys.path.insert(0, str(ROOT / "scripts/visual_groups"))
from visual_groups import fingerprint, is_candidate


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(s):
    return " ".join(re.findall(r"\w+|[%+−-]", s.lower()))


def fmt(number):
    return format(Decimal(number).normalize(), "f")


def save(path, content):
    raw = content.encode("utf8")
    if not path.exists() or path.read_bytes() != raw:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)


def jsonl(rows):
    return "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)


def curated():
    rules = {}
    for line in (HERE / "curated.tsv").read_text(encoding="utf8").splitlines():
        if not line or line.startswith("#"):
            continue
        item, template, correct, wrong = line.split("|")
        assert item not in rules and template.count("{x}") == 1 and correct != wrong
        rules[item] = (template.format(x=correct), template.format(x=wrong))
    return rules


def science(row):
    q, e, ans = row["question"], row["evidence"], row["answer"]
    options = dict(re.findall(r"^([A-E])\. ([^\n]+)", row["prompt"], re.M))
    target = options.get(ans, "")
    if "higher concentration of" in q and target in {"Solution A", "Solution B"}:
        label = target[-1]
        other = {"A": "B", "B": "A"}[label]
        template = "At equal solvent volumes, the larger particle count belongs to solution {}, giving it the higher concentration."
        if re.search(r"solution a (?:contains|has) (\d+)", e, re.I):
            counts = re.findall(r"solution [ab] (?:contains|has) (\d+)", e, re.I)
            assert len(counts) == 2 and counts[0] != counts[1]
            assert label == ("A" if int(counts[0]) > int(counts[1]) else "B")
        else:
            assert re.search(r"Solution A shows more \w+ solute particles than Solution B", e) and label == "A"
        return template.format(label), template.format(other), "solution_particle_count"
    if target in {"repel", "attract"} and "attract or repel" in q:
        other = {"repel": "attract", "attract": "repel"}[target]
        template = "Applying the pole rule to the ends facing across the gap, the magnets {} each other."
        return template.format(target), template.format(other), "facing_poles"
    m = re.search(r"Which of these states is farthest (east|west|north|south)", q, re.I)
    if m and len(options) == 4:
        direction = m[1].lower()
        alt = next(v for k, v in sorted(options.items()) if k != ans)
        assert target in e and alt in e
        template = "Following the compass to the " + direction + ", {} is farthest in that direction among the four labeled states."
        return template.format(target), template.format(alt), "compass_direction"


def music(row):
    answer, ev = row["answer"], row["evidence"]
    m = re.search(r"(\d+) (sharp|flat)s? between", ev)
    if m:
        orders = {"sharp": "F# C# G# D# A# E# B#".split(), "flat": "Bb Eb Ab Db Gb Cb Fb".split()}
        n, kind = int(m[1]), m[2]
        expected = orders[kind][:n]
        assert expected == answer.split(", ")
        wrong = orders[kind][1:n+1]
        template = "The {} signature mark{} correspond{} to {} in order."
        intro = [str(n), "s" if n != 1 else "", "" if n != 1 else "s"]
        return template.format(*intro, ", ".join(expected)), template.format(*intro, ", ".join(wrong)), "signature_order"
    match = re.search(r"first two notes.*?([A-G][b#]?\d+) and ([A-G][b#]?\d+)", ev)
    if match:
        p, q = match.groups()
        def key(pitch):
            n = "C D E F G A B".split().index(pitch[0])
            return 12 * int(pitch[-1]) + [0, 2, 4, 5, 7, 9, 11][n] + pitch.count("#") - pitch.count("b")
        change = key(q) - key(p)
        assert change
        direction = "ascending" if change > 0 else "descending"
        inverse = "descending" if change > 0 else "ascending"
        assert answer.startswith(direction + " ")
        interval = answer.split(" ", 1)[1]
        template = f"From {p} to {q}, the distance is {abs(change)} semitones. Reading the written pitches in order gives an {{}} {interval}."
        # Avoid leaking preference via a/an in either direction.
        template = template.replace("gives an {}", "gives the {}")
        return template.format(direction), template.format(inverse), "interval_direction"
    return None


def source_pairs():
    receipt = read(PRIOR / "receipt.json")
    assert sha(PRIOR / "train.jsonl") == receipt["splits"]["train"]["sha256"]
    assert sha(OFFICIAL) == receipt["reference_sha256"] and sha(FINGERPRINTS) == receipt["fingerprint_source_sha256"]
    manual = curated()
    found = set()
    repeated = Counter()
    for line in (PRIOR / "train.jsonl").read_text(encoding="utf8").splitlines():
        x = json.loads(line)
        if x["id"] in manual:
            found.add(x["id"])
            pair = (*manual[x["id"]], "item_specific_binding")
        elif x["dataset"] == "ScienceQA":
            pair = science(x)
        elif x["dataset"] == "D/MusiXQA":
            pair = music(x)
        else:
            pair = None
        if not pair:
            continue
        pos, neg, family = pair
        if family in {"signature_order", "interval_direction"}:
            repeated[(pos, neg)] += 1
            if repeated[(pos, neg)] > 2:
                continue
        result = dict(id=x["id"], prompt=x["prompt"], question=x["question"].split("\nOn the last line")[0],
                      evidence=x["evidence"], rule=x["rule"], chosen_application=pos, rejected_application=neg,
                      final_answer=x["answer"], source=x["dataset"], source_id=x["source_id"],
                      source_revision=x["source_revision"], source_group=x["source_partition_group"], contrast=family)
        yield result, PRIOR / x["image"], x["image_sha256"]
    assert found == set(manual), set(manual) - found


def chart_pairs():
    reference = ANCHORS / "reference.json"
    assert sha(reference) == read(ANCHORS / "receipt.json")["reference_sha256"]
    rows = [r for r in read(reference) if r["mode"] == "direct"]
    random.Random(77131).shuffle(rows)
    for index, original in enumerate(rows[:72]):
        rng = random.Random(original["id"] + "/contrast4")
        values = {n: Decimal(v) for n, v in original["rows"].items()}
        lo, hi = sorted(values, key=values.get), sorted(values, key=values.get, reverse=True)
        assert len(set(values.values())) == len(values) >= 5
        a, b, c, d = lo[:4]
        t = index % 4
        if t in {0, 1}:
            direction = "smallest" if t == 0 else "largest"
            if t == 1:
                a, b, c, d = hi[:4]
            q = f"What is the sum of the three {direction} values on the chart?"
            rule = f"Order the bars by their printed values, take the three {direction}, and add those values."
            right = sum(values[n] for n in (a, b, c))
            wrong = sum(values[n] for n in (a, b, d))
            def app(third):
                terms = [a, b, third]
                return "The selected bars are " + ", ".join(f"{n} ({fmt(values[n])})" for n in terms) + f", whose values sum to {fmt(sum(values[n] for n in terms))}."
            pos, neg = app(c), app(d)
            contrast = direction + "_selection"
        elif t == 2:
            a, b, c = hi[:3]
            q = "How much larger is the highest value than the second-highest value?"
            rule = "Order the measured values from largest to smallest, then subtract the second from the first."
            right, wrong = values[a] - values[b], values[a] - values[c]
            def app(second):
                return f"{a} is highest at {fmt(values[a])} and {second} is next at {fmt(values[second])}. Their difference is {fmt(values[a]-values[second])}."
            pos, neg = app(b), app(c)
            contrast = "rank_difference"
        else:
            a, b = rng.sample(list(values), 2)
            q = f"What is the combined value for {a} and {b}?"
            rule = "Locate the two named rows, read the value of each row, and add the two readings."
            right = values[a] + values[b]
            wrong = right  # The arithmetic and endpoint coincide despite the incorrect row-value assignment.
            def app(first, second):
                return f"{a} contributes {fmt(first)} and {b} contributes {fmt(second)}, so the combined value is {fmt(first+second)}."
            pos, neg = app(values[a], values[b]), app(values[b], values[a])
            contrast = "swapped_bindings_same_total"
        if t != 3:
            assert right != wrong
        evidence = "The labeled bars read " + ", ".join(f"{n} {fmt(v)}" for n, v in values.items()) + "."
        if index % 8 < 4:
            answer = fmt(right)
            prompt = 'Question: ' + q + '\nOn the last line, write "Final answer: ANSWER", replacing ANSWER with a concise final answer.'
        else:
            choices = []
            for v in [right, wrong, values[a], values[b], right+values[d], right+values[c]]:
                if fmt(v) not in choices:
                    choices.append(fmt(v))
            assert len(choices) >= 4, original
            choices = choices[:4]
            rng.shuffle(choices)
            answer = "ABCD"[choices.index(fmt(right))]
            prompt = "Question: " + q + "\nOptions:\n" + "\n".join(f"{l}. {v}" for l, v in zip("ABCD", choices)) + '\nOn the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'
        result = dict(id="r4_" + original["id"], prompt=prompt, question=q, evidence=evidence, rule=rule,
                      chosen_application=pos, rejected_application=neg, final_answer=answer, source="train_anchor_chart",
                      source_id=original["id"], source_revision=sha(reference), source_group=original["chart"],
                      contrast=contrast, table=original["rows"])
        yield result, ANCHORS / original["image"], original["image_sha256"]


def main():
    official = read(OFFICIAL)["records"]
    comparisons = list(read(FINGERPRINTS))
    holdouts = read(HOLD / "inputs.json")
    assert len(official) == 900 and len(comparisons) == 982 and len(holdouts) == 48
    for h in holdouts:
        pic = HOLD / h["image"]
        fp = fingerprint(pic)
        assert h["image_sha256"] == fp["file_sha256"]
        comparisons.append(dict(id=h["id"], **fp))
    questions = {normalized(x["question"]) for x in official + holdouts}
    dest = HERE / "images"
    dest.mkdir(exist_ok=True)
    rows, blocked, cache = [], {}, {}
    for row, path, expected in [*source_pairs(), *chart_pairs()]:
        if expected not in cache:
            fp = fingerprint(path)
            assert fp["file_sha256"] == expected
            cache[expected] = fp, [v["id"] for v in comparisons if fp["pixel_sha256"] == v["pixel_sha256"] or is_candidate(fp, v)]
        fp, matches = cache[expected]
        question_hit = normalized(row["question"]) in questions
        if matches or question_hit:
            blocked[row["id"]] = dict(images=matches, same_question=question_hit)
            continue
        name = expected + path.suffix.lower()
        target = dest / name
        if not target.exists():
            shutil.copyfile(path, target)
        assert sha(target) == expected
        assert all(row[k] and "\n" not in row[k] for k in ["evidence","rule","chosen_application","rejected_application","final_answer"])
        assert row["chosen_application"] != row["rejected_application"]
        rows.append(dict(**row, image="images/" + name, image_sha256=expected, pixel_sha256=fp["pixel_sha256"]))
    assert len(rows) == len({r["id"] for r in rows}) and len(rows) >= 120
    assert len(rows) == len({(r["pixel_sha256"], r["prompt"]) for r in rows})
    random.Random(3407).shuffle(rows)
    train = HERE / "train.jsonl"
    save(train, jsonl(rows))
    report = dict(verified=True, n=len(rows), train_sha256=sha(train), images=len({r["image_sha256"] for r in rows}),
                  source_groups=len({r["source_group"] for r in rows}), sources=dict(Counter(r["source"] for r in rows)),
                  contrasts=dict(Counter(r["contrast"] for r in rows)), blocked=blocked,
                  official_questions=len(official), official_images=982, independent_connection_items=len(holdouts),
                  heldout_sha256=sha(HOLD / "inputs.json"), official_results_sha256=sha(OFFICIAL), official_references_sha256=sha(FINGERPRINTS),
                  original_train_sha256=sha(PRIOR / "train.jsonl"), anchor_source_sha256=sha(ANCHORS / "reference.json"),
                  build_sha256=sha(Path(__file__)), curated_sha256=sha(HERE / "curated.tsv"),
                  contrast_mask="Application span only; both arms learn identical complete correct ERA sequence using supervised loss.",
                  overlap_rule="Excluded any equal normalized question, exact RGBA pixels, dHash Hamming <=4, or pHash <=6 versus official MMMU or frozen 48-diagnostic; templates/tasks may still be related.")
    save(HERE / "receipt.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
