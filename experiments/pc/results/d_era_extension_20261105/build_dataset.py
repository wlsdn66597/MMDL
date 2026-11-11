"""Build and verify the derivative using the pinned reviewed400 zip and local tokenizer."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parent
PIN = "3d877fe99fddb2bea32f030552b4ec6fff9c82581b8e0756f5dc6029400b4c4a"
# Source ordinals are tied to PIN. These are 20 first measures reviewed side by side in qa/chord{1,2}.png.
CHORD_VISIBLE = {1, 10, 57, 154, 185, 235}
CHORD_HOLD = {37, 49, 78, 85, 102, 111, 118, 124, 175, 202, 209, 212, 225, 229}
EVIDENCE_CHANGES = {
    41: "The question identifies the central figure as Powhatan. In the upper-left inset he sits above the surrounding figures.",
    98: "The painting shows killings in an urban setting. The question identifies the event as the St. Bartholomew's Day Massacre.",
    100: "The bank diagram shows receipt of 42.4 million baht, a conversion rate of 50 baht per dollar on the right, and a $1.05 million debt.",
    131: "The diagram labels intra-alveolar pressure as 760 mm Hg and intrapleural pressure as 756 mm Hg.",
    134: "The schedule lists $75,000 beginning work in process, $63,000 current manufacturing costs and $43,000 actual ending work in process. The question substitutes $50,000 for the ending balance.",
    144: "The K+ label identifies the blue ions. They are concentrated on the cytosolic side; the gradient arrow for K+ points outward.",
    176: "Elizabeth places her hand on a globe. The scenes behind her feature ships at sea.",
}
LETTERS = "CDEFGAB"
PITCH_CLASS = dict(zip(LETTERS, (0, 2, 4, 5, 7, 9, 11)))
INTERVALS = {(3, 3): "minor third", (3, 4): "major third", (4, 5): "perfect fourth", (5, 7): "perfect fifth", (6, 8): "minor sixth", (6, 9): "major sixth", (8, 12): "perfect octave"}
def sha(data):
    return hashlib.sha256(data).hexdigest()
def read_rows(z, name):
    return list(map(json.loads, z.read(name).splitlines()))
def write_rows(path, rows):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
def pitch(note):
    letter, accidental, octave = re.fullmatch(r"([A-G])([b#]?)(-?\d+)", note).groups()
    return 12 * (int(octave) + 1) + PITCH_CLASS[letter] + {"": 0, "#": 1, "b": -1}[accidental], 7 * int(octave) + LETTERS.index(letter)
def pc(name):
    return pitch(name + "4")[0] % 12

def music(i, r, meta):
    task, ans = r["task"], r["answer"]
    if task == "key_signature":
        sig = meta["signature"]
        order = "Bb Eb Ab Db Gb Cb Fb".split() if sig and sig[0].endswith("b") else "F# C# G# D# A# E# B#".split()
        assert sig == order[:len(sig)]
        assert ans == (", ".join(sig) if sig else "none")
        rule = "Read the signature between the clef and time signature, taking the indicated prefix of the standard order of sharps or flats."
        if sig:
            kind = "flat" if sig[0].endswith("b") else "sharp"
            evidence = f"There {'is' if len(sig) == 1 else 'are'} {len(sig)} {kind}{'s' if len(sig) != 1 else ''} between the clef and time signature."
            apply = f"The first {len(sig)} {'name is' if len(sig) == 1 else 'names are'} {', '.join(sig)} in standard signature order."
        else:
            evidence = "There are no accidentals between the clef and the time signature."
            apply = "The empty signature alters no pitch names."
    elif task == "scale_degree":
        key, scale, first = meta["key"], meta["scale"], meta["first_note"]
        tonic, mode = key.split()
        intervals = (0, 2, 4, 5, 7, 9, 11) if mode == "Major" else (0, 2, 3, 5, 7, 8, 10)
        assert [(pc(n) - pc(tonic)) % 12 for n in scale] == list(intervals)
        spelling = re.sub(r"-?\d+$", "", first)
        degree = scale.index(spelling) + 1
        assert str(degree) == ans and degree == meta["degree"] and first in r["evidence"]
        evidence = f"The first written note on the staff specified in the question is {first}."
        rule = "Count the tonic as degree 1 and use the ordered notes of the specified scale, ignoring octave."
        if degree == 1:
            apply = f"{spelling} is the tonic of {tonic} {mode.lower()}, so the first note is degree 1."
        else:
            apply = f"Counting up the {tonic} {mode.lower()} scale gives {', '.join(scale[:degree])}. The written {first} matches position {degree}."
    elif task == "named_interval":
        a, b = re.findall(r"[A-G](?:b|#)?-?\d+", r["evidence"])
        m1, l1 = pitch(a)
        m2, l2 = pitch(b)
        number, half = abs(l2 - l1) + 1, abs(m2 - m1)
        direction = "ascending" if m2 > m1 else "descending"
        name = INTERVALS[number, half]
        assert (number, half, direction) == (meta["interval_number"], meta["semitones"], meta["direction"])
        assert ans == direction + " " + name
        evidence = f"The first two notes on the specified staff are {a} and {b}, in that order."
        rule = "Use inclusive letter-name distance for the interval number, semitone distance for quality, and pitch order for direction."
        apply = (f"What is the interval number? Counting inclusively from {a} to {b} gives {number}. "
                 f"What is its size and direction? The pitches move {half} semitones {'up' if m2 > m1 else 'down'}. "
                 f"A {number}-note span of {half} semitones is a {name}, so the directed interval is {direction} {name}.")
    elif task == "chord_member":
        assert i in CHORD_VISIBLE | CHORD_HOLD
        first, chord, quality = meta["first_pitch"], meta["native_chord"], meta["quality"]
        root = chord[:-1] if quality == "minor" else chord
        off = (pitch(first)[0] - pc(root)) % 12
        third = 3 if quality == "minor" else 4
        member = {0: "root", third: "third", 7: "fifth"}[off]
        assert off == meta["offset"] and pc(root) == meta["root_pc"] and ans == f"{first}; {member}"
        if i in CHORD_HOLD:
            # The archived symbolic gold is preserved, but would require input beyond the pixels.
            return None
        evidence = f"The first written pitch is {first} and the chord symbol above the measure reads {chord}."
        rule = f"For a {quality} triad the root, third, and fifth lie 0, {third}, and 7 semitones above the root modulo the octave."
        apply = (f"How far is {first} above root {root} modulo an octave? The offset is {off} semitones. "
                 f"That offset matches the {member} in this {quality} triad, so the note is {first}, the {member}.")
    else:
        raise ValueError(task)
    return evidence, rule, apply

def main(archive, tokenizer_file, tokenizer_packages):
    if tokenizer_packages:
        sys.path.insert(0, str(tokenizer_packages))
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    ids = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
    assert sha(archive.read_bytes()) == PIN
    rules = {}
    for name in ("anatomy", "other"):
        for line in (ROOT / f"qa/{name}_authoring.tsv").read_text(encoding="utf-8").splitlines():
            num, rule, apply = line.split("|")
            assert int(num) not in rules
            rules[int(num)] = (rule, apply)
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read("manifest.json"))
        assert len(z.namelist()) == len(set(z.namelist()))
        assert set(z.namelist()) == set(manifest) | {"manifest.json"}
        assert all(sha(z.read(path)) == digest for path, digest in manifest.items())
        rows = read_rows(z, "data/d_era.jsonl")
        general = {r["id"]: r for r in read_rows(z, "data/d_general.jsonl")}
        provenance = {r["id"]: r for r in read_rows(z, "metadata/provenance.jsonl")}
        pre = {r["id"]: r for r in json.loads(z.read("metadata/era_processor_preflight.json"))["per_row"]}
        original_ids = [r["id"] for r in rows]
        assert len(set(original_ids)) == len(rows) == 240 and set(original_ids) == set(general) == set(pre)
        assert all(r["role"] == "D" for r in rows)
        image_paths = sorted({p for r in rows for p in r["images"]})
        pixel_groups = defaultdict(list)
        for n in image_paths:
            assert n.startswith("images/") and Path(n).name == n[len("images/"):]
            b = z.read(n)
            with Image.open(BytesIO(b)) as im:
                im.load()
                pixel_groups[(im.size, sha(im.convert("RGB").tobytes()))].append(n)
            (ROOT / n).parent.mkdir(exist_ok=True, parents=True)
            (ROOT / n).write_bytes(b)
        for n in ["LICENSES.md", "metadata/sources.json", "credits/OpenStax_anatomy_LICENSE.txt", "credits/OpenStax_anatomy_pin.json", "credits/SemArt_README.txt", "credits/MusiXQA_card.md"]:
            (ROOT/n).parent.mkdir(exist_ok=True, parents=True)
            (ROOT/n).write_bytes(z.read(n))
        result, held, audit = [], [], []
        for i,r in enumerate(rows):
            sid = r["id"]
            p, g = provenance[sid], general[sid]
            assert r["input_prompt"] == g["input_prompt"] and r["images"] == g["images"] == p["images"] and r["answer"] == g["answer"]
            assert p["images_sha256"] == [manifest[n] for n in r["images"]]
            assert len(r["images"]) == 1
            assert sha(r["target"].encode()) == r["target_sha256"]
            old = pre[sid]
            assert len(ids(r["target"])) + 1 == old["reason_tokens"] + old["answer_tokens"]
            if r["source_slug"] != "MusiXQA":
                evidence = EVIDENCE_CHANGES.get(i, r["evidence"])
                parts = (evidence, *rules[i])
            else:
                parts = music(i, r, p["source_evidence"])
            record = {k:r[k] for k in ("id","images","input_prompt","role","source_slug","source_group","source_url","source_revision","domain","task","answer")}
            base_audit = dict(id=sid, source_index_zero_based=i, original_target_sha256=r["target_sha256"],
                              paired_original_general_sha256=g["target_sha256"], original_image_sha256=p["images_sha256"],
                              input_prompt_sha256=sha(r["input_prompt"].encode()),
                              reference_answer_sha256=sha(r["answer"].encode()), inherited_source_status=r["review_status"])
            if parts is None:
                held.append(dict(**record, status="hold_missing_first_measure_chord_symbol",
                                 source_era_target_archived=r["target"], source_symbolic_annotation=p["source_evidence"],
                                 reason="Source gold needs a chord symbol above the first measure. None is printed there; the source symbolic annotation is unavailable to the student."))
                audit.append(dict(**base_audit, disposition="held", evidence_action="quarantined"))
                continue
            evidence, rule, apply = parts
            ans = r["answer"]
            assert not re.search(r"(?m)^(Evidence|Rule|Application|Final answer):", "\n".join(parts))
            for letter in re.findall(r"\boption ([A-Z])\b", apply):
                assert letter == ans, (sid, letter, ans)
            if re.search(r"^Options:$", r["input_prompt"], re.M):
                assert re.search(rf"^{ans}\. ", r["input_prompt"], re.M)
            target = f"Evidence: {evidence}\nRule: {rule}\nApplication: {apply}\nFinal answer: {ans}"
            total = old["prompt_tokens"] + len(ids(target)) + 1
            assert total == old["total_tokens"] + len(ids(target)) - len(ids(r["target"]))
            assert total <= 4096, (sid, total)
            style = "subquestion_return" if "?" in apply else "direct"
            result.append(dict(**record, evidence=evidence, rule=rule, application=apply, target=target,
                               target_sha256=sha(target.encode()), reasoning_style=style))
            action = "rewritten" if evidence != r["evidence"] else "kept"
            audit.append(dict(**base_audit, disposition="included", reasoning_style=style, target_sha256=sha(target.encode()),
                              evidence_action=action, total_tokens_same_prompt_plus_new_target=total))
        train_file, hold_file = ROOT/"data/d_era.jsonl", ROOT/"metadata/held.jsonl"
        write_rows(train_file,result)
        write_rows(hold_file,held)
        write_rows(ROOT/"metadata/row_audit.jsonl",audit)
        write_rows(ROOT/"metadata/d_provenance.jsonl",[provenance[s] for s in original_ids])
    assert len(rules) == 160 and len(held) == len(CHORD_HOLD) and len(result)+len(held)==240
    decoded = [json.loads(s) for s in train_file.read_text(encoding="utf-8").splitlines()]
    assert len(decoded) == len({r["id"] for r in decoded}) and len(decoded)==len({(r["input_prompt"],tuple(r["images"])) for r in decoded})
    for r in decoded:
        assert sha(r["target"].encode()) == r["target_sha256"]
        assert re.findall(r"^(Evidence|Rule|Application|Final answer):",r["target"],re.M) == ["Evidence","Rule","Application","Final answer"]
        assert r["target"].splitlines()[-1] == "Final answer: " + r["answer"]
        for n in r["images"]:
            assert sha((ROOT/n).read_bytes()) == manifest[n]
    stats = dict(source_zip_sha256=PIN, source_member_hashes_verified=len(manifest), source_D=240,
                 included=len(decoded), held=len(held), R_excluded=160,
                 count_by_source=dict(Counter(r["source_slug"] for r in decoded)),
                 reasoning_styles=dict(Counter(r["reasoning_style"] for r in decoded)), all_d_source_groups=len({r["source_group"] for r in rows}),
                 included_source_groups=len({r["source_group"] for r in decoded}), unique_images_all_D=len(image_paths),
                 unique_images_included=len({n for r in decoded for n in r["images"]}),
                 identical_pixel_files={str(k):v for k,v in pixel_groups.items() if len(v)>1},
                 included_unique_question_image_pairs=len(decoded),
                 included_answer_types=dict(Counter("multiple_choice" if "\nOptions:\n" in r["input_prompt"] else "free_response" for r in decoded)),
                 held_reason="14 first-measure chord symbols absent from student images",
                 inherited_status_counts=dict(Counter(r["review_status"] for r in rows)),
                 maximum_length_new_same_prompt_from_original_preflight=max(r["total_tokens_same_prompt_plus_new_target"] for r in audit if r["disposition"]=="included"),
                 tokenizer_sha256=sha(tokenizer_file.read_bytes()), all_sources_reviewed=True, separate_expert_review=False,
                 length_check_scope="Verified original processor prompt length + local pinned target token count + termination allowance; no fresh processor run or training performed.",
                 new_train_sha256=sha(train_file.read_bytes()), archived_hold_sha256=sha(hold_file.read_bytes()))
    dump(ROOT/"validation.json",stats)
    print(json.dumps(stats, ensure_ascii=False))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("source_zip", type=Path)
    repo = ROOT.parents[1]
    parser.add_argument("--tokenizer", type=Path, default=repo/".runtime/qwen_tokenizer/tokenizer.json")
    parser.add_argument("--tokenizer-packages", type=Path, default=repo/".runtime/tokenizers")
    args = parser.parse_args()
    main(args.source_zip, args.tokenizer, args.tokenizer_packages)
