"""Inventory source-grounded training candidates without opening evaluation predictions."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts/visual_groups"))
from visual_groups import fingerprint


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s]


def normalized(question):
    return " ".join(re.findall(r"\w+|[%+−-]", question.casefold()))


def signatures(fp):
    return fp["pixel_sha256"], int(fp["dhash"], 16), int(fp["phash"], 16)


def collides(sig, refs):
    pixel, dh, ph = sig
    return any(ref[0] == pixel or (ref[1] ^ dh).bit_count() <= 4 or (ref[2] ^ ph).bit_count() <= 6
               for ref in refs)


def checked_scope():
    photos = read(ROOT / "reports/era_abcd_review_20261020/mmmu_reference_fingerprints.json")
    mmmu = read(ROOT / "reports/mmmu_eval_protocol/evaluation_items.json")
    paths = (
        (ROOT / "results/era_single_call_campaign/data", ["train.jsonl", "dev.jsonl"]),
        (ROOT / "results/era_a_pair_round/data_work", ["train.jsonl"]),
    )
    checked = set()
    questions = set()
    signatures_to_avoid = {signatures(fp) for fp in photos}
    for path, names in paths:
        for file in names:
            for record in rows(path / file):
                pic = path / record["image"]
                if pic not in checked:
                    fp = fingerprint(pic)
                    assert fp["file_sha256"] == record["image_sha256"]
                    signatures_to_avoid.add(signatures(fp))
                    checked.add(pic)
                questions.add(normalized(record["question"]))
    path = ROOT / "results/era_a_pair_round/connection_work/spec"
    for record in read(path / "inputs.json"):
        fp = fingerprint(path / record["image"])
        assert fp["file_sha256"] == record["image_sha256"]
        questions.add(normalized(record["question"]))
        signatures_to_avoid.add(signatures(fp))
    if "question" not in mmmu[0]:
        # This input-only roster has checksums; read only question and id fields from archived inputs.
        archive = read(ROOT / "results/mmmu_first_baseline_20261009/baseline_seed3407/results.json")
        assert [r["id"] for r in mmmu] == [r["id"] for r in archive["records"]]
        mmmu = archive["records"]
    questions.update(normalized(r["question"]) for r in mmmu)
    return questions, signatures_to_avoid


def main():
    source = ROOT / "tmp/era_prereview_20261021/scienceqa_train.parquet"
    with source.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = read(ROOT / "tmp/era_prereview_20261021/replacement_asset_provenance.json")
    assert checksum == receipt["parquet_sha256"]
    metadata_root = ROOT / "reports/era_dataset_examples_20261014"
    metadata = read(metadata_root / "scienceqa_problems.json")
    train_ids = read(metadata_root / "scienceqa_splits.json")["train"]
    assert len(train_ids) == pq.ParquetFile(source).metadata.num_rows == 12726
    reference_questions, reference_images = checked_scope()
    count, subjects = Counter(), Counter()
    survivors = []
    index = 0
    image_cache = {}
    with tempfile.TemporaryDirectory(dir=HERE) as folder:
        pic = Path(folder)/"image.png"
        for batch in pq.ParquetFile(source).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                sid = str(train_ids[index])
                index += 1
                gold = metadata[sid]
                assert gold["split"] == "train" and all(row[k] == gold[k] for k in row if k != "image")
                count["metadata_matched"] += 1
                if not gold["image"]:
                    continue
                assert row["image"] and row["image"]["bytes"]
                count["embedded_image"] += 1
                if not row["solution"].strip():
                    continue
                count["image_plus_source_solution"] += 1
                q = normalized(row["question"])
                if q in reference_questions:
                    count["excluded_matching_question"] += 1
                    continue
                blob = row["image"]["bytes"]
                key = hashlib.sha256(blob).digest()
                if key not in image_cache:
                    pic.write_bytes(blob)
                    sig = signatures(fingerprint(pic))
                    image_cache[key] = (sig, collides(sig, reference_images))
                sig, collision = image_cache[key]
                if collision:
                    count["excluded_matching_image"] += 1
                    continue
                survivors.append((sid, q, sig, row["subject"], row["topic"]))
                subjects[row["subject"]] += 1
    assert index == len(train_ids)
    # A small outcome-independent reservoir. No exact prompt or exact/near image is shared
    # by any two rows, including across calibration and assistant-generated training data.
    ordered = sorted(survivors, key=lambda row: hashlib.sha256(("ERA-original-solutions-v1/" + row[0]).encode()).digest())
    selected, selected_questions, selected_sigs = [], set(), []
    for sid, q, sig, subject, topic in ordered:
        if q in selected_questions or collides(sig, selected_sigs):
            continue
        selected_questions.add(q)
        selected_sigs.append(sig)
        selected.append(dict(scienceqa_id=sid, subject=subject, topic=topic, image_signature=sig[0]))
    for idx,item in enumerate(selected):
        item["role"] = "diagnostic" if idx % 3 == 0 else "train_candidate"
    subroles = {role:dict(Counter(r["subject"] for r in selected if r["role"] == role))
                for role in ("diagnostic","train_candidate")}
    result = dict(source_sha256=checksum, source_revision=receipt["revision"], **count, after_refs=len(survivors),
                  subject_counts=subjects, reference_normalized_questions=len(reference_questions),
                  reference_image_signatures=len(reference_images), unique_questions_images=len(selected),
                  reserved_roles=subroles, near_image_rule="same RGBA or dHash <=4 or pHash <=6",
                  note="Source reference solutions, not Qwen-generated traces. Roles are a proposed split, not an executed benchmark. Membership is greedy within a fixed hash order.")
    for file, data in (("source_inventory.json", result), ("candidate_ids.json", selected)):
        (HERE / file).write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf8")
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
