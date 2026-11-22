"""CPU-only checks for mistakes that would invalidate a long GPU evaluation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("protocol", ROOT / "scripts/protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
sys.path.insert(0, str(ROOT / "scripts/diag"))
from common import grade_prediction


class FrozenEvaluatorTest(unittest.TestCase):
    def test_archived_source_bytes_are_unchanged(self):
        protocol.source_check()

    def test_loop_stop_cannot_be_counted_correct(self):
        rec = {"question_type": "multiple-choice", "options": {"A": "yes", "B": "no"}, "reference_answer": "A"}
        pred = {"model_response": {"raw": "Final answer: A", "generation_status": "loop_stopped"}}
        self.assertEqual(grade_prediction(pred, rec)["verdict"], "incorrect")


class ResumeValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "predictions.jsonl"
        self.allowed = {"test_Art_1", "test_Math_1"}
        self.rows = [{"id": rid, "seed": 42, "model_path": "/models/merged"} for rid in sorted(self.allowed)]

    def tearDown(self):
        self.temp.cleanup()

    def save(self, rows):
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def read(self, **kwargs):
        return protocol.read_rows(self.path, self.allowed, seed=42, model="/models/merged", **kwargs)

    def test_partial_generation_can_resume_but_is_not_complete(self):
        self.save(self.rows[:1])
        self.assertEqual(len(self.read()), 1)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            self.read(complete=True)

    def test_duplicates_do_not_count_as_complete(self):
        self.save([self.rows[0], self.rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.read(complete=True)

    def test_wrong_test_split_is_rejected(self):
        self.save([dict(self.rows[0], id="validation_Art_1")])
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.read()

    def test_other_seed_or_model_cannot_resume(self):
        for bad in (dict(self.rows[0], seed=3407), dict(self.rows[0], model_path="/models/other")):
            self.save([bad])
            with self.assertRaisesRegex(ValueError, "Mixed"):
                self.read()

    def test_baseline_null_model_is_checked_not_skipped(self):
        self.save([dict(r, model_path=None) for r in self.rows])
        self.assertEqual(len(protocol.read_rows(self.path, self.allowed, seed=42, model=None)), 2)
        self.save(self.rows)
        with self.assertRaisesRegex(ValueError, "Mixed model"):
            protocol.read_rows(self.path, self.allowed, seed=42, model=None)

    def test_invalid_seed_is_rejected(self):
        for seed in (-1, 2**32, True, "42"):
            with self.assertRaises(ValueError):
                protocol.validate_seed(seed)
        for seed in (0, 42, 3407):
            protocol.validate_seed(seed)

    def test_truncated_row_is_reported_without_modifying_outputs(self):
        self.path.write_text('{"id":', encoding="utf-8")
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.read()
        self.assertEqual(self.path.read_bytes(), original)

    def test_complete_id_set_passes(self):
        self.save(self.rows)
        self.assertEqual(set(self.read(complete=True)), self.allowed)

    def test_record_hash_change_is_rejected(self):
        with patch.object(protocol, "reference", return_value={"records_sha256": "0" * 64}):
            self.path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                protocol.records_check(self.path)

    def test_tags_cannot_inject_paths_or_shell_text(self):
        for tag in ("../other", "a:b", "a'", "a b", "", "-other"):
            with self.assertRaises(ValueError):
                protocol.validate_tag(tag)
        protocol.validate_tag("era_plus_test_seed42")


class RunIdentityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = {"records": [{"id": "test_Art_1"}, {"id": "test_Math_1"}],
                     "evaluation": {"settings": {}}}
        self.model = {"path": "/models/merged", "files": {"model.safetensors": "original-weight-hash"}}
        mocks = [patch.object(protocol, "ROOT", self.root),
                 patch.object(protocol, "source_check"),
                 patch.object(protocol, "records_check", return_value=self.data),
                 patch.object(protocol, "runtime_info", return_value={"gpu": "RTX 4090", "packages": {}}),
                 patch.object(protocol, "model_fingerprint", return_value=self.model),
                 patch.object(protocol, "reference", return_value={"records_sha256": "fixed-records",
                              "source_sha256": {}, "evaluation": {"generation_seed": 42}})]
        for item in mocks:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.temp.cleanup)

    def test_same_checkpoint_resumes(self):
        protocol.begin("/models/merged", "run42")
        path = self.root / "outputs/run42.manifest.json"
        original = path.read_bytes()
        protocol.begin("/models/merged", "run42")
        self.assertEqual(path.read_bytes(), original)

    def test_changed_weights_cannot_reuse_tag(self):
        protocol.begin("/models/merged", "run42")
        self.model["files"]["model.safetensors"] = "new-weight-hash"
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            protocol.begin("/models/merged", "run42")

    def test_changed_seed_cannot_reuse_tag(self):
        protocol.begin("/models/merged", "run42", seed=42)
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            protocol.begin("/models/merged", "run42", seed=3407)

    def test_seed3407_is_recorded_as_effective_setting(self):
        protocol.begin("/models/merged", "run3407", seed=3407)
        manifest = protocol.read_json(self.root / "outputs/run3407.manifest.json")
        self.assertEqual(manifest["identity"]["evaluation"]["generation_seed"], 3407)

    def test_baseline_records_null_model_path_and_pinned_weights(self):
        ref = {"records_sha256": "fixed-records", "source_sha256": {},
               "evaluation": {"generation_seed": 42}, "profile": "baseline"}
        with patch.object(protocol, "reference", return_value=ref), \
                patch.object(protocol, "base_model_path", return_value="/models/merged"):
            protocol.begin("/models/merged", "base42")
        manifest = protocol.read_json(self.root / "outputs/base42.manifest.json")
        self.assertIsNone(manifest["identity"]["prediction_model_path"])
        self.assertEqual(manifest["identity"]["model"]["files"], self.model["files"])

    def test_legacy_outputs_cannot_silently_skip_new_generation(self):
        out = self.root / "outputs"
        out.mkdir()
        (out / "run42.jsonl").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no identity manifest"):
            protocol.begin("/models/merged", "run42")

    def test_partial_graded_file_cannot_be_treated_as_complete(self):
        protocol.begin("/models/merged", "run42")
        path = self.root / "outputs/run42.graded.jsonl"
        path.write_text(json.dumps({"id": "test_Art_1", "seed": 42}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            protocol.begin("/models/merged", "run42")
        self.assertTrue(path.exists())


class CompletedRunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / "outputs"
        self.out.mkdir()
        self.addCleanup(self.temp.cleanup)
        self.ref = {"records_sha256": "records", "source_sha256": {},
                    "evaluation": {"generation_seed": 42, "judge_seed": 3407}, "profile": "dpo_e"}
        self.record = {"id": "test_Art_1", "question_type": "multiple-choice", "reference_answer": "A",
                       "options": {"A": "yes", "B": "no"}}
        self.pred = {"id": self.record["id"], "seed": 42, "run_tag": "run42", "model_path": "/model",
                     "model_response": {"raw": "Final answer: A", "output_tokens": 5, "generation_status": "normal_stop"}}
        self.grade = {"id": self.record["id"], "seed": 42, **grade_prediction(self.pred, self.record),
                      "output_tokens": 5, "generation_status": "normal_stop"}
        identity = {"model": {"path": "/model"}, "prediction_model_path": "/model", "tag": "run42",
                    "records_sha256": "records", "source_sha256": {}, "evaluation": self.ref["evaluation"], "profile": "dpo_e"}
        protocol.write_json(self.out / "run42.manifest.json", {"identity": identity})
        self.write("jsonl", self.pred)
        self.write("graded.jsonl", self.grade)
        (self.out / "run42.judge.jsonl").write_text("", encoding="utf-8")
        for item in [patch.object(protocol, "ROOT", self.root), patch.object(protocol, "source_check"),
                     patch.object(protocol, "reference", return_value=self.ref),
                     patch.object(protocol, "records_check", return_value={"records": [self.record]})]:
            item.start(); self.addCleanup(item.stop)

    def write(self, suffix, row):
        (self.out / ("run42." + suffix)).write_text(json.dumps(row)+"\n", encoding="utf-8")

    def test_complete_run_replays_and_writes_verified_result(self):
        protocol.verify("run42")
        result = protocol.read_json(self.out / "run42.verification.json")
        self.assertEqual((result["status"], result["correct"], result["generation_seed"]), ("PASS", 1, 42))

    def test_incorrect_saved_grade_is_rejected(self):
        self.write("graded.jsonl", dict(self.grade, verdict="incorrect"))
        with self.assertRaisesRegex(ValueError, "Regrade mismatch"):
            protocol.verify("run42")

    def test_unneeded_judge_rows_are_rejected(self):
        self.write("judge.jsonl", {"id": self.record["id"], "answer": "A"})
        with self.assertRaisesRegex(ValueError, "Judge output IDs"):
            protocol.verify("run42")


if __name__ == "__main__":
    unittest.main()
