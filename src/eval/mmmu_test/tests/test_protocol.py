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

    def test_generation_seed_is_part_of_resume_identity(self):
        tag = "run3407"
        protocol.begin("/models/merged", tag, 3407)
        out = self.root / "outputs"
        manifest = protocol.read_json(out / f"{tag}.manifest.json")
        self.assertEqual(manifest["identity"]["evaluation"]["generation_seed"], 3407)
        row = {"id": "test_Art_1", "seed": 3407, "model_path": "/models/merged"}
        predictions = out / f"{tag}.jsonl"
        predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
        protocol.begin("/models/merged", tag, 3407)
        with self.assertRaisesRegex(ValueError, "different checkpoint/protocol"):
            protocol.begin("/models/merged", tag, 42)
        row["seed"] = 42
        predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Mixed generation seeds"):
            protocol.begin("/models/merged", tag, 3407)

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


if __name__ == "__main__":
    unittest.main()
