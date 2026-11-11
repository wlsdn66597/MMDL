"""CPU-only tests: no model weights, transformers, torch, or GPU required."""
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from era_lora.data import (AssistantOnlyCollator, PairedSFTDataset,
                          assistant_label_ids, load_paired_records,
                          processor_source, validate_encoded_pair, validate_processor_files)
from era_lora.common import sha256_file


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return copy.deepcopy(self.values)

    def new_tensor(self, values):
        return FakeTensor(values)


class FakeTokenizer:
    unk_token_id = 999
    token_ids = {"<|im_end|>": 9, "<|im_start|>": 10, "<|image_pad|>": 3,
                 "<|vision_start|>": 11, "<|vision_end|>": 12, "<|video_pad|>": 13}

    def convert_tokens_to_ids(self, token):
        return self.token_ids.get(token, self.unk_token_id)

    def decode(self, values, **kwargs):
        return "".join("\n" if v == 8 else "x" for v in values)


class FakeProcessor:
    def __init__(self):
        self.image_processor = SimpleNamespace(patch_size=16)
        self.tokenizer = FakeTokenizer()
        self.seen_prompts = []
        self.grid_changes_with_target = False
        self.prefix_changes = False

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.seen_prompts.append(messages[0]["content"][1]["text"])
        prefix = "PROMPT<assistant>"
        return prefix if add_generation_prompt else prefix + messages[-1]["content"] + "<end>\n"

    def __call__(self, *, text, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["do_resize"] is False
        assert kwargs["padding"] is False
        has_target = text[0] != "PROMPT<assistant>"
        tokens = [1, 3, 4]
        if has_target:
            tokens += [5, 6, 9, 8]
            if self.prefix_changes:
                tokens[0] = 2
        grid = [[1, 2, 4 if has_target and self.grid_changes_with_target else 2]]
        return {"input_ids": FakeTensor([tokens]), "attention_mask": FakeTensor([[1] * len(tokens)]),
                "image_grid_thw": FakeTensor(grid), "pixel_values": FakeTensor([[1.0]])}


class PairDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "image.png").write_bytes(b"test-image-bytes")
        self.prompt = 'Question: What?\nOptions:\nA. One\nB. Two\nPlease select the correct answer from the options above.\nOn the last line, write "Final answer: LETTER", replacing LETTER with one uppercase option letter.'
        self.plain = dict(sample_id="sample-1", visual_group_id="group-1", image_path="image.png",
                          input_prompt=self.prompt, response="There is one object.\nFinal answer: A")
        self.era = dict(self.plain, response="Evidence: One object.\nRule: Count objects.\nApplication: One is A.\nFinal answer: A")
        self.config = {"_config_dir": str(self.root), "dataset": {"root": ".", "plain_jsonl": "plain.jsonl",
                       "era_jsonl": "era.jsonl", "validation_status": "status.json", "expected_items": 1},
                       "training": {"min_pixels": 1003520, "max_pixels": 4014080,
                                    "max_seq_length": 8192, "micro_batch_size": 1}}
        (self.root / "status.json").write_text(json.dumps({"dataset_items": 1, "training_release_ready": False}), encoding="utf-8")
        self.write_rows()

    def write_rows(self):
        for name, row in (("plain", self.plain), ("era", self.era)):
            (self.root / f"{name}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    def collator(self, processor=None, config=None):
        image = SimpleNamespace(size=(640, 480))
        def vision(messages, **kwargs):
            self.assertEqual(kwargs, {"image_patch_size": 16, "return_video_kwargs": True, "return_video_metadata": True})
            return [image], None, {}
        return AssistantOnlyCollator(processor or FakeProcessor(), config or self.config,
                                     vision_processor=vision, image_loader=lambda path: image)

    def test_pair_integrity_and_status_are_recorded_without_approval_gate(self):
        records, manifest = load_paired_records(self.config)
        self.assertEqual(records[0]["input_prompt"], self.prompt)
        self.assertFalse(manifest["source_validation_status"]["training_release_ready"])
        self.assertEqual(len(manifest["ordered_pair_sha256"]), 64)
        self.assertEqual(PairedSFTDataset(records, "era")[0]["response"], self.era["response"])

    def test_mismatched_prompt_or_answer_fails(self):
        for field, value in (("input_prompt", self.prompt + "\nUse ERA."),
                             ("response", self.era["response"].replace("Final answer: A", "Final answer: B"))):
            with self.subTest(field=field):
                original = self.era[field]
                self.era[field] = value
                self.write_rows()
                with self.assertRaises(ValueError):
                    load_paired_records(self.config)
                self.era[field] = original

    def test_era_instruction_in_both_inputs_fails(self):
        self.plain["input_prompt"] += "\nPlease use ERA for the solution."
        self.era["input_prompt"] = self.plain["input_prompt"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "ERA instruction"):
            load_paired_records(self.config)

    def test_image_escape_and_declared_hash_mismatch_fail(self):
        for value in ("../image.png", "C:\\outside\\image.png", "/outside/image.png"):
            with self.subTest(path=value):
                self.plain["image_path"] = self.era["image_path"] = value
                self.write_rows()
                with self.assertRaises(ValueError):
                    load_paired_records(self.config)
        self.plain["image_path"] = self.era["image_path"] = "image.png"
        self.plain["image_sha256"] = "0" * 64
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_paired_records(self.config)

    def test_duplicate_and_reordered_ids_fail(self):
        self.config["dataset"]["expected_items"] = 2
        for name, row in (("plain", self.plain), ("era", self.era)):
            (self.root / f"{name}.jsonl").write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_paired_records(self.config)
        self.era["sample_id"] = "another-id"
        self.config["dataset"]["expected_items"] = 1
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "ordered sample_id"):
            load_paired_records(self.config)

    def test_reserved_chat_tokens_and_nonterminal_answers_fail(self):
        for ending in ("\nmore text", "\nFinal answer: A", "<|im_end|>"):
            with self.subTest(ending=ending):
                original = self.era["response"]
                self.era["response"] += ending
                self.write_rows()
                with self.assertRaises(ValueError):
                    load_paired_records(self.config)
                self.era["response"] = original

    def test_mask_supervises_body_and_end_only_and_keeps_frozen_input(self):
        records, _ = load_paired_records(self.config)
        processor = FakeProcessor()
        collator = self.collator(processor)
        batch, report = collator.encode(PairedSFTDataset(records, "era")[0])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, -100, 5, 6, 9, -100]])
        self.assertEqual(report["supervised_tokens"], 3)
        self.assertTrue(report["assistant_end_in_loss"])
        self.assertEqual(processor.seen_prompts, [self.prompt, self.prompt])
        self.assertNotIn("sample_id", batch)
        self.assertNotIn("response", batch)
        pair = validate_encoded_pair(records[0], processor, self.config, collator=collator)
        self.assertTrue(pair["paired_prefix_and_image_grid_equal"])

    def test_no_truncation_or_silent_grid_prefix_changes(self):
        records, _ = load_paired_records(self.config)
        feature = PairedSFTDataset(records, "plain")[0]
        small = copy.deepcopy(self.config)
        small["training"]["max_seq_length"] = 5
        with self.assertRaisesRegex(ValueError, "no truncation"):
            self.collator(config=small).encode(feature)
        for attribute in ("grid_changes_with_target", "prefix_changes"):
            with self.subTest(attribute=attribute):
                processor = FakeProcessor()
                setattr(processor, attribute, True)
                with self.assertRaises(ValueError):
                    self.collator(processor).encode(feature)

    def test_microbatch_must_be_one(self):
        with self.assertRaisesRegex(ValueError, "one example"):
            self.collator()([])
        bad = copy.deepcopy(self.config)
        bad["training"]["micro_batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "micro_batch_size"):
            self.collator(config=bad)

    def test_end_token_is_required_and_padding_never_supervised(self):
        self.assertEqual(assistant_label_ids([1, 3, 5, 9, 0], [1, 3], [1, 1, 1, 1, 0], 9),
                         [-100, -100, 5, 9, -100])
        with self.assertRaisesRegex(ValueError, "end token"):
            assistant_label_ids([1, 3, 5], [1, 3], [1, 1, 1], 9)
        with self.assertRaisesRegex(ValueError, "padding"):
            assistant_label_ids([1, 3, 5, 9], [1, 3], [1, 1, 1, 0], 9)

    def test_processor_auxiliary_files_must_match_baseline(self):
        config = dict(self.config, model={"local_path": str(self.root), "repo": "pinned/model", "revision": "fixed"})
        artifact = self.root / "tokenizer.json"
        artifact.write_text("frozen tokenizer", encoding="utf-8")
        model_config = self.root / "config.json"
        model_config.write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
        with patch("era_lora.evaluate.AUX_HASHES", {"tokenizer.json": sha256_file(artifact)}), \
                patch("era_lora.evaluate.BASE_CONFIG_SHA256", sha256_file(model_config)):
            source, provenance = processor_source(config)
            self.assertEqual(source, self.root)
            self.assertTrue(provenance["verified_against_baseline"])
            artifact.write_text("changed tokenizer", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from the frozen baseline"):
                processor_source(config)

    def test_extra_processor_overrides_are_rejected(self):
        config = dict(self.config, model={"local_path": str(self.root), "repo": "pinned/model", "revision": "fixed"})
        for name in ("special_tokens_map.json", "added_tokens.json", "processor_config.json", "chat_template.jinja", "chat_templates"):
            with self.subTest(name=name):
                target = self.root / name
                if name == "chat_templates":
                    target.mkdir()
                else:
                    target.write_text("changed processor", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "override"):
                    processor_source(config)
                if target.is_dir():
                    target.rmdir()
                else:
                    target.unlink()

    def test_processor_model_metadata_is_also_pinned(self):
        artifact = self.root / "config.json"
        artifact.write_text("original model config", encoding="utf-8")
        digest = sha256_file(artifact)
        with patch("era_lora.evaluate.AUX_HASHES", {}), patch("era_lora.evaluate.BASE_CONFIG_SHA256", digest):
            self.assertEqual(validate_processor_files(self.root), {"config.json": digest})
            artifact.write_text("altered model config", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config.json"):
                validate_processor_files(self.root)


class RealPilotIntegrityTests(unittest.TestCase):
    def test_current_local_pilot_pair(self):
        root = Path(__file__).resolve().parents[4] / "results" / "era_pilot_hf_private_20261028"
        if not root.is_dir():
            root = Path(__file__).resolve().parents[1] / "data" / "pilot200"
        if not root.is_dir():
            self.skipTest("Pilot package is absent from both repository results and deployed data/pilot200")
        config = {"_config_dir": str(root), "dataset": {"root": ".", "plain_jsonl": "exports/train_plain.jsonl",
                  "era_jsonl": "exports/train_era.jsonl", "validation_status": "validation_status.json", "expected_items": 200}}
        records, manifest = load_paired_records(config)
        self.assertEqual(len(records), 200)
        self.assertEqual(manifest["visual_groups"], 197)
        self.assertFalse(manifest["source_validation_status"]["training_release_ready"])


if __name__ == "__main__":
    unittest.main()
