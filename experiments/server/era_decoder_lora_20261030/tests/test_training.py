"""Dependency-light safety tests. No model download, GPU run or data learning."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from era_lora import merge, train
from era_lora.common import config_digest, write_json


class Linear:
    pass


class FakeTensor:
    def __init__(self, data=b"one", shape=(1, 3), dtype="torch.float32", requires_grad=True):
        self.data, self.shape, self.dtype, self.requires_grad = data, shape, dtype, requires_grad
        self.grad = None

    def numel(self):
        size = 1
        for dimension in self.shape:
            size *= dimension
        return size


class TinyStructure:
    def __init__(self, layers=2):
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=layers))
        self.modules = [(f"model.language_model.layers.{i}.self_attn.{p}", Linear())
                        for i in range(layers) for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
        self.modules.append(("model.visual.blocks.0.attn.q_proj", Linear()))
        self.parameters = []

    def named_modules(self):
        return self.modules

    def named_parameters(self):
        return self.parameters


def test_config(root: Path) -> dict:
    return {"_config_dir": str(root), "model": {"repo": train.MODEL_REPO, "revision": train.MODEL_REVISION},
            "training": {"output_root": "runs", "micro_batch_size": 1,
                         "gradient_accumulation_steps": 16, "epochs": 2},
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05}}


class DecoderTargetTests(unittest.TestCase):
    def test_exact_decoder_paths_exclude_visual_modules(self):
        targets = train.discover_decoder_targets(TinyStructure(), Linear)
        self.assertEqual(len(targets), 8)
        self.assertTrue(all("language_model" in item for item in targets))
        self.assertTrue(all("visual" not in item for item in targets))

    def test_missing_projection_is_fatal(self):
        model = TinyStructure()
        model.modules.pop(0)
        with self.assertRaisesRegex(RuntimeError, "Decoder target mismatch"):
            train.discover_decoder_targets(model, Linear)

    def test_wrong_module_type_is_fatal(self):
        model = TinyStructure()
        model.modules[0] = (model.modules[0][0], object())
        with self.assertRaisesRegex(RuntimeError, "Decoder target mismatch"):
            train.discover_decoder_targets(model, Linear)

    def test_trainable_matrix_a_and_b_for_every_target(self):
        model = TinyStructure()
        targets = train.discover_decoder_targets(model, Linear)
        model.parameters = [(f"base_model.model.{target}.lora_{matrix}.default.weight", FakeTensor())
                            for target in targets for matrix in ("A", "B")]
        model.parameters.append(("model.visual.patch_embed.weight", FakeTensor(requires_grad=False)))
        self.assertEqual(len(train.assert_trainables(model, targets)), 16)
        model.parameters[-1][1].requires_grad = True
        with self.assertRaisesRegex(RuntimeError, "Unexpected trainable parameter"):
            train.assert_trainables(model, targets)

    def test_duplicate_or_missing_lora_matrix_is_fatal(self):
        model = TinyStructure(layers=1)
        targets = train.discover_decoder_targets(model, Linear)
        model.parameters = [(f"base_model.model.{target}.lora_A.default.weight", FakeTensor())
                            for target in targets for _ in range(2)]
        with self.assertRaisesRegex(RuntimeError, "Missing or duplicate"):
            train.assert_trainables(model, targets)


class FingerprintTests(unittest.TestCase):
    def test_base_index_must_match_verified_shard_contents(self):
        class SafeOpen:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def keys(self):
                return ["frozen.weight"]

        fake = SimpleNamespace(safe_open=lambda *args, **kwargs: SafeOpen())
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"safetensors": fake}):
            root = Path(directory)
            shard = "model-00001-of-00001.safetensors"
            (root / shard).write_bytes(b"synthetic; mocked parser")
            write_json(root / "model.safetensors.index.json", {"weight_map": {"frozen.weight": shard}})
            train.validate_base_weight_index(root, {shard})
            write_json(root / "model.safetensors.index.json", {"weight_map": {"other.weight": shard}})
            with self.assertRaisesRegex(RuntimeError, "index does not describe"):
                train.validate_base_weight_index(root, {shard})
            (root / "model.safetensors").write_bytes(b"unrecognized override")
            with self.assertRaisesRegex(RuntimeError, "Unexpected additional"):
                train.validate_base_weight_index(root, {shard})

    def test_tensor_hash_ignores_dict_order_but_includes_shape_dtype_bytes(self):
        encode = lambda tensor: tensor.data
        first, second = FakeTensor(b"first"), FakeTensor(b"second")
        reference = train.state_digest({"a": first, "b": second}, encode)
        self.assertEqual(reference, train.state_digest({"b": second, "a": first}, encode))
        second.shape = (3, 1)
        self.assertNotEqual(reference, train.state_digest({"a": first, "b": second}, encode))
        second.shape = (1, 3)
        second.dtype = "torch.bfloat16"
        self.assertNotEqual(reference, train.state_digest({"a": first, "b": second}, encode))

    def test_real_cpu_bfloat16_hash_and_first_gradient_guard(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional CPU torch is unavailable")
        value = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).transpose(0, 1)
        self.assertEqual(train.tensor_bytes(value), train.tensor_bytes(value.contiguous()))
        # This is a tiny synthetic gradient check, with no optimizer or training data.
        frozen = torch.nn.Parameter(torch.ones(3, 3), requires_grad=False)
        a = torch.nn.Parameter(torch.ones(2, 3))
        b = torch.nn.Parameter(torch.zeros(3, 2))
        (frozen + b @ a).sum().backward()
        model = SimpleNamespace(named_parameters=lambda: [("base", frozen), ("lora_A", a), ("lora_B", b)])
        self.assertTrue(train.validate_gradient_ownership(model)["all_gradients_finite"])
        self.assertEqual(torch.count_nonzero(a.grad).item(), 0)
        frozen.grad = torch.ones_like(frozen)
        with self.assertRaisesRegex(RuntimeError, "Invalid first gradients"):
            train.validate_gradient_ownership(model)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "plain"
        self.checkpoint = self.output / "checkpoint-13"
        self.checkpoint.mkdir(parents=True)
        self.identity = {"arm": "plain", "config_sha256": "c", "data_sha256": "d", "source_sha256": {"train.py": "s"}}
        write_json(self.output / "run_manifest.json", {"status": "failed", "identity": self.identity})
        for name in ("adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt",
                     "rng_state.pth", "trainer_state.json"):
            (self.checkpoint / name).write_bytes(name.encode())
        write_json(self.checkpoint / "run_identity.json", self.identity)
        self.refresh_integrity()

    def refresh_integrity(self):
        write_json(self.checkpoint / "checkpoint_integrity.json",
                   train.file_manifest(self.checkpoint, exclude=("checkpoint_integrity.json",)))

    def test_valid_explicit_resume_keeps_optimizer_and_rng(self):
        checkpoint, manifest = train.check_resume(self.output, str(self.checkpoint), self.identity)
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(manifest["status"], "failed")

    def test_nonempty_run_without_resume_is_not_overwritten(self):
        with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
            train.check_resume(self.output, None, self.identity)

    def test_other_arm_or_changed_configuration_cannot_resume(self):
        for changed in (dict(self.identity, arm="era"), dict(self.identity, config_sha256="changed")):
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                train.check_resume(self.output, str(self.checkpoint), changed)
        with self.assertRaisesRegex(RuntimeError, "belong directly"):
            train.check_resume(self.root / "era", str(self.checkpoint), self.identity)

    def test_missing_rng_and_corrupted_optimizer_are_rejected(self):
        (self.checkpoint / "rng_state.pth").unlink()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            train.check_resume(self.output, str(self.checkpoint), self.identity)
        (self.checkpoint / "rng_state.pth").write_bytes(b"rng")
        self.refresh_integrity()
        (self.checkpoint / "optimizer.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            train.check_resume(self.output, str(self.checkpoint), self.identity)

    def test_completed_or_exported_adapter_cannot_be_overwritten(self):
        (self.output / "adapter").mkdir()
        with self.assertRaisesRegex(RuntimeError, "cannot be overwritten"):
            train.check_resume(self.output, str(self.checkpoint), self.identity)


class MergeTests(unittest.TestCase):
    def test_merge_allows_only_decoder_target_weight_changes(self):
        target = "model.language_model.layers.0.self_attn.q_proj"
        before = {target + ".weight": "base", "model.visual.weight": "fixed"}
        after = {target + ".weight": "trained", "model.visual.weight": "fixed"}
        report = merge.validate_merged_state(before, after, [target])
        self.assertTrue(report["all_non_target_tensors_identical"])
        after["model.visual.weight"] = "unexpected"
        with self.assertRaisesRegex(RuntimeError, "outside decoder"):
            merge.validate_merged_state(before, after, [target])

    def test_missing_update_or_changed_architecture_is_rejected(self):
        target = "model.language_model.layers.0.self_attn.q_proj"
        before = {target + ".weight": "base"}
        with self.assertRaisesRegex(RuntimeError, "identical to baseline"):
            merge.validate_merged_state(before, before.copy(), [target])
        with self.assertRaisesRegex(RuntimeError, "state keys"):
            merge.validate_merged_state(before, {"other": "changed"}, [target])

    def test_processor_files_preserve_exact_bytes_and_exclude_base_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "snapshot", Path(directory) / "merged"
            source.mkdir()
            destination.mkdir()
            for name in ("config.json", "tokenizer_config.json", "preprocessor_config.json", "chat_template.jinja"):
                (source / name).write_bytes((name + "\r\n  preserve spacing").encode())
            (source / "model.safetensors").write_bytes(b"base weights")
            (source / "model.safetensors.index.json").write_bytes(b"base index")
            hashes = merge.copy_processor_files(source, destination)
            self.assertEqual(len(hashes), 4)
            for name in hashes:
                self.assertEqual((source / name).read_bytes(), (destination / name).read_bytes())
            self.assertFalse((destination / "model.safetensors").exists())
            self.assertFalse((destination / "model.safetensors.index.json").exists())

    def test_merge_requires_completed_matching_adapter_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = test_config(root)
            output = train.output_directory(config, "plain")
            adapter = output / "adapter"
            adapter.mkdir(parents=True)
            (adapter / "adapter_model.safetensors").write_bytes(b"synthetic; never loaded")
            identity = {"arm": "plain", "config_sha256": config_digest(config),
                        "base_repo": train.MODEL_REPO, "base_revision": train.MODEL_REVISION}
            manifest = {"status": "complete", "identity": identity,
                        "adapter_files_sha256": train.file_manifest(adapter)}
            write_json(output / "run_manifest.json", manifest)
            self.assertEqual(merge.validate_adapter_run(config, "plain")[0], output.resolve())
            (adapter / "adapter_model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                merge.validate_adapter_run(config, "plain")

    def test_saved_weights_allow_only_genuinely_tied_missing_aliases(self):
        class SafeOpen:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def keys(self):
                return ["embed.weight", "layer.weight"]

            def get_slice(self, name):
                return SimpleNamespace(get_dtype=lambda: "BF16")

        fake = SimpleNamespace(safe_open=lambda *args, **kwargs: SafeOpen())
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"safetensors": fake}):
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"synthetic; mocked parser")
            expected = {"embed.weight", "head.weight", "layer.weight"}
            report = merge.validate_saved_weights(root, expected, [["embed.weight", "head.weight"]])
            self.assertEqual(report["omitted_tied_aliases"], ["head.weight"])
            with self.assertRaisesRegex(RuntimeError, "differs"):
                merge.validate_saved_weights(root, expected)


class PlanningTests(unittest.TestCase):
    def test_train_cli_without_execute_never_trains_or_downloads(self):
        config = test_config(Path.cwd())
        with patch.object(train, "load_config", return_value=config), \
             patch("era_lora.data.load_paired_records", return_value=([{}] * 200, {"items": 200})), \
             patch.object(train, "execute_training") as execute, \
             patch.object(train, "resolve_model_snapshot") as snapshot, \
             patch("builtins.print"):
            self.assertEqual(train.main(["--config", "unused.json", "--arm", "era"]), 0)
        execute.assert_not_called()
        snapshot.assert_not_called()

    def test_merge_cli_without_execute_never_loads_or_writes(self):
        config = test_config(Path.cwd())
        with patch.object(merge, "load_config", return_value=config), \
             patch.object(merge, "execute_merge") as execute, patch("builtins.print"):
            self.assertEqual(merge.main(["--config", "unused.json", "--arm", "plain"]), 0)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
