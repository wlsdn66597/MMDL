import copy
import json
from pathlib import Path
import tempfile
import unittest

from era_lora.common import config_digest, load_config, resolve_path, write_json

ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_digest_ignores_machine_path_metadata(self):
        config = load_config(ROOT / "configs/pilot.json")
        other = copy.deepcopy(config)
        other["_config_dir"] = "/another/machine"
        self.assertEqual(config_digest(config), config_digest(other))
        other["training"]["seed"] += 1
        self.assertNotEqual(config_digest(config), config_digest(other))

    def test_relative_paths_follow_config_not_cwd(self):
        config = load_config(ROOT / "configs/pilot.json")
        self.assertEqual(resolve_path(config, "../baseline"), ROOT / "baseline")

    def test_changed_base_revision_fails(self):
        config = json.loads((ROOT / "configs/pilot.json").read_text())
        config["model"]["revision"] = "main"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            write_json(path, config)
            with self.assertRaisesRegex(ValueError, "pinned baseline"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
