"""Run CPU unit tests with temporary files inside the project workspace."""
from pathlib import Path
import datetime
import platform
import sys
import tempfile
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
scratch = root / "artifacts" / "test-tmp"
scratch.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(scratch)
suite = unittest.defaultTestLoader.discover(str(root / "tests"))
result = unittest.TextTestRunner(verbosity=2).run(suite)
from era_lora.common import write_json
write_json(root / "artifacts" / "unit_tests.json", {
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "python": sys.version, "platform": platform.platform(),
    "tests_run": result.testsRun, "skipped": len(result.skipped),
    "failures": len(result.failures), "errors": len(result.errors),
    "passed": result.wasSuccessful(), "training_executed": False,
    "gpu_evaluation_executed": False,
})
raise SystemExit(0 if result.wasSuccessful() else 1)
