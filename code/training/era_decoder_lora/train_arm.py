"""Train the plain target with the shared decoder LoRA configuration."""
from pathlib import Path
import sys

from era_lora.train import main


if __name__ == "__main__":
    config = Path(__file__).resolve().parent / "configs" / "paired_4epoch.json"
    raise SystemExit(main(["--config", str(config), *sys.argv[1:], "--arm", "plain"]))
