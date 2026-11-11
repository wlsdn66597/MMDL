"""Small dependency-free helpers shared by preparation and execution commands."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

BASE_REPO = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EVALUATION_SEEDS = [3407, 42, 123, 2026, 9876]


def reject_unpinned_processor_overrides(path: str | Path) -> None:
    """Reject optional loader overrides absent from the audited base snapshot.

    Known baseline artifacts may be Hugging Face cache symlinks; their bytes
    are checked separately. Documentation and run manifests are unrestricted.
    """
    root = Path(path)
    for name in ("special_tokens_map.json", "added_tokens.json", "processor_config.json",
                 "chat_template.jinja", "chat_templates"):
        candidate = root / name
        if candidate.exists() or candidate.is_symlink():
            raise ValueError(f"Unexpected unpinned processor override: {name}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def config_digest(config: dict) -> str:
    return canonical_json_hash({key: value for key, value in config.items() if not key.startswith("_")})


def resolve_path(config: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_config_dir"]) / path
    return path.resolve()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically: a crash must not leave a manifest appearing complete.
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment_versions(names: list[str] | tuple[str, ...]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema_version")
    if config["model"]["repo"] != BASE_REPO or config["model"]["revision"] != BASE_REVISION:
        raise ValueError("This pilot must start from the pinned baseline model revision")
    if config["training"]["micro_batch_size"] != 1:
        raise ValueError("The audited multimodal collator requires micro_batch_size=1")
    if config["evaluation"]["seeds"] != EVALUATION_SEEDS:
        raise ValueError("The existing five evaluation seeds and order must be preserved")
    if config["training"]["min_pixels"] > config["training"]["max_pixels"]:
        raise ValueError("min_pixels exceeds max_pixels")
    for key in ("epochs", "gradient_accumulation_steps", "max_seq_length"):
        if int(config["training"][key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    config["_config_dir"] = str(path.parent)
    config["_config_path"] = str(path)
    return config
