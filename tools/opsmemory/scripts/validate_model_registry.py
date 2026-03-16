"""Validate the OpsMemory model registry YAML.

Checks that:
- The YAML file is parseable.
- Every entry has required fields.
- embedding_dim is present when supports_embeddings is True.
- No duplicate model IDs exist.

Usage
-----
    python tools/opsmemory/scripts/validate_model_registry.py

Exit code 0 on success, non-zero on validation failure.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

REGISTRY_PATH = Path(__file__).parent.parent / "providers" / "model_registry.yaml"

REQUIRED_FIELDS = [
    "id",
    "provider",
    "supports_generation",
    "supports_embeddings",
    "supports_structured",
    "supports_rerank",
    "mcp_safe",
    "production_approved",
]


def validate(path: Path = REGISTRY_PATH) -> List[str]:
    """Return a list of validation error messages (empty = all OK)."""
    errors: List[str] = []

    if not path.exists():
        return [f"Registry file not found: {path}"]

    try:
        with open(path, encoding="utf-8") as fh:
            data: Any = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        return [f"YAML parse error: {exc}"]

    if not isinstance(data, dict) or "models" not in data:
        return ["Registry must be a mapping with a 'models' key."]

    models: List[Dict] = data["models"]
    if not isinstance(models, list):
        return ["'models' must be a list."]

    seen_ids: set = set()

    for i, entry in enumerate(models):
        prefix = f"models[{i}] (id={entry.get('id', '<missing>')})"

        for field in REQUIRED_FIELDS:
            if field not in entry:
                errors.append(f"{prefix}: missing required field '{field}'")

        entry_id = entry.get("id")
        if entry_id:
            if entry_id in seen_ids:
                errors.append(f"{prefix}: duplicate id '{entry_id}'")
            seen_ids.add(entry_id)

        if entry.get("supports_embeddings") and not entry.get("embedding_dim"):
            errors.append(
                f"{prefix}: supports_embeddings=true but embedding_dim is missing or null"
            )

    return errors


def main() -> None:
    errors = validate()
    if errors:
        print(f"Model registry validation FAILED ({len(errors)} error(s)):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        models_count = 0
        with open(REGISTRY_PATH, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        models_count = len(data.get("models", []))
        print(f"Model registry valid. {models_count} model(s) registered.")
        sys.exit(0)


if __name__ == "__main__":
    main()
