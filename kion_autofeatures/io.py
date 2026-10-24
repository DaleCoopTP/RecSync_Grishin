"""Atomic receipts and content-addressed identities for resumable stages."""
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    from kion.io import sha256
    directory = Path(__file__).parent
    return {p.name: sha256(p) for p in sorted(directory.glob("*.py"))}


def verify_files(root, files):
    from kion.io import sha256
    for name, expected in files.items():
        path = Path(root) / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact missing or checksum mismatch: {name}")


def checked_receipt(path, identity):
    path = Path(path)
    if not path.exists():
        return None
    value = read_json(path)
    if value["identity"] != identity:
        raise ValueError(f"Cannot reuse incompatible stage: {path}")
    verify_files(path.parent, value["files"])
    return value
