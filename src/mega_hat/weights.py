"""Verify the published pretrained MegaPose model files before loading them."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path


def verify_weights(root):
    expected = json.loads(files("mega_hat").joinpath("weights.json").read_text())
    for relative, digest in expected.items():
        path = Path(root) / relative
        if not path.is_file():
            raise ValueError(f"Missing pretrained model file: {path}")
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(block)
        if sha.hexdigest() != digest:
            raise ValueError(f"Model file differs from the published checkpoint: {path}")
