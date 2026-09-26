"""Fetch the source revisions used by the Mega-HAT adapters."""

import argparse
import subprocess
from pathlib import Path

SOURCES = {
    "megapose6d": (
        "https://github.com/megapose6d/megapose6d.git",
        "f3b8e1247f133f3d098833a251b8f2d744c03e1f",
    ),
    "DROID-SLAM": (
        "https://github.com/princeton-vl/DROID-SLAM.git",
        "2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99",
    ),
}


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, revision) in SOURCES.items():
        target = root / name
        patch = Path(__file__).resolve().parent.parent / "patches/megapose-runtime.patch"
        already_patched = False
        if target.exists():
            current = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
            ).strip()
            changes = subprocess.check_output(
                ["git", "-C", str(target), "diff", "--binary", "HEAD"]
            )
            already_patched = name == "megapose6d" and changes == patch.read_bytes()
            if current != revision or (changes and not already_patched):
                raise RuntimeError(
                    f"{target} differs from the expected source. Use a fresh directory."
                )
        else:
            subprocess.run(["git", "init", str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "remote", "add", "origin", url], check=True)
            subprocess.run(
                ["git", "-C", str(target), "fetch", "--depth", "1", "origin", revision], check=True
            )
            subprocess.run(
                ["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True
            )
        if name == "megapose6d" and not already_patched:
            subprocess.run(["git", "-C", str(target), "apply", str(patch)], check=True)
        if name == "DROID-SLAM":
            subprocess.run(
                ["git", "-C", str(target), "submodule", "update", "--init", "--recursive"],
                check=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("third_party"))
    fetch(parser.parse_args().out.resolve())
