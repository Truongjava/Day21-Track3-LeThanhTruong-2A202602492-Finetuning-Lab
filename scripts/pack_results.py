#!/usr/bin/env python3
"""Zip the graded artifacts, and download them when running on Colab.

`results/` is what the grader cross-checks every number in REPORT.md against, and a
Colab VM is destroyed the moment its session ends. So this must run before the tab
closes -- every time, and whether or not the pipeline reached the end.

The first full run of this lab took 66 minutes and produced nothing usable, because
the download was a snippet in a document the student had to remember to run. A step
that must happen for the work to count does not belong in a document.

Called from the notebook's `finally`, and safe to re-run by hand:

    python scripts/pack_results.py
"""
from __future__ import annotations

import pathlib
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = "lab21_2A202602492.zip"


def build_zip(dest: pathlib.Path | None = None) -> pathlib.Path:
    """Zip `results/` plus the `correct` adapter, which is what the rubric asks to ship."""
    dest = dest or ROOT / NAME
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT / "results").rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(ROOT))
        # adapters/correct is required by rubric Option A. Deliberately NOT the whole
        # adapters/ directory: adapters/merged is ~9.3 GB.
        adapter = ROOT / "adapters" / "correct"
        if adapter.exists():
            for path in sorted(adapter.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(ROOT))
    return dest


def main() -> int:
    results = ROOT / "results"
    n_files = sum(1 for p in results.rglob("*") if p.is_file()) if results.exists() else 0
    if not n_files:
        print("nothing to pack: results/ is empty")
        return 1

    path = build_zip()
    print(f"packed {n_files} files from results/ -> {path.name} "
          f"({path.stat().st_size / 1024 ** 2:.1f} MB)")

    try:
        from google.colab import files
    except ImportError:
        print("not on Colab — the zip is at", path)
        return 0

    try:
        files.download(str(path))
        print("download started")
    except Exception as exc:                       # browser blocked it, usually
        print(f"could not auto-download: {exc}")
        print(f"download {path.name} by hand from the file pane on the left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
