#!/usr/bin/env python3
"""Static validation for the repository's Jupyter notebooks.

This intentionally does not execute TensorFlow training. It checks notebook structure,
Python syntax, stale outputs, and non-portable path patterns. If Ruff is installed,
it also runs its syntax/undefined-name rules against all code cells.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
NOTEBOOKS = sorted(REPO.glob("*.ipynb"))
BANNED_PATTERNS = {
    "os.chdir": re.compile(r"\bos\.chdir\s*\("),
    "Kaggle absolute path": re.compile(r"/kaggle/input/|/content/", re.IGNORECASE),
    "Windows absolute path": re.compile(r"[A-Za-z]:\\\\"),
}


def extract_code(notebook: Path) -> str:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    if payload.get("nbformat", 0) < 4 or not isinstance(payload.get("cells"), list):
        raise ValueError("invalid notebook format")
    chunks: list[str] = []
    for index, cell in enumerate(payload["cells"]):
        if cell.get("cell_type") == "code":
            if cell.get("outputs"):
                raise ValueError(f"code cell {index} contains stale outputs")
            if cell.get("execution_count") is not None:
                raise ValueError(f"code cell {index} has a stale execution count")
            source = "".join(cell.get("source", []))
            ast.parse(source, filename=f"{notebook.name}:cell-{index}")
            chunks.append(f"# notebook cell {index}\n{source}")
    return "\n\n".join(chunks) + "\n"


def main() -> int:
    failures: list[str] = []
    if not NOTEBOOKS:
        failures.append("no notebooks found")
    with tempfile.TemporaryDirectory(prefix="plant-notebook-validation-") as temp_dir:
        extracted: list[Path] = []
        for notebook in NOTEBOOKS:
            try:
                raw = notebook.read_text(encoding="utf-8")
                for label, pattern in BANNED_PATTERNS.items():
                    if pattern.search(raw):
                        failures.append(f"{notebook.name}: non-portable {label}")
                target = Path(temp_dir) / f"{notebook.stem.replace(' ', '_')}.py"
                target.write_text(extract_code(notebook), encoding="utf-8")
                extracted.append(target)
                print(f"PASS {notebook.name}: JSON, AST, outputs, and paths")
            except (OSError, SyntaxError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"{notebook.name}: {exc}")
        ruff = shutil.which("ruff")
        if ruff and extracted:
            result = subprocess.run(
                [ruff, "check", "--select", "E9,F821", *map(str, extracted)],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode:
                failures.append("Ruff failed for notebook code:\n" + result.stdout + result.stderr)
            else:
                print("PASS Ruff E9/F821")
        elif not ruff:
            print("SKIP Ruff: command not installed")
    if failures:
        print("\n".join(f"FAIL: {failure}" for failure in failures), file=sys.stderr)
        return 1
    print("All notebook static checks passed; no model training was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
