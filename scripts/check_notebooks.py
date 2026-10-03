#!/usr/bin/env python3
"""Validate notebook structure and syntax without installing notebook/ML packages."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    notebooks = sorted(ROOT.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found at repository root.", file=sys.stderr)
        return 1

    failed = False
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Do not echo file content or notebook outputs in diagnostics.
            print(f"{path.name}: invalid JSON ({type(exc).__name__}).", file=sys.stderr)
            failed = True
            continue

        if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
            print(f"{path.name}: missing notebook cells list.", file=sys.stderr)
            failed = True
            continue

        code_cells = 0
        for index, cell in enumerate(notebook["cells"]):
            if not isinstance(cell, dict):
                print(f"{path.name}: cell {index} is not an object.", file=sys.stderr)
                failed = True
                continue
            if cell.get("cell_type") != "code":
                continue

            code_cells += 1
            if cell.get("outputs"):
                print(f"{path.name}: cell {index} contains saved outputs; clear them before committing.", file=sys.stderr)
                failed = True
            if cell.get("execution_count") is not None:
                print(f"{path.name}: cell {index} has an execution count; clear it before committing.", file=sys.stderr)
                failed = True

            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if not isinstance(source, str):
                print(f"{path.name}: cell {index} has invalid source type.", file=sys.stderr)
                failed = True
                continue
            try:
                ast.parse(source, filename=f"{path.name}:cell-{index}")
            except SyntaxError as exc:
                print(
                    f"{path.name}: cell {index} has invalid Python syntax at line {exc.lineno}.",
                    file=sys.stderr,
                )
                failed = True

        print(f"{path.name}: checked {len(notebook['cells'])} cells ({code_cells} code cells).")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
