from __future__ import annotations

import json
from pathlib import Path

import pytest

from validate_notebooks import NOTEBOOKS, extract_code


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_repository_notebook_code_is_parseable_and_clean(notebook: Path) -> None:
    code = extract_code(notebook)
    assert code.endswith("\n")
    assert "notebook cell" in code


def test_extract_code_rejects_stale_output(tmp_path: Path) -> None:
    notebook = tmp_path / "stale.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "metadata": {},
                        "outputs": [{"output_type": "stream", "text": ["old"]}],
                        "source": ["print('old')\n"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stale outputs"):
        extract_code(notebook)


def test_extract_code_rejects_execution_count(tmp_path: Path) -> None:
    notebook = tmp_path / "executed.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 3,
                        "metadata": {},
                        "outputs": [],
                        "source": ["x = 1\n"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="execution count"):
        extract_code(notebook)
