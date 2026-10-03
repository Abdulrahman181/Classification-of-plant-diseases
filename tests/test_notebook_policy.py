"""Fast, dependency-free checks for repository notebook policy and split hygiene."""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_notebook(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def code_cells(notebook):
    return [
        (index, "".join(cell.get("source", [])))
        for index, cell in enumerate(notebook["cells"])
        if cell.get("cell_type") == "code"
    ]


class NotebookPolicyTests(unittest.TestCase):
    def test_all_notebooks_are_source_only(self):
        paths = sorted(ROOT.glob("*.ipynb"))
        self.assertTrue(paths, "expected at least one notebook")
        for path in paths:
            with self.subTest(notebook=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                for cell in notebook["cells"]:
                    if cell.get("cell_type") == "code":
                        self.assertEqual(cell.get("outputs", []), [])
                        self.assertIsNone(cell.get("execution_count"))

    def test_pdd_uses_stratified_three_way_split_and_validation_for_selection(self):
        notebook = load_notebook("PDD_Classification.ipynb")
        cells = code_cells(notebook)
        sources = [source for _, source in cells]
        split_cells = [s for s in sources if "x_holdout" in s and "stratify=class_ids" in s]
        self.assertEqual(len(split_cells), 1)
        split = split_cells[0]
        self.assertIn("x_valid, x_test", split)
        self.assertIn("random_state=42", split)

        fit_cells = [s for s in sources if re.search(r"model\.fit\(", s)]
        self.assertTrue(fit_cells)
        self.assertTrue(any("validation_data=(x_valid, y_valid)" in s for s in fit_cells))
        self.assertFalse(any("validation_data=(x_test, y_test)" in s for s in fit_cells))

        # No feature/model selection cell can inspect the test partition; it appears only
        # in the split definition and the final explicitly selected-model assessment.
        for index, source in cells:
            if re.search(r"\bx_test\b|\by_test\b", source):
                self.assertTrue(
                    "x_holdout" in source or "selected_model" in source,
                    f"test partition referenced before final assessment in cell {index}",
                )
        final = [s for s in sources if "selected_model =" in s]
        self.assertEqual(len(final), 1)
        self.assertIn("model.evaluate(x_test, y_test", final[0])

    def test_directory_workflow_keeps_augmentation_out_of_validation(self):
        notebook = load_notebook("plant diseases.ipynb")
        sources = [source for _, source in code_cells(notebook)]
        self.assertTrue(any("train_gen = data_gen.flow_from_dataframe" in s for s in sources))
        self.assertTrue(any("valid_gen = test_gen.flow_from_dataframe" in s for s in sources))
        self.assertTrue(any("validation_data=valid_gen" in s for s in sources))

    def test_paths_and_prediction_inputs_are_explicitly_validated(self):
        pdd_sources = [s for _, s in code_cells(load_notebook("PDD_Classification.ipynb"))]
        self.assertTrue(any("PLANT_DISEASE_DATA_DIR" in s and "FileNotFoundError" in s for s in pdd_sources))
        self.assertTrue(any("if n_classes == 2" in s and "np.column_stack" in s for s in pdd_sources))
        self.assertTrue(any('loss="categorical_crossentropy"' in s for s in pdd_sources))
        self.assertTrue(all("os.chdir" not in s for s in pdd_sources))

        other_sources = [s for _, s in code_cells(load_notebook("plant diseases.ipynb"))]
        self.assertTrue(any("PLANT_DISEASE_TRAIN_DIR" in s and "FileNotFoundError" in s for s in other_sources))
        self.assertTrue(any('source_image.convert("RGB")' in s and "predictions.shape" in s for s in other_sources))
        self.assertTrue(any("def save_model_with_schema(" in s and '"schema_version": 1' in s for s in other_sources))
        self.assertTrue(any("efficientnet_train_gen =" in s and "ImageDataGenerator()" in s for s in other_sources))
        self.assertTrue(any('input_scale="float32 in [0, 255]' in s for s in other_sources))
        self.assertTrue(any("scale_to_unit=False" in s for s in other_sources))
        self.assertTrue(any("resolved_splits" in s and "directories must be disjoint" in s for s in other_sources))
        self.assertTrue(any("valid_gen.class_indices != class_dict" in s for s in other_sources))
        self.assertTrue(any("include_preprocessing=True" in s for s in other_sources))

    def test_generated_artifacts_are_ignored_by_default(self):
        ignore_rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.h5", ignore_rules)
        self.assertIn("*.labels.json", ignore_rules)
        self.assertIn("data/", ignore_rules)


if __name__ == "__main__":
    unittest.main()
