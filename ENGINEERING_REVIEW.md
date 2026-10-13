# Engineering review — Plant Disease Classification

**Repository:** `Abdulrahman181/Classification-of-plant-diseases`  
**Scope:** Python/Notebook/README/dependency audit in the requested repository only  
**Review status:** Static/runtime-structure fixes completed; model training intentionally not run in this sandbox

## Executive summary

The repository contained two notebooks and no Python package or committed dataset. The notebooks were syntactically parseable, but the original workflows were not portable or safely reproducible: one used a machine-specific Windows path and changed the process working directory; the other used a Kaggle path in its original form, treated an unlabeled directory as a dataset, called example files that may not exist, and had incomplete controls around split integrity and reproducibility. Dependency metadata also omitted the notebook and validation tooling.

Both notebooks have been rewritten as clean, top-to-bottom workflows with explicit data-root configuration, clear missing-data errors, deterministic seeds, supported image extensions, stable class ordering, and no stored outputs. A lightweight static validator now checks notebook JSON, AST syntax, stale outputs/execution counts, common non-portable paths, and Ruff `E9/F821` when Ruff is installed.

## Findings and fixes

### Critical/high-impact issues fixed

1. **Non-portable data paths and process state**
   - Removed the Windows absolute path and the Kaggle-specific path assumptions.
   - Removed `os.chdir`; all paths are resolved with `pathlib.Path`.
   - Added `PLANTVILLAGE_ROOT`, `PLANT_DISEASE_OUTPUT_DIR`, and explicit supported layouts.
   - Missing or empty data now raises a targeted `FileNotFoundError`/`ValueError` rather than continuing with undefined or empty variables.

2. **Incorrect image discovery in `PDD_Classification.ipynb`**
   - Replaced the erroneous loop over the current working directory with deterministic traversal of class directories.
   - Added extension filtering, readable-image checks, RGB conversion, resize normalization, and a configurable per-class cap.

3. **Multiclass loss/split correctness**
   - Replaced the custom CNN's `binary_crossentropy` with `categorical_crossentropy` for one-hot multiclass labels.
   - Added a seeded, stratified holdout split and minimum-per-class checks.
   - Added exact-content SHA-256 duplicate detection before the split to prevent obvious duplicate leakage across train/validation.

4. **Generator split integrity in `plant diseases.ipynb`**
   - Added deterministic sorted file discovery, class-set equality checks, path-overlap leakage detection, and `shuffle=False` for validation prediction/evaluation.
   - Removed the unusable unlabeled `image_dataset_from_directory(..., labels=None)` test object.
   - The notebook reports validation metrics only; it does not claim an independent test result when no labelled test split is supplied.

5. **Fragile prediction/plotting/runtime cells**
   - Removed unconditional calls to example filenames that may not exist.
   - Added an opt-in `PLANTVILLAGE_EXAMPLE_IMAGE` prediction path with a clear missing-file error.
   - Used a real heatmap colormap (`"Blues"`) and stable class-index ordering.
   - Saved models to an ignored `outputs/` directory in modern `.keras` format.
   - EfficientNet ImageNet weights are opt-in (`PLANT_DISEASE_IMAGENET_WEIGHTS=1`) to avoid an implicit network download.

6. **Reproducibility and dependency documentation**
   - Added seeded Python/NumPy/TensorFlow setup and deterministic TensorFlow operations where supported.
   - Added `jupyterlab` and `ruff` to `requirements.txt`, with documented minimum versions for the existing numerical, image, plotting, and model stack.
   - Expanded `.gitignore` for outputs, generated images, caches, and local environments.

### Moderate/quality fixes

- Removed redundant imports and stale notebook metadata/output state by rebuilding both notebooks with empty outputs and null execution counts.
- Added consistent environment-variable controls for seed, batch size, epoch counts, output location, and optional transfer learning.
- Added a README covering data layout, execution, validation, evaluation semantics, and the absence of committed data/results.

## Validation performed

The following checks passed after the changes:

- `python3 validate_notebooks.py`
  - JSON structure and notebook format
  - Python AST parse for every code cell
  - no stale outputs or execution counts
  - no detected Windows/Kaggle/Colab absolute paths
  - Ruff `E9/F821` on extracted notebook code
- `ruff check validate_notebooks.py`
- `python3 -m py_compile validate_notebooks.py`
- `python3 -m json.tool PDD_Classification.ipynb`
- `python3 -m json.tool 'plant diseases.ipynb'`
- `git diff --check`

## Deliberately not claimed

The sandbox did not contain the PlantVillage dataset. Heavy runtime dependencies including TensorFlow, scikit-learn, OpenCV, and XGBoost were also unavailable in the current environment. Therefore, no notebook was executed end-to-end, no training/evaluation metric is reported, and no model quality result is claimed. Install `requirements.txt`, provide a licensed local dataset via `PLANTVILLAGE_ROOT`, and run the notebooks to perform the actual experiment.

## Files changed

- `.gitignore`
- `PDD_Classification.ipynb`
- `plant diseases.ipynb`
- `README.md`
- `requirements.txt`
- `validate_notebooks.py`
